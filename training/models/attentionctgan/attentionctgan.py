"""Attention CTGAN module."""

import warnings

import numpy as np
import pandas as pd
import torch
import einops
from torch import optim
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.kl import kl_divergence
from torch.nn import BatchNorm1d, Dropout, LeakyReLU, Linear, Module, ReLU, Sequential, functional, Parameter, ModuleList, LayerNorm, Embedding
from tqdm import tqdm

from models.ctgan.base import BaseSynthesizer, random_state
from models.attentionctgan.data.data_sampler import DataSampler
from models.attentionctgan.data.data_transformer import DataTransformer


class SingleHead(Module):
    """
    Implements a single head of attention (unmasked)
    """

    def __init__(self, R, head_size, manual_seed=123):
        super().__init__()
        # single head
        self.head_size = head_size
        self.scale = 1 / torch.sqrt(torch.tensor(self.head_size))
        self.R = torch.tensor(R)
        self.Q = Parameter(torch.randn(self.R, head_size))
        self.K = Parameter(torch.randn(self.R, head_size))
        self.V = Parameter(torch.randn(self.R, head_size))

    def forward(self, x):
        q = x @ self.Q  # (B,T,R) * (R,h) => (B,T,R) x (B,R,h) => (B,T,h)
        k = x @ self.K
        v = x @ self.V

        comm = q @ einops.rearrange(
            k, "B T h -> B h T"
        )  # (B,T,h) @ (B,h,T) => B,T,T @ each tokens interaction with each token would be T,T
        att1 = functional.softmax(comm * self.scale, dim=2)  # along tokens head
        attention = att1 @ v  # B,T,T @ B,T,h => B,T,h
        return attention


class Multihead(Module):
    def __init__(self, R, n_heads, seed=123):
        self.R = R
        assert R % n_heads == 0, "n_heads is not divisible by R"
        self.head_size = self.R // n_heads
        self.n_heads = n_heads

        super().__init__()
        self.multiheads = ModuleList(
            [SingleHead(self.R, self.head_size) for _ in range(self.n_heads)]
        )

    def forward(self, x):
        return torch.cat([head(x) for head in self.multiheads], dim=2)


class Transformer(Module):
    def __init__(self, context_window, n_embed, n_heads, vocab_length, transformer_blocks,device):
        super().__init__()
        self.context_window = context_window
        self.n_embed = n_embed
        self.n_heads = n_heads
        self.n_vocab = vocab_length
        self.transformer_blocks = transformer_blocks

        #==== Token (C) and positional Embedding (pe)
        self.C = Embedding(self.n_vocab,self.n_embed,device=device) # token embedding
        self.pe = Parameter(torch.randn(1,self.n_embed)).to(device) # position embedding

        #===== Transformer Blocks
        self.multi_heads = Multihead(R=self.n_embed, n_heads=self.n_heads)
        self.layer_norm = LayerNorm(self.n_embed) #at the end of the transformer block

        #==== MLP Head
        self.mlp_head0 = Linear(self.context_window * self.n_embed , self.n_embed)
        self.mlp_head1 = Linear(self.n_embed,self.n_vocab) # logits



    def forward(self,x):
        emb = self.C(x) # token emb
        emb += self.pe # add pos emb

        # write a loop later : 2 blocks
        attn = self.multi_heads(emb)
        # skip connection
        skip = torch.nn.functional.leaky_relu(attn) + emb
        normed = self.layer_norm(skip)# layer norm and skip connection

        # flatten
        normed = einops.rearrange(normed,"b t e -> b (t e)")
        x = self.mlp_head0(normed)
        logits = self.mlp_head1(x) # logits

        return logits

    def get_embedding(self,x):
        emb = self.C(x) # token emb
        emb += self.pe # add pos emb

        # write a loop later : 2 blocks
        attn = self.multi_heads(emb)
        # skip connection
        skip = torch.nn.functional.leaky_relu(attn) + emb
        normed = self.layer_norm(skip)# layer norm and skip connection

        # flatten
        normed = einops.rearrange(normed,"b t e -> b (t e)")
        x = functional.tanh(self.mlp_head0(normed))
        return x

    def calculate_loss(self,x,y):
        logits = self(x)
        return logits , functional.cross_entropy(logits,y)

    def generate(self,seed = [0] * 38,verbose=False):
        generation = list()
        i = 0
        while True:
            if i > 50:
                # clearly untrained
                break
            if verbose:
                print(seed)
            logits = self(torch.tensor(seed).view(1,-1))
            probs = functional.softmax(logits,dim=1)
            prediction = torch.multinomial(probs,num_samples=1)
            generation.append(prediction.item())
            if prediction.item() == 21976:
                break
            seed = seed[1:] + [prediction.item()]
            if verbose:
                print(prediction.item())
            i = i+1
        return generation

class ConditioningAugmentation(Module):
    def __init__(self, input_embedding,outputdim):
        super(ConditioningAugmentation, self).__init__()
        self.net = Sequential(
            Linear(input_embedding, 256),
            ReLU(),
        )
        self.mean = Linear(256, outputdim)
        self.var = Linear(256, outputdim)

    def forward(self, x):
        x = self.net(x)
        return self.mean(x),functional.sigmoid(self.var(x))

class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real_data, fake_data, device='cpu', pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real_data.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real_data.size(1))
        alpha = alpha.view(-1, real_data.size(1))

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates, inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]

        gradients_view = gradients.view(-1, pac * real_data.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class GeneratorWithAttention(Module):
    """Generator with Attention for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim,transformer_embedding):
        super(GeneratorWithAttention, self).__init__()
        dim = list(generator_dim)
        self.net1 = Linear(embedding_dim, dim[0])
        self.mapper = Linear(transformer_embedding, dim[0])
        self.net2 = Sequential(
            Residual(dim[0]*2, dim[1]),
            Linear(dim[0]*2+dim[1], data_dim),
        )

    def forward(self, input_,embedding):
        """Apply the Generator to the `input_`."""
        data = self.net1(input_)
        translated_embedding = self.mapper(embedding)
        attention = functional.softmax(translated_embedding*data,dim=1)
        x = torch.concat((data,attention),dim=1)
        return self.net2(x)

class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class AttentionCTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.

    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.

    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
        vocabulary_length (int):
                    Length of the vocabulary for the transformer. Defaults to 21979.
                context_window (int):
                    Size of the context window for the transformer. Defaults to 38.
                transformer_embedding_length (int):
                    Size of the embedding layer in the transformer. Defaults to 992.
                num_heads (int):
                    Number of attention heads in the transformer. Defaults to 31.
                transformer_blocks (int):
                    Number of transformer blocks in the transformer. Defaults to 2.
    """

    def __init__(self, embedding_dim=128, generator_dim=(256, 256), discriminator_dim=(256, 256),
                 generator_lr=2e-4, generator_decay=1e-6, enable_generator_attention= False,
                 discriminator_lr=2e-4, discriminator_decay=1e-6, batch_size=500, discriminator_steps=1,
                 log_frequency=True, verbose=False, epochs=300, pac=10, cuda=True,
                 vocabulary_length=21979, context_window=38, transformer_embedding_length=992,
                 num_heads=31, transformer_blocks=2,transformer_model_path = "transformer_model.pth",
                 conditioning_augmentation_dim = 32, conditioning_augmentation_lr = 1e-3, enable_conditioning_augmentation = True):
        assert batch_size % 2 == 0
        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim
        self.enable_generator_attention = enable_generator_attention
        self.enable_conditioning_augmentation = enable_conditioning_augmentation

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
        self._attention_model = None

        self.vocabulary_length = vocabulary_length
        self.context_window = context_window
        self.transformer_embedding_length = transformer_embedding_length
        self.num_heads = num_heads
        self.transformer_blocks = transformer_blocks
        self.transformer_model_path = transformer_model_path
        self.conditioning_augmentation_dim = conditioning_augmentation_dim
        self.conditioning_augmentation_lr = conditioning_augmentation_lr

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = 'cpu'
        elif isinstance(cuda, str):
            device = cuda
        else:
            device = 'cuda'

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None
        self.loss_values = None
        self._attention_model = None

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.

        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing

        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.

        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        for _ in range(10):
            transformed = functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)
            if not torch.isnan(transformed).any():
                return transformed

        raise ValueError('gumbel_softmax returning NaN.')

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == 'tanh':
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == 'softmax':
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(f'Unexpected activation function {span_info.activation_fn}.')

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != 'softmax':
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction='none'
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, train_data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``train_data``.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(train_data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(train_data.columns)
        elif isinstance(train_data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= train_data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError('``train_data`` should be either pd.DataFrame or np.array.')

        if invalid_columns:
            raise ValueError(f'Invalid columns found: {invalid_columns}')

    @random_state
    def fit(self, train_data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.

        Args:
            train_data (numpy.ndarray or pandas.DataFrame):
                Training Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``train_data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(train_data, discrete_columns)

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                ('`epochs` argument in `fit` method has been deprecated and will be removed '
                 'in a future version. Please pass `epochs` to the constructor instead'),
                DeprecationWarning
            )
        self._attention_model = Transformer(
            context_window=self.context_window,
            n_embed=self.transformer_embedding_length,
            n_heads=self.num_heads,
            vocab_length=self.vocabulary_length,
            transformer_blocks=self.transformer_blocks,
            device=self._device,
        ).to(self._device)
        self._attention_model.load_state_dict(torch.load(self.transformer_model_path,map_location=self._device),strict=False)
        self._attention_model.eval()

        self._transformer = DataTransformer()
        self._transformer.fit(train_data, discrete_columns)

        train_data = self._transformer.transform(train_data)
        input_embedding = torch.tensor(self._transformer.get_input_embedding()).to(self._device)

        self._data_sampler = DataSampler(
            train_data,
            input_embedding,
            self._transformer.output_info_list,
            self._log_frequency,
            self.conditioning_augmentation_dim,
            self.enable_conditioning_augmentation)

        data_dim = self._transformer.output_dimensions

        self.conditioning_augmentation = ConditioningAugmentation(self.transformer_embedding_length,self.conditioning_augmentation_dim)

        if self.enable_generator_attention:
            self._generator = GeneratorWithAttention(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,
                self.transformer_embedding_length
            ).to(self._device)
        else:
            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim
            ).to(self._device)

        discriminator = Discriminator(
            data_dim + self._data_sampler.dim_cond_vec(),
            self._discriminator_dim,
            pac=self.pac
        ).to(self._device)

        optimizerG = optim.Adam(
            self._generator.parameters(), lr=self._generator_lr, betas=(0.5, 0.9),
            weight_decay=self._generator_decay
        )

        optimizerD = optim.Adam(
            discriminator.parameters(), lr=self._discriminator_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        optimizerCA = optim.Adam(
            self.conditioning_augmentation.parameters(), lr=self.conditioning_augmentation_lr,
            betas=(0.5, 0.9), weight_decay=self._discriminator_decay
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(columns=['Epoch', 'Generator Loss', 'Distriminator Loss'])

        epoch_iterator = tqdm(range(epochs), disable=(not self._verbose))
        if self._verbose:
            description = 'Gen. ({gen:.2f}) | Discrim. ({dis:.2f})'
            epoch_iterator.set_description(description.format(gen=0, dis=0))

        steps_per_epoch = max(len(train_data) // self._batch_size, 1)
        for i in epoch_iterator:
            for id_ in range(steps_per_epoch):

                for n in range(self._discriminator_steps):
                    fakez = torch.normal(mean=mean, std=std)

                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    if condvec is None:
                        c1, m1, col, opt = None, None, None, None
                        real,input_embedding = self._data_sampler.sample_data(
                            train_data, self._batch_size, col, opt)
                    else:
                        c1, m1, col, opt = condvec
                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)
                        real,input_embedding = self._data_sampler.sample_data(
                            train_data, self._batch_size, col[perm], opt[perm])

                        with torch.no_grad():
                            output_embedding = self._attention_model.get_embedding(input_embedding)
                        if self.enable_conditioning_augmentation:
                            with torch.no_grad():
                                mean_ca,var_ca = self.conditioning_augmentation(output_embedding.cpu())
                            mvn = MultivariateNormal(mean_ca, var_ca.unsqueeze(1)*torch.eye(self.conditioning_augmentation_dim))
                            sampled_embedding = mvn.sample().to(self._device)
                            fakez = torch.cat([fakez, sampled_embedding], dim=1)
                            c2 = sampled_embedding[perm]
                        else:
                            fakez = torch.cat([fakez, c1], dim=1)
                            c2 = c1[perm]

                    if self.enable_generator_attention:
                        fake = self._generator(fakez,output_embedding)
                    else:
                        fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    real = torch.from_numpy(real.astype('float32')).to(self._device)

                    if c1 is not None:
                        if self.enable_conditioning_augmentation:
                            fake_cat = torch.cat([fakeact, sampled_embedding], dim=1)
                        else:
                            fake_cat = torch.cat([fakeact, c1], dim=1)
                        real_cat = torch.cat([real, c2], dim=1)
                    else:
                        real_cat = real
                        fake_cat = fakeact

                    y_fake = discriminator(fake_cat)
                    y_real = discriminator(real_cat)

                    pen = discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac)
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad(set_to_none=False)
                    pen.backward(retain_graph=True)
                    loss_d.backward()
                    optimizerD.step()

                fakez = torch.normal(mean=mean, std=std)
                condvec = self._data_sampler.sample_condvec(self._batch_size)

                if condvec is None:
                    c1, m1, col, opt = None, None, None, None
                else:
                    c1, m1, col, opt = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    m1 = torch.from_numpy(m1).to(self._device)
                    perm = np.arange(self._batch_size)
                    np.random.shuffle(perm)
                    _, input_embedding = self._data_sampler.sample_data(
                        train_data, self._batch_size, col[perm], opt[perm]
                    )
                    with torch.no_grad():
                        output_embedding = self._attention_model.get_embedding(
                            input_embedding
                        )
                    if self.enable_conditioning_augmentation:
                        optimizerCA.zero_grad(set_to_none=False)
                        mean_ca,var_ca = self.conditioning_augmentation(output_embedding.cpu())
                        mvn = MultivariateNormal(mean_ca, var_ca.unsqueeze(1)*torch.eye(self.conditioning_augmentation_dim))
                        sampled_embedding = mvn.sample().to(self._device)
                        fakez = torch.cat([fakez, sampled_embedding], dim=1)
                    else:
                        fakez = torch.cat([fakez, c1], dim=1)

                if self.enable_generator_attention:
                    fake = self._generator(fakez,output_embedding)
                else:
                    fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)

                if c1 is not None:
                    if self.enable_conditioning_augmentation:
                        y_fake = discriminator(torch.cat([fakeact, sampled_embedding], dim=1))
                    else:
                        y_fake = discriminator(torch.cat([fakeact, c1], dim=1))
                else:
                    y_fake = discriminator(fakeact)

                if condvec is None:
                    cross_entropy = 0
                else:
                    cross_entropy = self._cond_loss(fake, c1, m1)

                
                loss_g = -torch.mean(y_fake) + cross_entropy
                if self.enable_conditioning_augmentation:
                    loss_kldivergence = kl_divergence(mvn,MultivariateNormal(torch.zeros(self.conditioning_augmentation_dim), torch.eye(self.conditioning_augmentation_dim))).mean()
                    loss_g += loss_kldivergence

                optimizerG.zero_grad(set_to_none=False)
                loss_g.backward()
                optimizerG.step()
                optimizerCA.step()

            generator_loss = loss_g.detach().cpu().item()
            discriminator_loss = loss_d.detach().cpu().item()

            epoch_loss_df = pd.DataFrame({
                'Epoch': [i],
                'Generator Loss': [generator_loss],
                'Discriminator Loss': [discriminator_loss]
            })
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            if self._verbose:
                epoch_iterator.set_description(
                    description.format(gen=generator_loss, dis=discriminator_loss)
                )

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.

        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.

        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.

        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value)
            global_condition_vec = self._data_sampler.generate_cond_from_condition_column_info(
                condition_info, self._batch_size)
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec,input_embedding_batch = self._data_sampler.sample_original_condvec(self._batch_size)

            if input_embedding_batch is None or condvec is None:
                pass
            else:
                with torch.no_grad():
                    output_embedding = self._attention_model.get_embedding(input_embedding_batch)
                if self.enable_conditioning_augmentation:
                    mean_ca,var_ca = self.conditioning_augmentation(output_embedding.cpu())
                    mvn = MultivariateNormal(mean_ca, var_ca.unsqueeze(1)*torch.eye(self.conditioning_augmentation_dim))
                    sampled_embedding = mvn.sample().to(self._device)
                    fakez = torch.cat([fakez, sampled_embedding], dim=1)
                else:
                    c1 = condvec
                    c1 = torch.from_numpy(c1).to(self._device)
                    fakez = torch.cat([fakez, c1], dim=1)

            if self.enable_generator_attention:
                fake = self._generator(fakez,output_embedding)
            else:
                fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
