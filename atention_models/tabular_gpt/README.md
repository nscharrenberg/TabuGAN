## Tabular Gpt for generating syntehtic data
 > Exemplar on Adult income Dataset

## settings.ini
* The settings for the model and the tokenized documents goes into `settings.ini` file
    * you would need to run `Encoding.ipynb` to fill some of the values of settings.ini parameters (like `context_window` and `vocab_size`)

### Encoding.ipynb
* Convert the csv file into Documents
    * model to not apply attention across documents
    * make (sorted on time) long document for time series data. 
* Pickle tokenizer and column codes to be used in both in training and at inference time.

  ![image](https://github.com/nscharrenberg/TabuGAN/assets/46932291/385be7cd-e61f-43cf-9767-1e8c561a6232)


## Training.ipynb
* Load in the training and eval documents.
* Train the model. NOTE :: We should aim at an extremely low loss value if the dataset has a few skewed float columns.
    * Generating skewed columns are difficult ; not all of the generations lead to a valid input for the decoder
* Save the Model weights

  ![image](https://github.com/nscharrenberg/TabuGAN/assets/46932291/dd9bae52-e9aa-4bfc-bb61-079af37d8719)


## Eval.ipynb
* Load the Model weights and the test loader
* calculate the test loss
* Adjust `n_gens` var for number of generations. (Over Estimate this variable to account for invalid generations)
* Covert the decoded documents (Synthetic Dataset) to a csv file.
* Perform Evaluation Metrics.
