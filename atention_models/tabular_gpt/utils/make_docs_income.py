# dont do this for long documents

# so the script looks like:
import pandas as pd
import pickle
from utils.column_code import ColumnTokenizer, FloatTokenizer, CategoricalTokenizer
from utils.tabular_tokenizer import TabularTokenizer

# Constants
START = "<start>"
ENDOFTEXT = '<end>'
DELIMITER = '|'
VOCABULARY_PATH = 'income_coder.pickle'
FLOAT_COLS = ['age', 'education_num',
              'capital_gain', 'capital_loss', 'hours_per_week']
EXCLUDED_COLS = []

# Sample DataFrame (replace with your actual data)
# df = pd.read_csv("../../datasets/income/adult.csv")  # Assuming the dataset path is correct

# Fill missing values
df = df.fillna('?')

# Initialize ColumnTokenizer
column_codes = ColumnTokenizer()
beg = 0
cc = None
columns = [col for col in df.columns if col not in EXCLUDED_COLS]

# Register columns
for column in columns:
    start_id = beg if cc is None else cc.end_id
    if column in FLOAT_COLS:
        cc = FloatTokenizer(column, df[[column]], start_id, transform="log")
    else:
        cc = CategoricalTokenizer(column, df[column], start_id)
    column_codes.register(column, cc)

# Save the encoder and decoder
with open(VOCABULARY_PATH, 'wb') as handle:
    pickle.dump(column_codes, handle)

# Load the tokenizer
tokenizer = TabularTokenizer(VOCABULARY_PATH, special_tokens=[
                             '\n', ENDOFTEXT], delimiter=DELIMITER)

# Encode the DataFrame
encoded_docs = []
for _, row in df.iterrows():
    encoded_row = []
    for col in columns:
        encoded_value = column_codes.encode(col, str(row[col]))
        encoded_row.extend(encoded_value)
    encoded_docs.append(encoded_row)

# Decode a sample encoded row
sample_encoded_row = encoded_docs[0]
decoded_row = []
for col, size in zip(columns, column_codes.sizes):
    token_ids = sample_encoded_row[:size]
    decoded_value = column_codes.decode(col, token_ids)
    decoded_row.append(decoded_value)
    sample_encoded_row = sample_encoded_row[size:]

# Display the encoded documents and decoded row
# Show the first 3 encoded documents
print(f"Encoded Documents (first 3): {encoded_docs[:3]}")
# Show the decoded values of the first encoded row
print(f"Decoded Row: {decoded_row}")
