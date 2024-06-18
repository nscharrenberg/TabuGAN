import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import einops

import pickle

from column_code import ColumnTokenizer, FloatTokenizer, CategoricalTokenizer
from tabular_tokenizer import TabularTokenizer


# ┌─────────┐
# │Constants│
# └─────────┘
START = "<start>"
ENDOFTEXT = '<end>'
DELIMITER = '|'
# make a dirctory called encoded_vars
VOCABULARY_PATH = './income_coder.pickle'
FLOAT_COLS = ['age', 'education_num',
              'capital_gain', 'capital_loss', 'hours_per_week']
EXCLUDED_COLS = []


def rename_income_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df.rename(columns={"marital.status": "marital_status",
                       "education.num": "education_num",
                       "capital.gain": "capital_gain",
                       "capital.loss": "capital_loss",
                       "hours.per.week": "hours_per_week",
                       "native.country": "native_country"}, inplace=True)
    return df


def read_dataset(df: pd.DataFrame, rename: bool = True) -> pd.DataFrame:
    """
    Adds <start> and <end> token in the DataFrame
    """
    df.insert(0, "start", "<start>")
    df["end"] = "<end>"
    if rename:
        df = rename_income_dataset(df)
    return df


def tokenize_dataset(df):
    new_msg = """
    _______________________________ 
    / Tokenizing your dataset       \\
    \     > This might take a while /
    ------------------------------- 
           \   ^__^
            \  (oo)\_______
               (__)\       )\/\\
                   ||----w |
                   ||     ||
    """
    print(new_msg)

    # ┌────────────────────────────────┐
    # │    Tokenizing stuff is in utils│
    # └────────────────────────────────┘

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
            cc = FloatTokenizer(
                column, df[[column]], start_id, transform="log")
        else:
            cc = CategoricalTokenizer(column, df[column], start_id)
        column_codes.register(column, cc)

    # Save the encoder and decoder
    with open(VOCABULARY_PATH, 'wb') as handle:
        pickle.dump(column_codes, handle)

    # Load the tokenizer
    tokenizer = TabularTokenizer(VOCABULARY_PATH, special_tokens=[
                                 '\n', ENDOFTEXT], delimiter=DELIMITER)

    # Encode the DataFrame (This is expensive avoid running this)
    encoded_docs = []
    for _, row in df.iterrows():
        encoded_row = []
        for col in columns:
            encoded_value = column_codes.encode(col, str(row[col]))
            encoded_row.extend(encoded_value)
        encoded_docs.append(encoded_row)
    return tokenizer, column_codes, encoded_docs


def decoder(seq, df, column_codes):
    assert len(
        seq) == 38, "Incorrect number of seq length should be 38. See encoded docs"

    columns = [col for col in df.columns if col not in EXCLUDED_COLS]
    sample_encoded_row = seq
    decoded_row = []
    for col, size in zip(columns, column_codes.sizes):
        token_ids = sample_encoded_row[:size]
        decoded_value = column_codes.decode(col, token_ids)
        decoded_row.append(decoded_value)
        sample_encoded_row = sample_encoded_row[size:]

    return decoded_row


def main():
    pass


if __name__ == "__main__":
    main()
