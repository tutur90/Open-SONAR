import itertools
import pandas as pd
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

import os
import sys
from pathlib import Path


def process_df(df, nllb_codes):
    processed_df = pd.DataFrame(columns=["id", "source.nllb.code", "target.nllb.code", "source.text", "target.text"])

    prod = list(itertools.product(nllb_codes, repeat=2))

    print(f"Total language pairs to process: {len(prod)}")
    
    prod = sorted(prod, key=lambda x: (x[0], x[1]))

    for (src, tgt) in prod:
        if src == "eng_Latn" or tgt == "eng_Latn" or src == tgt:
            print(f"Processing pair: {src} - {tgt}")
            
            src_df = df[df["nllb_code"] == src]
            tgt_df = df[df["nllb_code"] == tgt]

            if src_df.empty or tgt_df.empty:
                print(f"Skipping pair {src} - {tgt} due to empty DataFrame.")
                continue

            merged_df = pd.merge(src_df, tgt_df, on="id", suffixes=("_source", "_target"))
            
            processed_df = pd.concat([processed_df, merged_df], ignore_index=True)
    
    processed_df = processed_df[["id", "nllb_code_source", "nllb_code_target", "sentence_source", "sentence_target"]]
    processed_df.columns = ["id", "source.nllb_code", "target.nllb_code", "source.text", "target.text"]
    
    return processed_df


if __name__ == "__main__":
    
    datasets = load_dataset("tutur90/flores200_xsim")
    
    print(datasets)
    
    datasets["dev"].to_parquet("datasets/flores200/dev.parquet")
    datasets["devtest"].to_parquet("datasets/flores200/devtest.parquet")

    dev_df = pd.read_parquet("datasets/flores200/dev.parquet")
    devtest_df = pd.read_parquet("datasets/flores200/devtest.parquet")

    devtest_df = devtest_df[devtest_df["id"] >= 0]

    nllb_codes = dev_df["nllb_code"].unique()


    processed_df = process_df(dev_df, nllb_codes)

    print(f"Total rows in processed DataFrame: {len(processed_df)}")
    
    processed_df = processed_df.drop_duplicates(subset=["source.nllb_code", "target.nllb_code", "target.text"])


    processed_df.to_parquet("datasets/flores200/dev_paired.parquet")

    processed_df = processed_df.drop_duplicates(subset=["source.nllb_code", "target.nllb_code", "target.text"])

    processed_df_test = process_df(devtest_df, nllb_codes)
    processed_df_test.to_parquet("datasets/flores200/devtest_paired.parquet")
