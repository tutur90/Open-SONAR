import itertools
import pandas as pd
from datasets import load_dataset, load_from_disk, DatasetDict, Dataset

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

print(sys.path)

from data.text.nllb_langs import code_mapping, FLORES_LANGUAGES



def get_df(split="dev", augmented=None):


    dfs = []
    
    if augmented:
        pdf_aug = pd.read_csv(augmented, sep="\t", names=["sentence"])
        
        pdf_aug["nllb_code"] = "eng_Latn"
        
        pdf_aug["id"] = -pdf_aug.index-1

        dfs.append(pdf_aug)


    for lang in FLORES_LANGUAGES:
        if lang not in code_mapping.values() and lang != "sat_Olck":
            print(f"Language {lang} not found in code_mapping, skipping.")
            continue
        ds = load_dataset("facebook/flores", lang)
        
        
        df = ds[split].to_pandas()
        
        df = df[["sentence", "id"]]
        
        df["nllb_code"] = lang if lang != "sat_Olck"  else "sat_Beng"  # Use sat_Beng for sat_Olck in NLLB
        
        dfs.append(df)
        
        print(f"Language {lang} has {len(df)} sentences.")


    df_full = pd.concat(dfs, ignore_index=True)

    return df_full



# if __name__ == "__main__":
    
    
    
dev_df = get_df(split="dev")
dev_df.to_parquet("datasets/flores_200_dev.parquet")

# # print(dev_df.drop_duplicates(["nllb_code", "sentence"]).groupby("nllb_code").size().reset_index(name='counts').sort_values(by="counts", ascending=False))

# # print(dev_df.drop_duplicates(["nllb_code", "id"]).groupby("nllb_code").size().sort_values(by="counts", ascending=False))

devtest_df = get_df(split="devtest", augmented="datasets/flores200/devtest_augmented/eng_Latn_augmented.devtest")
devtest_df.to_parquet("datasets/flores_200_devtest.parquet")


dev_df = pd.read_parquet("datasets/flores_200_dev.parquet")
devtest_df = pd.read_parquet("datasets/flores_200_devtest.parquet")

devtest_df = devtest_df[devtest_df["id"] >= 0]

nllb_codes = code_mapping.values()

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

# # nllb_codes = [ "eng_Latn", "fra_Latn", "spa_Latn", "deu_Latn", "ita_Latn", "por_Latn",]

processed_df = process_df(dev_df, nllb_codes)

print(f"Total rows in processed DataFrame: {len(processed_df)}")

# print(f"Processed DataFrame shape: {processed_df.shape}")

processed_df.to_parquet("datasets/flores_200_dev_paired.parquet")

# print(processed_df.groupby(["source.nllb_code", "target.nllb_code"]).size().reset_index(name='counts').sort_values(by="counts", ascending=False))

processed_df = processed_df.drop_duplicates(subset=["source.nllb_code", "target.nllb_code", "target.text"])

# print(processed_df.groupby(["source.nllb_code", "target.nllb_code"]).size().reset_index(name='counts').sort_values(by="counts", ascending=False))

processed_df_test = process_df(devtest_df, nllb_codes)
processed_df_test.to_parquet("datasets/flores_200_devtest_paired.parquet")
