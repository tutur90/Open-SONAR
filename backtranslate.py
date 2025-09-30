#!/usr/bin/env python

import logging
import os
import sys
from datasets import load_dataset, Dataset
import numpy as np
import polars as pl
import pandas as pd
from tqdm import tqdm

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    M2M100Tokenizer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from sonar.args import ModelArguments, DataTrainingArguments
from sonar.preprocessor import NllbPreprocessor
from sonar.tokenizerV2 import NllbTokenizerFast
from sonar.trainer import SONARTrainer
from data.scripts.nllb_langs import code_mapping

check_min_version("4.53.0")
require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
        if "local_rank" in sys.argv:
            training_args.local_rank = sys.argv[sys.argv.index("local_rank")+1]
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    set_seed(training_args.seed)
    
    #### HERE YOU CAN BUILD YOUR OWN DATASET ####
    # Should have columns: id, source.text, source.nllb_code, target.text, target.nllb_code, source.len

    # Load dataset
    raw_datasets = load_dataset(
        data_args.dataset_name,
        data_args.dataset_config_name,
        cache_dir=model_args.cache_dir,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    df = raw_datasets["train"].to_polars()
    
    # Prepare data
    df = df.with_columns(
        (pl.col("iso_639_3") + "_" + pl.col("iso_15924")).alias("nllb_code"),
        pl.lit(None).cast(pl.Float32).alias("lid"),
    ).drop(["iso_639_3", "iso_15924"])
    
    english_data = df.filter(pl.col("nllb_code") == "eng_Latn")

    existing_codes = set(df["nllb_code"].unique().to_list())
    available_codes = set(code_mapping.values())
    missing_codes = available_codes - existing_codes
    
    data = pl.DataFrame()
    
    for code in missing_codes:
        subset = english_data.with_columns(
            pl.lit(code).alias("target.nllb_code"),
            pl.col("id").alias("target.text").cast(pl.Utf8),
            pl.col("text").alias("source.text"),
            pl.col("nllb_code").alias("source.nllb_code"),
            pl.col("text").str.len_chars().alias("source.len"),
        ).select([
            "id", "source.text", "source.nllb_code", "target.text", "target.nllb_code", "source.len"
        ])
        data = pl.concat([data, subset])
        
    data = data.sort(
        by=["source.len", "source.nllb_code", "target.nllb_code", "id"],
        descending=True
    )

    ds = Dataset.from_polars(data)
    
    #### END OF DATASET BUILDING ####

    # Load model and tokenizer
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    tokenizer = NllbTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    model.mse_ratio = model_args.mse_ratio
    
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    try:
        data_args.max_source_length = int(data_args.max_source_length)
    except:
        logger.warning("Failed to convert max_source_length to int, using default value.", data_args.max_source_length)

    if (
        hasattr(model.config, "max_position_embeddings")
        and not hasattr(model.config, "relative_attention_max_distance")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        raise ValueError(
            f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has"
            f" {model.config.max_position_embeddings} position encodings. Consider either reducing"
            f" `--max_source_length` to {model.config.max_position_embeddings} or using a model with larger position "
            "embeddings"
        )

    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    print("Processing dataset...")

    eval_processor = NllbPreprocessor(
        tokenizer=tokenizer,
        data_args=data_args,
        padding=padding,
        mode="predict" if training_args.do_predict else "eval",
    )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        predict_dataset = ds
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                eval_processor,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=os.path.join(training_args.output_dir, f"predict_dataset.arrow"),
                desc="Running tokenizer on prediction dataset",
                remove_columns=['source.text', 'source.nllb_code', 'target.text', 'target.nllb_code', ]
            ).remove_columns(
            ['source.len','labels_len',]
            )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding="longest",
        max_length=data_args.max_source_length,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
    )

    def compute_metrics(eval_preds):
        preds, labels, inputs, loss = eval_preds
        
        source_langs_ids = inputs[:, 0]
        target_langs_ids = labels[:, 0]

        if isinstance(preds, tuple):
            preds = preds[0]
            
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        results = {}
        list_ids = target_langs_ids.tolist()

        df = pl.DataFrame({"id": list_ids, "labels": decoded_labels})
        df = pd.DataFrame({
            "source_langs": [tokenizer.convert_ids_to_tokens(int(src)) for src in source_langs_ids.tolist()],
            "target_langs": [tokenizer.convert_ids_to_tokens(int(tgt)) for tgt in target_langs_ids.tolist()],
            "preds": decoded_preds,
            "labels": decoded_labels,
        })
        df.to_csv("data/backtranslations.csv", index=False)
        return results

    trainer = SONARTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams

    if training_args.do_predict:
        logger.info("*** Predict ***")
        
        num_beams = data_args.predict_num_beams if data_args.predict_num_beams is not None else num_beams
        print("Predicting with num_beams = ", num_beams, " and max_length = ", max_length)

        predict_results = trainer.predict(
            predict_dataset, 
            metric_key_prefix="predict", 
            max_length=max_length, 
            num_beams=num_beams, 
            do_sample=False, 
            no_repeat_ngram_size=num_beams-1, 
            renormalize_logits=True, 
            min_length=1,
        )
        metrics = predict_results.metrics

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    main()


if __name__ == "__main__":
    main()