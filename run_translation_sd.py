#!/usr/bin/env python
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
from math import e
import os
import sys

import datasets
import evaluate
import numpy as np
from datasets import load_dataset

from statistics import mean

from torch.utils.data import DataLoader

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)



from datasets import Dataset


from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from open_sonar.text.args import ModelArguments, DataTrainingArguments
from open_sonar.preprocessor import NllbPreprocessor
from open_sonar.tokenizerV2 import NllbTokenizerFast
from open_sonar.text.models.modeling_sonar import SONARForText2Text
from open_sonar.text.models.sd_model import SONARSDForConditionalGeneration
from open_sonar.text.collator import DataCollator, DAEProcessor
from open_sonar.trainer import SONARTrainer
from open_sonar.xsim import XSim
from open_sonar.text.collator import DataCollator
from open_sonar.optim import get_inverse_sqrt_schedule

from sacrebleu import BLEU, corpus_bleu

import os


import pandas as pd


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.53.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif sys.argv[-1].endswith(".yaml"):
        model_args, data_args, training_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[-1]))
        if "local_rank" in sys.argv:
            training_args.local_rank = sys.argv[sys.argv.index("local_rank")+1]
        
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if data_args.total_batch_size is not None:
        training_args.gradient_accumulation_steps = data_args.total_batch_size // (training_args.per_device_train_batch_size * int(os.environ.get("WORLD_SIZE", 1))) + 1
        logger.warning(f"Setting gradient_accumulation_steps to {training_args.gradient_accumulation_steps} to match total_batch_size {data_args.total_batch_size}")

    # training_args.accelerator_config.non_blocking = True # For multimodes
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_translation", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    



    # Detecting last checkpoint.
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        raw_datasets = {}
        
        if data_args.train_file is not None:
            data_files = {}
            data_files["train"] = data_args.train_file.replace("#", "")
            # print("data_files", data_files)
            extension = data_args.train_file.split(".")[-1]
            train_dataset = load_dataset(extension, data_files=data_files, streaming=data_args.dataset_streaming)["train"]
            raw_datasets["train"] = train_dataset
        if data_args.validation_file is not None:
            data_files = {}
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
            eval_dataset = load_dataset(extension, data_files=data_files, 
                                        # download_mode='force_redownload' if data_args.overwrite_cache else 'reuse_dataset_if_exists'
                                        )["validation"]
            raw_datasets["validation"] = eval_dataset
            
            
        if data_args.test_file is not None:
            data_files = {}
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
            predict_dataset = load_dataset(extension, data_files=data_files,
                                        #    download_mode='force_redownload' if data_args.overwrite_cache else 'reuse_dataset_if_exists'
                                           )["test"]
            raw_datasets["test"] = predict_dataset
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    config.mse_ratio = model_args.mse_ratio

    tokenizer = NllbTokenizerFast.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    


    config.mse_ratio = model_args.mse_ratio
    



    if model_args.semantic_decompression > 1 and not hasattr(config, "semantic_decompression"):
        config.semantic_decompression = model_args.semantic_decompression

    model = SONARForText2Text.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )

    # logger.info(f"Encoder num parameters: {model.get_encoder().num_parameters()/1e6:.2f}M")
    # logger.info(f"Decoder num parameters: {model.get_decoder().num_parameters()/1e6:.2f}M")
    
    
    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")


    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # For translation we set the codes of our source and target languages (only useful for mBART, the others will
    # ignore those attributes).
    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert data_args.target_lang is not None and data_args.source_lang is not None, (
            f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --source_lang and "
            "--target_lang arguments."
        )

        tokenizer.src_lang = data_args.source_lang
        tokenizer.tgt_lang = data_args.target_lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id


    # Check the whether the source target length fits in the model, if it has absolute positional embeddings
    
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

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for "
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    buffer_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * data_args.sorting_buffer_ratio
    
    print("buffer_size", buffer_size)

    processor = NllbPreprocessor(
        tokenizer=tokenizer,
        data_args=data_args,
        padding=padding,
        sort_buffer_size=buffer_size,
    )
    
    noiser = DAEProcessor(
        tokenizer=tokenizer,
        key_input="target_ids",
        
        mask_ratio=0.3,
        insert_ratio=0.1,
        permute_sentence_ratio=0.2,
        random_ratio=0.0,
    )
    
    # noiser = lambda x: x

    cache_dir = "./.cache"

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        # train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            column_names = train_dataset.column_names
            if isinstance(train_dataset, datasets.Dataset):
                
                train_dataset = train_dataset.map(
                    processor,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    cache_file_name=os.path.join(cache_dir, "data", f"train_dataset.arrow"),
                    desc="Running tokenizer on train dataset",
                )
                
                
            elif isinstance(train_dataset, datasets.IterableDataset):
                # train_dataset = split_dataset_by_node(train_dataset, rank=int(os.environ["RANK"]), world_size=int(os.environ["WORLD_SIZE"]))
                train_dataset = train_dataset.shuffle(buffer_size=buffer_size*data_args.shuffle_buffer_size, seed=training_args.seed
                                                      ).map(
                    processor,
                    remove_columns=column_names,
                    batched=True,
                    batch_size=buffer_size*2,
                )
                                                      

    eval_processor = NllbPreprocessor(
        tokenizer=tokenizer,
        data_args=data_args,
        padding=padding,
        mode="predict" if training_args.do_predict else "eval",
    )
    


    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            column_names = eval_dataset.column_names
            eval_dataset = eval_dataset.map(
                eval_processor   ,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=os.path.join(cache_dir, "data", f"eval_dataset.arrow"),
                desc="Running tokenizer on validation dataset",
                
            ).remove_columns("labels_len")
            

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        
        
        
        
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                eval_processor,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                cache_file_name=os.path.join(cache_dir, "data", f"predict_dataset.arrow"),
                desc="Running tokenizer on prediction dataset",
            ).sort("labels_len", reverse=True)

            print("first 10 predict dataset", predict_dataset[:10].keys(), predict_dataset[:10]["labels_len"])

            predict_dataset = predict_dataset.remove_columns("labels_len")


    data_collator = DataCollator(
        tokenizer=tokenizer,
        dae_ratio=model_args.dae_ratio,
        dae_processor=noiser,
        padding=padding,
        max_src_length=data_args.max_source_length,
        max_target_length=max_target_length,
        label_pad_token_id=-100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id,
        pad_to_max_length=False,
        predict_with_generate=training_args.predict_with_generate,
    )

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        return preds

    def compute_spbleu(eval_preds):
        preds, labels, inputs, loss = eval_preds
        
        # print("loss", loss, "inputs", inputs)

        
        source_langs_ids = inputs[:, 0]
        target_langs_ids = labels[:, 0]

        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = postprocess_text(tokenizer.batch_decode(preds, skip_special_tokens=True))
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = postprocess_text(tokenizer.batch_decode(labels, skip_special_tokens=True))
        
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)
        decoded_inputs = postprocess_text(tokenizer.batch_decode(inputs, skip_special_tokens=True))

        results = {}

        
        df = pd.DataFrame({
            "source_langs": [tokenizer.convert_ids_to_tokens(int(src)) for src in source_langs_ids.tolist()],
            "target_langs": [tokenizer.convert_ids_to_tokens(int(tgt)) for tgt in target_langs_ids.tolist()],
            "inputs": decoded_inputs,
            "preds": decoded_preds,
            "labels": decoded_labels,
        })

        df = df.drop_duplicates(subset=["source_langs", "target_langs", "labels", "inputs"])   

        df.to_csv(os.path.join(training_args.output_dir, "data", f"generated_predictions.csv"), index=False)
        

        unique_pairs = df.groupby(["source_langs", "target_langs"]).size().reset_index(name='counts').values.tolist()
        

        for src_lang, tgt_lang, n in unique_pairs:
            pairs = df[(df["source_langs"] == src_lang) & (df["target_langs"] == tgt_lang)]

            results[f"spbleu_{src_lang}-{tgt_lang}"] = corpus_bleu(
                pairs["preds"].tolist(),
                [pairs["labels"].tolist()],
                smooth_method="exp",
                force=True,
                tokenize="flores200",
            ).score
            
        results["avg_spbleu"] = mean(
            [results[f"spbleu_{src_lang}-{tgt_lang}"] for src_lang, tgt_lang, n in unique_pairs]
        ) if len(unique_pairs) > 0 else 0.0

        l = [results[f"spbleu_{src_lang}-{tgt_lang}"] for src_lang, tgt_lang, n in unique_pairs if tgt_lang == "eng_Latn"]
        if len(l) > 0:
            results["avg_spbleu_X-eng"] = mean(l)

        l = [results[f"spbleu_{src_lang}-{tgt_lang}"] for src_lang, tgt_lang, n in unique_pairs if src_lang == "eng_Latn"]
        if len(l) > 0:
            results["avg_spbleu_eng-X"] = mean(l)

        l = [results[f"spbleu_{src_lang}-{tgt_lang}"] for src_lang, tgt_lang, n in unique_pairs if src_lang == tgt_lang ]
        if len(l) > 0:
            results["avg_spbleu_AE"] = mean(l)

        return results
        
    def compute_xsim(eval_preds):
        preds, labels, inputs, loss = eval_preds
        
        source_langs_ids = inputs[:, 0]
        
        
        lang_codes = [tokenizer.convert_ids_to_tokens(int(src)) for src in source_langs_ids.tolist()]   
        
        xsim = XSim()
        
        df = pd.DataFrame({
            "lang": lang_codes,
            "id": labels[:, 0].tolist(),
            "emb": preds.tolist()
        })
        
        score = xsim.calc_xsim(
            df, 
            tgt_langs=["eng_Latn"],
            verbose=True
        )
        
        for (pair, xsim_score, nbex) in score:
            if pair == "average":
                results["avg_xsim"] = xsim_score
            else:
                results[f"xsim_{pair}"] = xsim_score
            
        score = xsim.calc_xsim(
            df, 
            tgt_langs=["eng_Latn"],
            xsimpp=True,    
            verbose=True
        )
        for (pair, xsim_score, nbex) in score:
            if pair == "average":
                results["avg_xsim++"] = xsim_score
            else:
                results[f"xsim++_{pair}"] = xsim_score

        return results
    
    # Initialize our Trainer
    trainer = SONARTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_spbleu if training_args.predict_with_generate else None,
        specific_parameters=data_args.specific_parameters,
        specific_learning_rate=data_args.specific_learning_rate,
        # xsim=xsim
    )
    
    

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    
    generation_kwargs = data_args.generation_kwargs if data_args.generation_kwargs is not None else {}
    
    generation_kwargs["max_length"] = max_length
    generation_kwargs["num_beams"] = num_beams
    

    
    
    if training_args.do_eval:
        
        logger.info("*** Xsim ***")

        data_files = {"eval": data_args.xsim_eval_file}
        xsim_ds = load_dataset("parquet", data_files=data_files, split="eval")

        xsim_preprocessor = NllbPreprocessor(
            tokenizer=tokenizer,
            data_args=data_args,
            padding=padding,
            mode="xsim",
        )

        xsim_ds = xsim_ds.map(
            xsim_preprocessor,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=xsim_ds.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            cache_file_name=os.path.join(cache_dir, "data", f"xsim_dataset.arrow"),
            desc="Running tokenizer on xsim dataset",
        ).sort("lengths", reverse=True)

        
        trainer.compute_metrics = compute_xsim

        xsim_results = trainer.predict(
            xsim_ds, metric_key_prefix="predict", eval_mode="embed"

        )
        
        logger.info("*** Evaluate ***")
        logger.info(f"Generation kwargs: {generation_kwargs}")
        metrics = trainer.evaluate(metric_key_prefix="eval", **generation_kwargs)


        metrics.update(xsim_results.metrics)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        
                ## XSIM evaluation
                
        logger.info("*** Xsim ***")

        data_files = {"predict": data_args.xsim_test_file}
        xsim_ds = load_dataset("parquet", data_files=data_files, split="predict")

        xsim_preprocessor = NllbPreprocessor(
            tokenizer=tokenizer,
            data_args=data_args,
            padding=padding,
            mode="xsim",
        )

        xsim_ds = xsim_ds.map(
            xsim_preprocessor,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=xsim_ds.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
            cache_file_name=os.path.join(cache_dir, "data", f"xsim_dataset.arrow"),
            desc="Running tokenizer on xsim dataset",
        ).sort("lengths", reverse=True)

        
        trainer.compute_metrics = compute_xsim

        xsim_results = trainer.predict(
            xsim_ds, metric_key_prefix="predict", eval_mode="embed"
            # batch_size=data_args.xsim_batch_size
        )

        metrics = xsim_results.metrics
        
        logger.info("*** Predict ***")
        
        
        trainer.compute_metrics = compute_spbleu  
        
        generation_kwargs["num_beams"] = data_args.predict_num_beams if data_args.predict_num_beams is not None else num_beams

        logger.info(f"Generation kwargs: {generation_kwargs}")
        
        ## Translation



        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", **generation_kwargs
        )


        metrics.update(predict_results.metrics)

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
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()