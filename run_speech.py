#!/usr/bin/env python
# Copyright 2021 The HuggingFace Team. All rights reserved.
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
Fine-tuning the library models for sequence to sequence speech recognition.
"""
# You can also adapt this script on your own sequence to sequence speech
# recognition task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
import pandas as pd
from typing import Any, Optional, Union

import datasets
import evaluate
from sacrebleu import corpus_bleu
import torch
from datasets import DatasetDict, load_dataset, load_from_disk

import transformers
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version


from open_sonar.speech.models.modeling import SONARForSpeech2Text, SONARSpeechConfig

from open_sonar.speech.args import ModelArguments, DataTrainingArguments, DataCollatorSpeechSeq2SeqWithPadding
from open_sonar.trainer import SONARTrainer

import numpy as np
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.53.0")

require_version("datasets>=1.18.0", "To fix: pip install -r examples/pytorch/speech-recognition/requirements.txt")

logger = logging.getLogger(__name__)


def main():
    # 1. Parse input arguments
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
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
        

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_speech_recognition_seq2seq", model_args, data_args)

    # 2. Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # 3. Detecting last checkpoint and eventually continue from last checkpoint
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

    # 4. Load dataset
    raw_datasets = DatasetDict()

    if training_args.do_train:
        raw_datasets["train"] = load_from_disk(
            data_args.dataset_name,
            # split=data_args.train_split_name,
            # cache_dir=model_args.cache_dir,
            # token=model_args.token,
            # trust_remote_code=model_args.trust_remote_code,
        )

    
    if training_args.do_eval:
        raw_datasets["eval"] = load_from_disk(
            data_args.eval_dataset_name
        )["validation"]
        
    if training_args.do_predict:
        raw_datasets["predict"] = load_dataset(
            data_args.eval_dataset_name
        )["test"]

    # if data_args.audio_column_name not in next(iter(raw_datasets.values())).column_names:
    #     raise ValueError(
    #         f"--audio_column_name '{data_args.audio_column_name}' not found in dataset '{data_args.dataset_name}'. "
    #         "Make sure to set `--audio_column_name` to the correct audio column - one of "
    #         f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
    #     )

    # if data_args.text_column_name not in next(iter(raw_datasets.values())).column_names:
    #     raise ValueError(
    #         f"--text_column_name {data_args.text_column_name} not found in dataset '{data_args.dataset_name}'. "
    #         "Make sure to set `--text_column_name` to the correct text column - one of "
    #         f"{', '.join(next(iter(raw_datasets.values())).column_names)}."
    #     )

    # 5. Load pretrained model, tokenizer, and feature extractor
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    config = SONARSpeechConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        model_args.feature_extractor_name if model_args.feature_extractor_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )
    model = SONARForSpeech2Text.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        token=model_args.token,
        trust_remote_code=model_args.trust_remote_code,
    )

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")



    model.freeze_decoder()


    if hasattr(model.generation_config, "is_multilingual") and model.generation_config.is_multilingual:
        # We only need to set the language and task ids in a multilingual setting
        tokenizer.set_prefix_tokens(language=data_args.language, task=data_args.task)
        model.generation_config.language = data_args.language
        model.generation_config.task = data_args.task
    elif data_args.language is not None:
        raise ValueError(
            "Setting language token for an English-only checkpoint is not permitted. The language argument should "
            "only be set for multilingual checkpoints."
        )

    # TODO (Sanchit): deprecate these arguments in v4.41
    if model_args.forced_decoder_ids is not None:
        logger.warning(
            "The use of `forced_decoder_ids` is deprecated and will be removed in v4.41."
            "Please use the `language` and `task` arguments instead"
        )
        model.generation_config.forced_decoder_ids = model_args.forced_decoder_ids
    else:
        model.generation_config.forced_decoder_ids = None
        model.config.forced_decoder_ids = None

    if model_args.suppress_tokens is not None:
        logger.warning(
            "The use of `suppress_tokens` is deprecated and will be removed in v4.41."
            "Should you need `suppress_tokens`, please manually set them in the fine-tuning script."
        )
        model.generation_config.suppress_tokens = model_args.suppress_tokens

    # 6. Resample speech dataset if necessary
    # dataset_sampling_rate = next(iter(raw_datasets.values())).features[data_args.audio_column_name].sampling_rate
    # if dataset_sampling_rate != feature_extractor.sampling_rate:
    #     raw_datasets = raw_datasets.cast_column(
    #         data_args.audio_column_name, datasets.features.Audio(sampling_rate=feature_extractor.sampling_rate)
    #     )

    # 7. Preprocessing the datasets.
    # We need to read the audio files as arrays and tokenize the targets.
    max_input_length = data_args.max_duration_in_seconds * feature_extractor.sampling_rate
    min_input_length = data_args.min_duration_in_seconds * feature_extractor.sampling_rate
    audio_column_name = data_args.audio_column_name
    num_workers = data_args.preprocessing_num_workers
    text_column_name = data_args.text_column_name
    model_input_name = feature_extractor.model_input_names[0]
    do_lower_case = data_args.do_lower_case
    # if SpecAugment is used for whisper models, return attention_mask to guide the mask along time axis
    forward_attention_mask = (
        getattr(config, "model_type", None) == "whisper"
        and getattr(config, "apply_spec_augment", False)
        and getattr(config, "mask_time_prob", 0) > 0
    )

    if data_args.max_train_samples is not None:
        raw_datasets["train"] = raw_datasets["train"].select(range(data_args.max_train_samples))

    if data_args.max_eval_samples is not None:
        raw_datasets["eval"] = raw_datasets["eval"].select(range(data_args.max_eval_samples))

    def prepare_dataset(batch):
        # process audio
        sample = batch[audio_column_name]
        inputs = feature_extractor(
            sample["array"], sampling_rate=sample["sampling_rate"], return_attention_mask=forward_attention_mask
        )
        # process audio length
        batch[model_input_name] = inputs.get(model_input_name)[0]
        batch["input_length"] = len(sample["array"])
        if forward_attention_mask:
            batch["attention_mask"] = inputs.get("attention_mask")[0]

        batch["target_embeddings"] = batch["embeddings"]

        # # process targets
        # input_str = batch[text_column_name].lower() if do_lower_case else batch[text_column_name]
        # batch["labels"] = tokenizer(input_str).input_ids
        return batch
    

    # with training_args.main_process_first(desc="dataset map pre-processing"):
    #     if training_args.do_train:
    #         print("Processing train dataset...", training_args.do_train)
    #         train_datasets = raw_datasets["train"]
    #     if training_args.do_eval:
    #         eval_datasets = raw_datasets["eval"]
            
    #     if training_args.do_predict:
    #         predict_dataset = raw_datasets["predict"]
            
        
    raw_datasets = raw_datasets.rename_column("length", "lengths")

    # print("train_datasets: ", train_datasets)
    # filter data that is shorter than min_input_length or longer than
    # max_input_length
    def is_audio_in_length_range(length):
        return length > min_input_length and length < max_input_length

    raw_datasets = raw_datasets.filter(
        is_audio_in_length_range,
        num_proc=num_workers,
        input_columns=["lengths"],
    )

    # print("raw_datasets: ", raw_datasets["eval"][0])

    # for large datasets it is advised to run the preprocessing on a
    # single machine first with `args.preprocessing_only` since there will mostly likely
    # be a timeout when running the script in distributed mode.
    # In a second step `args.preprocessing_only` can then be set to `False` to load the
    # cached dataset
    if data_args.preprocessing_only:
        cache = {k: v.cache_files for k, v in raw_datasets.items()}
        logger.info(f"Data preprocessing finished. Files cached at {cache}.")
        return

    # 8. Load Metric
    metric = evaluate.load("wer", cache_dir=model_args.cache_dir)

    def postprocess_text(preds):
        preds = [pred.strip() for pred in preds]

        return preds

    def compute_metrics(eval_preds):
        preds, labels = eval_preds


        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = postprocess_text(tokenizer.batch_decode(preds, skip_special_tokens=True))
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = postprocess_text(tokenizer.batch_decode(labels, skip_special_tokens=True))

        results = {}


        results[f"bleu"] = corpus_bleu(
                decoded_preds,
                [decoded_labels],
                smooth_method="exp",
                force=True,
                tokenize="flores200",
            ).score
            
        results[f"wer"] = 100 * metric.compute(predictions=decoded_preds, references=decoded_labels)

        return results

    # 9. Create a single speech processor
    # make sure all processes wait until data is saved
    with training_args.main_process_first():
        # only the main process saves them
        if is_main_process(training_args.local_rank):
            # save feature extractor, tokenizer and config
            feature_extractor.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            config.save_pretrained(training_args.output_dir)

    # breakpoint()
    processor = AutoProcessor.from_pretrained(training_args.output_dir)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

    # 10. Define data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        forward_attention_mask=forward_attention_mask,
    )
    

    # 11. Initialize Trainer
    trainer = SONARTrainer(
        model=model,
        args=training_args,
        train_dataset=raw_datasets["train"] if training_args.do_train else None,
        eval_dataset=raw_datasets["eval"] if training_args.do_eval else None,
        # processing_class=feature_extractor,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # 12. Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        # breakpoint()
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the feature extractor too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples
            if data_args.max_train_samples is not None
            else len(raw_datasets["train"])
        )
        metrics["train_samples"] = min(max_train_samples, len(raw_datasets["train"]))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # 13. Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            metric_key_prefix="eval",
            max_length=training_args.generation_max_length,
            num_beams=training_args.generation_num_beams,
        )
        max_eval_samples = (
            data_args.max_eval_samples if data_args.max_eval_samples is not None else len(raw_datasets["eval"])
        )
        metrics["eval_samples"] = min(max_eval_samples, len(raw_datasets["eval"]))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # 14. Write Training Stats
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "automatic-speech-recognition"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


if __name__ == "__main__":
    main()