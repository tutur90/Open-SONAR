#!/usr/bin/env python3
from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import torch

import sys 
import os
sys.path.append(os.getcwd())

from open_sonar.text.models.modeling_sonar import SONARForText2Text


model_path = "open_sonar/text/models/pretrained/nllb_1.3B"


model_id = "facebook/nllb-200-1.3B"

model = SONARForText2Text.from_m2m100_pretrained(model_id)

model.save_pretrained(model_path)


tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(model_path)


model = SONARForText2Text.from_pretrained(model_path)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

out = model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_length=30)

print(out)
