import sys 
import os
sys.path.append(os.getcwd())

from transformers import SpeechEncoderDecoderModel, AutoFeatureExtractor, AutoTokenizer
import torch
from open_sonar.speech.models.modeling import SONARForSpeech2Text



model_path = "open_sonar/speech/models/pretrained/sonar_speech"

encoder_id = "facebook/w2v-bert-2.0"
model_id = "open_sonar/text/models/pretrained/pretrained_sonar"


model = SONARForSpeech2Text.from_sonar_w2v_pretrained(model_id, encoder_id)

model.config.processor_class = "Wav2Vec2Processor"

print("Model config:", model.config)

model.save_pretrained(model_path)

feature_extractor = AutoFeatureExtractor.from_pretrained(encoder_id)
feature_extractor.save_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B")
tokenizer.save_pretrained(model_path)


model = SONARForSpeech2Text.from_pretrained(model_path)


num_params = sum(p.numel() for p in model.parameters())


inputs = feature_extractor(torch.ones((1, 10000)), return_tensors="pt")

# inputs["inputs_ids"] = inputs.pop("input_values")


out = model.generate(**inputs, max_length=128, num_beams=4)

print(out)
