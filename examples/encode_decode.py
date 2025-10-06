
import os
import sys
import torch
from open_sonar.tokenizer import NllbTokenizer

import os
sys.path.append(os.getcwd())

from open_sonar.text.models.modeling_sonar import SONARForText2Text


encoder = SONARForText2Text.from_pretrained("tutur90/SONAR-Text-to-Text")

encoder.set_encoder_only()

decoder = SONARForText2Text.from_pretrained("tutur90/SONAR-Text-to-Text")

decoder.set_decoder_only()

tokenizer = NllbTokenizer.from_pretrained("tutur90/SONAR-Text-to-Text")

def process_sentence(sentence, src_lang, tgt_lang):
    inputs = tokenizer(sentence, langs=src_lang, return_tensors="pt")

    encoder_outputs = encoder.encode(**inputs)

    print("Embedding for", sentence, ":", encoder_outputs)

    decoded = decoder.decode(
            encoder_outputs,
            target_lang_ids=[tokenizer.convert_tokens_to_ids(tgt_lang)],
            max_length=128,
            num_beams=1,
            do_sample=False,
        )
    decoded = tokenizer.batch_decode(decoded, skip_special_tokens=True)[0]
    print(f"SONAR {src_lang} -> {tgt_lang}: {decoded}")
        

sentences = [
    {"text": "My name is SONAR.", "src_lang": "eng_Latn", "tgt_lang": "fra_Latn"},
    {"text": "당신은 꿈의 깊은 곳에서 내게 말을 건다 영혼이 산 자에게 말하듯이...", "src_lang": "kor_Hang", "tgt_lang": "fra_Latn"},
]


for sentence in sentences:
    process_sentence([sentence['text']], sentence['src_lang'], sentence['tgt_lang'])