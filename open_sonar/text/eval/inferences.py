
import os
import sys
import torch
from transformers import NllbTokenizer

import os
cwd = os.getcwd()

sys.path.append(cwd)

from open_sonar.text.models.modeling_sonar import SONARForText2Text



# Chargement du modèle SONAR
sonar = SONARForText2Text.from_pretrained("tutur90/SONAR-Text-to-Text")



# Initialisation du tokenizer
tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-distilled-1.3B", legacy_behavior=False)

# Fonction pour traiter une phrase
def process_sentence(sentence, src_lang, tgt_lang, mode='generate'):
    tokenizer.src_lang = src_lang
    inputs = tokenizer(sentence, return_tensors="pt")

    decoded = sonar.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
            max_length=128,
            num_beams=1,
            do_sample=False,
        )
    decoded = tokenizer.batch_decode(decoded, skip_special_tokens=True)[0]
    print(f"SONAR {src_lang} -> {tgt_lang}: {decoded}")
        


# Exemple d'utilisation
sentences = [
    {"text": "My name is SONAR.", "src_lang": "eng_Latn", "tgt_lang": "fra_Latn"},
    {"text": "당신은 꿈의 깊은 곳에서 내게 말을 건다 영혼이 산 자에게 말하듯이...", "src_lang": "kor_Hang", "tgt_lang": "fra_Latn"},
]


for sentence in sentences:
    process_sentence([sentence['text']], sentence['src_lang'], sentence['tgt_lang'], mode=generation_mode)
