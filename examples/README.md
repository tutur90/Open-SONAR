# Uses

## Load the model and tokenizer


```Python

SONARForSpeech2Text.from_pretrained("tutur90/SONAR-Text-to-Text")

NllbTokenizer.from_pretrained("tutur90/SONAR-Text-to-Text")

```

The code of SONARForSpeech2Text avalaible in [Open SONAR - Model](https://github.com/tutur90/Open-SONAR/blob/main/open_sonar/text/models/modeling_sonar.py) and NllbTokenizer [Open SONAR - Tokenizer](https://github.com/tutur90/Open-SONAR/blob/main/open_sonar/tokenizer.py)


## Translation

```Python

inputs = tokenizer(sentence, langs=src_lang, return_tensors="pt")

generated = sonar.generate(
        **inputs,
        target_lang_ids=[tokenizer.convert_tokens_to_ids(tgt_lang)],
        max_length=128,
        num_beams=1,
        do_sample=False,
    )
decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]

print(f"SONAR {src_lang} -> {tgt_lang}: {decoded}")

```

## Encode

```Python

encoder = SONARForText2Text.from_pretrained("tutur90/SONAR-Text-to-Text")

encoder.set_encoder_only() # Delete decoder to save memory, this options is not needed

inputs = tokenizer(sentence, langs=src_lang, return_tensors="pt")

embeddings = encoder.encode(**inputs)

```


## Decode 

```Python

decoder = SONARForText2Text.from_pretrained("tutur90/SONAR-Text-to-Text")

decoder.set_decoder_only() # Same

decoded = decoder.decode(
        encoder_outputs,
        target_lang_ids=[tokenizer.convert_tokens_to_ids(tgt_lang)],
        max_length=128,
        num_beams=1,
        do_sample=False,
    )
decoded = tokenizer.batch_decode(decoded, skip_special_tokens=True)[0]


```




