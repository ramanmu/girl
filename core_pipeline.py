import re
import hashlib
import configparser

def load_config(config_file="girl.cfg"):
#{
  cfg = configparser.ConfigParser()
  cfg.read(config_file)
  return cfg["GLOBAL"]
#}

def clean_clinical_text(text, nlp_model):
#{
  # 1. Regex Sledgehammer (Surgical contraction removal)
  # This prevents punctuation from becoming orphaned tokens.
  safe_string = re.sub(r"\b's\b|\b't\b", "", text.lower())
  safe_string = re.sub(r'[^\w\s]', ' ', safe_string)
  
  doc = nlp_model(safe_string)
  clean_lemmas = []
  
  for token in doc:
  #{
    # Keep nouns, proper nouns, and adjectives as clinical indicators
    if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop:
    #{
      if nlp_model.vocab.strings[token.text]:
        clean_lemmas.append(token.lemma_.lower())
    #}
  #}
  return clean_lemmas
#}

def generate_state_hash(df):
#{
  # Ensures indexer and grepper are in lockstep
  index_str = "".join(map(str, df.index.tolist()))
  return hashlib.md5(index_str.encode('utf-8')).hexdigest()
#}
