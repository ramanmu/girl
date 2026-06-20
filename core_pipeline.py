import re
import hashlib
import configparser

def load_config(config_file="girl.cfg"):
#{
  cfg = configparser.ConfigParser()
  cfg.read(config_file)
  return cfg["GLOBAL"]
#}

def get_indexable_text(text):
#{
  # Strip formatting artifacts without destroying language content
  text = re.sub(r'[\u2022\u2023\u25E6\u2043\u2219\*\-]', '', str(text))
  return re.sub(r'\s+', ' ', text).strip()
#}

def clean_clinical_text(text, nlp_model):
#{
  # Normalize grammar for BM25/Bi-Encoder recall
  safe_text = get_indexable_text(text)
  safe_string = re.sub(r"\b's\b|\b't\b", "", safe_text.lower())
  safe_string = re.sub(r'[^\w\s]', ' ', safe_string)
  
  doc = nlp_model(safe_string)
  return [token.lemma_.lower() for token in doc 
    if token.pos_ in {"NOUN", "PROPN", "ADJ"} and not token.is_stop]
#}

def generate_state_hash(df):
#{
  # Hash the index to ensure display and semantic registries are congruent
  return hashlib.md5("".join(map(str, df.index.tolist())).encode('utf-8')).hexdigest()
#}
