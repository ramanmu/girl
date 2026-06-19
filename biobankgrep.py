import pandas as pd
import numpy as np
import faiss
import pickle
import json
import configparser
import re
from sentence_transformers import SentenceTransformer
import spacy
#from nltk.stem.snowball import SnowballStemmer

class BioBankGrep:
  def __init__(self):
  #{
    cfg = configparser.ConfigParser()
    cfg.read("girl.cfg")
    self.cfg = cfg["GLOBAL"]
        
    base = self.cfg.get("index_file_basename")
    self.index = faiss.read_index(f"{base}.faiss")
    with open(f"{base}_bm25.pkl", "rb") as f: self.bm25 = pickle.load(f)
    self.df = pd.read_pickle(f"{base}_df.pkl")
    with open(self.cfg.get("manifest_file"), "r") as f: self.schema = json.load(f)
        
    self.model = SentenceTransformer(self.cfg.get("model_name"))
    self.rrf_k = 60
        
    # Load the NLP brain once when the engine starts
    self.nlp = spacy.load("en_core_sci_sm");
  #}

  def clean_human_query(self, raw_query):
  #{
    custom_stop_lemmas = { "biobank", "repositor", "sampl", "specimen", "data", "databas" }
    clean_lemmas = []
    doc = self.nlp(raw_query.lower())
      
    for token in doc:
    #{
      if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
        if not token.is_stop:
          # Lexical Gatekeeper: Check raw text against dictionary before lemmatizing
          if self.nlp.vocab.strings[token.text]:
            lemma_word = token.lemma_.lower()
            if lemma_word not in custom_stop_lemmas:
              clean_lemmas.append(lemma_word)
    #}
      
    if not clean_lemmas:
      clean_lemmas = [
        token.lemma_.lower() for token in self.nlp(raw_query.lower()) 
        if token.lemma_.lower() not in custom_stop_lemmas and self.nlp.vocab.strings[token.text]
      ]
        
    return clean_lemmas
  #}

  def execute_query(self, dsl):
  #{
    raw_query = dsl.get("nlp", "").strip()
    filters = dsl.get("filters", {})
    top_k = dsl.get("top_k", self.schema["default_top_k"])
        
    # 1. ABSOLUTE SYMMETRY: Strip punctuation before any processing
    raw_query = re.sub(r'[^\w\s]', ' ', raw_query)
        
    # 2. THE ABSOLUTE FIREWALL
    if len(raw_query.strip()) <= 1:
      empty_df = pd.DataFrame(columns=self.df.columns)
      return empty_df.assign(rrf_score=[])
            
    # 3. Clean query
    clean_query_words = self.clean_human_query(raw_query)
    processed_query = " ".join(clean_query_words)
        
    # 4. Apply Pandas Filters
    f_df = self.df.copy()
    type_map = {f["column"]: f["type"] for f in self.schema["filters"]}
        
    for col, val in filters.items():
      if not val: continue
      f_type = type_map.get(col)
            
      if f_type == "multi": 
        f_df = f_df[f_df[col].isin(val)]
      elif f_type == "substring":
        search_str = " ".join(val).lower()
        f_df = f_df[f_df[col].astype(str).str.lower().str.contains(search_str, na=False, regex=False)]
                
    # Safeguard against empty results
    if f_df.empty or top_k <= 0 or not processed_query: 
      empty_df = pd.DataFrame(columns=f_df.columns)
      return empty_df.assign(rrf_score=[])

    # 5. Vector Score
    subset_ids = f_df.index.tolist()
    q_vec = self.model.encode([processed_query]).astype("float32")
    faiss.normalize_L2(q_vec)
    v_scores, _ = self.index.search(q_vec, self.index.ntotal)
    v_scores = v_scores[0][subset_ids]

    # 6. BM25 Score
    tokenized_q = processed_query.split()
    k_scores = np.array([self.bm25.get_scores(tokenized_q)[i] for i in subset_ids])

    # 7. RRF
    v_ranks, k_ranks = np.argsort(-v_scores), np.argsort(-k_scores)
    rrf_map = {idx: 0.0 for idx in subset_ids}
    for r, p in enumerate(v_ranks): rrf_map[subset_ids[p]] += 1.0/(self.rrf_k + r)
    for r, p in enumerate(k_ranks): rrf_map[subset_ids[p]] += 1.0/(self.rrf_k + r)

    # --- THE PURE NLP BOOSTER & GIBBERISH FIREWALL ---
    # Using enumerate(subset_ids) fixes the IndexError on v_scores
    for i, idx in enumerate(subset_ids):
    #{
      row_text = " ".join(map(str, f_df.loc[idx].values)).lower()
      
      # Mirror the punctuation stripping exactly
      clean_row_text = re.sub(r'[^\w\s]', ' ', row_text)
      
      # Use SpaCy to lemmatize the database row
      row_doc = self.nlp(clean_row_text)
      clean_row_lemmas = set([token.lemma_.lower() for token in row_doc if not token.is_space])
      
      # Match on neural lemmas, not chopped strings
      match_count = sum(1 for lemma in clean_query_words if lemma in clean_row_lemmas)

      if match_count > 0: rrf_map[idx] *= (1.0 + (match_count * 0.5))

      # The 0.25 heuristic to kill unconfident semantic noise
      if match_count == 0 and v_scores[i] < 0.25: 
        rrf_map[idx] = 0.0
    #}

    final_s = pd.Series(rrf_map)
    final_s = final_s[final_s > 0.0] 
        
    winners = final_s.sort_values(ascending=False).index[:top_k]
    return f_df.loc[winners].assign(rrf_score=final_s[winners])
  #}
