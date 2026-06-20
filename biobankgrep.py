import pandas as pd
import numpy as np
import faiss
import pickle
import json
import spacy
from sentence_transformers import SentenceTransformer, CrossEncoder
import core_pipeline

class BioBankGrep:
#{
  def __init__(self):
  #{
    self.cfg = core_pipeline.load_config()
    base = self.cfg.get("index_file_basename")
    self.nlp = spacy.load(self.cfg.get("clinical_nlp_model"))
    
    # 1. State Verification
    self.df = pd.read_pickle(f"{base}_df.pkl")
    with open(f"{base}_meta.json", "r") as f: meta = json.load(f)
    if core_pipeline.generate_state_hash(self.df) != meta["state_hash"]:
      raise Exception("Index synchronization error.")
    
    # 2. Infrastructure Load
    self.index = faiss.read_index(f"{base}.faiss")
    with open(f"{base}_bm25.pkl", "rb") as f: self.bm25 = pickle.load(f)
    with open(self.cfg.get("manifest_file"), "r") as f: self.manifest = json.load(f)
    
    self.bi_encoder = SentenceTransformer(self.cfg.get("bi_encoder_name"))
    self.cross_encoder = CrossEncoder(self.cfg.get("cross_encoder_name"))
    self.recall_limit = self.cfg.getint("stage_1_recall_limit", fallback=30)
  #}

  def execute_query(self, dsl):
  #{
    raw_query = dsl.get("nlp", "").strip()
    filters = dsl.get("filters", {})
    
    # Apply Filters (from manifest.json)
    f_df = self.df.copy()
    for f in self.manifest["filters"]:
    #{
      col, f_type = f["column"], f["type"]
      val = filters.get(col)
      if not val: continue
      
      if f_type == "multi": f_df = f_df[f_df[col].isin(val)]
      elif f_type == "substring": f_df = f_df[f_df[col].astype(str).str.contains(val, case=False)]
    #}
    
    processed_query = " ".join(core_pipeline.clean_clinical_text(raw_query, self.nlp))
    subset_ids = f_df.index.tolist()
    
    # Stage 1: Hybrid Recall (Recall limit constraint)
    recall_k = min(self.recall_limit, len(subset_ids))
    
    # Semantic
    q_vec = self.bi_encoder.encode([processed_query]).astype("float32")
    faiss.normalize_L2(q_vec)
    _, faiss_indices = self.index.search(q_vec, recall_k)
    
    # Lexical
    tokenized_q = processed_query.split()
    k_scores = np.array([self.bm25.get_scores(tokenized_q)[i] for i in subset_ids])
    bm25_indices = np.argsort(-k_scores)[:recall_k]
    
    candidates = list(set([subset_ids[i] for i in faiss_indices[0]] + [subset_ids[i] for i in bm25_indices]))
    
    # Stage 2: Cross-Encoder Precision
    ce_inputs = [(raw_query, " ".join(map(str, self.df.loc[idx].values))) for idx in candidates]
    scores = self.cross_encoder.predict(ce_inputs)
    
    # Results
    results = pd.Series(scores, index=candidates).sort_values(ascending=False)
    return self.df.loc[results.index].assign(ce_score=results)
  #}
#}
