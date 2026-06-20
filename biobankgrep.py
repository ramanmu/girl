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
    
    self.df = pd.read_pickle(f"{base}_display_df.pkl")
    self.df_sem = pd.read_pickle(f"{base}_semantic_df.pkl")
    self.nlp = spacy.load(self.cfg.get("clinical_nlp_model"))
    
    # Hash check
    with open(f"{base}_meta.json", "r") as f: meta = json.load(f)
    if core_pipeline.generate_state_hash(self.df) != meta["state_hash"]:
      raise Exception("Data registries desynchronized.")
      
    self.index = faiss.read_index(f"{base}.faiss")
    with open(f"{base}_bm25.pkl", "rb") as f: self.bm25 = pickle.load(f)
    with open(self.cfg.get("manifest_file"), "r") as f: self.manifest = json.load(f)
    
    self.bi_encoder = SentenceTransformer(self.cfg.get("bi_encoder_name"))
    self.cross_encoder = CrossEncoder(self.cfg.get("cross_encoder_name"))
    self.limit = self.cfg.getint("stage_1_recall_limit", fallback=30)
  #}

  def execute_query(self, dsl):
  #{
    raw_query = dsl.get("nlp", "").strip()
    
    # 1-character guard clause
    if len(re.sub(r'[^w]', '', raw_query) <= 1: return pd.DataFrame(columns=self.df.columns);
    
    # Filter on Display DF
    f_df = self.df.copy()
    for f in self.manifest["filters"]:
    #{
      val = dsl.get("filters", {}).get(f["column"])
      if val:
      #{
        if f["type"] == "multi": f_df = f_df[f_df[f["column"]].isin(val)]
        elif f["type"] == "substring": f_df = f_df[f_df[f["column"]].astype(str).str.contains(val, case=False)]
      #}
    #}
    
    # Semantic Search Space
    processed_q = " ".join(core_pipeline.clean_clinical_text(raw_query, self.nlp))
    subset_ids = f_df.index.tolist()
    
    # Hybrid Recall
    q_vec = self.bi_encoder.encode([processed_q]).astype("float32")
    faiss.normalize_L2(q_vec)
    _, faiss_indices = self.index.search(q_vec, min(self.limit, len(subset_ids)))
    
    bm25_scores = self.bm25.get_scores(processed_q.split())
    bm25_indices = np.argsort(-bm25_scores)[:self.limit]
    
    candidates = list(set([subset_ids[i] for i in faiss_indices[0]] + [subset_ids[i] for i in bm25_indices]))
    
    # Re-ranker
    inputs = [(raw_query, " ".join(self.df_sem.loc[idx].values.astype(str))) for idx in candidates]
    scores = self.cross_encoder.predict(inputs)
    
    results = pd.Series(scores, index=candidates).sort_values(ascending=False)
    return self.df.loc[results.index].assign(ce_score=results)
  #}
#}
