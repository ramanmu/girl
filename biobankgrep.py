import faiss
import json
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import pandas as pd
import pickle
import re
from sentence_transformers import SentenceTransformer, CrossEncoder
import spacy
from types import SimpleNamespace

class BioBankGrep:
  def __init__(self, cfg_file="girl.cfg"):
  #{
    # ---- LOAD GLOBAL CONFIG AND BIOBANK DATA -----------
    with open(cfg_file, "r") as f: rc = json.load(f);
    self.cfg = json.loads(json.dumps(rc), object_hook=lambda d: SimpleNamespace(**d));
    self.bi_encoder = SentenceTransformer(self.cfg.bi_encoder);
    self.cross_encoder = CrossEncoder(self.cfg.cross_encoder);
    self.nlp = spacy.load(self.cfg.nlp_model.strip("'\""));
    self.stemmer = SnowballStemmer("english");
    self.rrf_k = self.cfg.rrf_k;
    
    # ---- LOAD MANIFEST AND ARTIFACTS --------------
    with open(self.cfg.manifest_file, "r") as f: manifest = json.load(f);
    with open(manifest["df_file"], "rb") as f: self.f_df = pickle.load(f);
    with open(manifest["bm25_file"], "rb") as f: self.bm25 = pickle.load(f)
    self.index = faiss.read_index(manifest["faiss_file"]);
  #}

  def clean_human_query(self, raw_query):
  #{ Linguistic tokenization and normalization
    doc = self.nlp(raw_query.lower());
    clean_stems = [];
    for w in doc:
    #{
      if not w.is_punct and not w.is_space:
        clean_stems.append(self.stemmer.stem(w.lemma_));
    #}
    return clean_stems;
  #}

  def execute_query(self, raw_query, set_filters={}, top_n=None):
  #{
    if top_n is None: top_n = self.cfg.default_top_k;

    # --- APPLY FILTER CONSTRAINTS ---------------
    f_df = self.f_df.copy();
    for col, val in set_filters.items():
    #{
      if col in f_df.columns:
        if col in self.cfg.filters
          f_df = f_df[f_df[col] in val];
        elif col in self.cfg.text_columns:
          f_df = f_df[f_df[col].astype(str).str.contains(val, flags=re.IGNORECASE, na=False)];
    #}
    f_df_ids = f_df.index.tolist();
    if not f_df_ids: return pd.DataFrame(columns=[c for c in self.f_df.columns if c != 'docs']);

    # --- PHASE 1: THE WIDE NET RETRIEVAL (Top 40 Candidates) ---
    # BM25 scoring
    clean_stems = self.clean_human_query(raw_query);
    bm25_scores = self.bm25.get_scores(clean_stems);

    # FAISS Dense Vector scoring
    query_vector = self.bi_encoder.encode([raw_query], convert_to_numpy=True);
    faiss.normalize_L2(query_vector)
    _, faiss_rankings = self.index.search(query_vector, len(self.f_df))
    faiss_rankings = faiss_rankings[0]

    # Standard RRF blend to find the top candidate subset
    rrf_scores = np.zeros(len(self.f_df))
    for rank, idx in enumerate(np.argsort(bm25_scores)[::-1]):
      if idx in f_df_ids:
        rrf_scores[idx] += 1.0 / (self.rrf_k + rank)
    for rank, idx in enumerate(faiss_rankings):
      if idx in f_df_ids:
        rrf_scores[idx] += 1.0 / (self.rrf_k + rank)

    # Sort the rankings.
    # If needed, the candidates can be
    # limited to say, just the top 40, as follows:
    # candidate_ids = np.argsort(rrf_scores)[::-1][:40]
    candidate_ids = np.argsort(rrf_scores)[::-1]
    candidate_ids = [idx for idx in candidate_ids if rrf_scores[idx] > 0];

    # --- CROSS-ENCODER NEURAL RE-RANKING ---
    # Build pairs of [ "User Query", "Full Row Text Context" ]
    eval_pairs = []
    for idx in candidate_ids:
      row_text = self.f_df.loc[idx, "docs"];
      eval_pairs.append([raw_query, row_text])

    # Compute exact semantic/proximity relevance scores simultaneously
    # Higher scores = higher structural and context alignment
    candidates = []
    if eval_pairs:
    #{
      cs = self.cross_encoder.predict(eval_pairs)
      for i, idx in enumerate(candidate_ids): candidates.append( {"idx": idx, "score": float(cs[i])} );
      candidates = sorted(candidates, key=lambda x: x["score"], reverse=True);
    #}

    # --- CONSTRUCT OUTPUT DATAFRAME ---
    final_indices = [item["idx"] for item in candidates[:top_n]]
    final_scores = [item["score"] for item in candidates[:top_n]]
    results_df = self.f_df.loc[final_indices].copy()
    results_df = results_df.drop(columns=['docs']);
    results_df["Ranking"] = final_scores
    return results_df
  #}
