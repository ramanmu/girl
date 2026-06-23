import os
import re
import json
import faiss
import pickle
import numpy as np
import pandas as pd
import spacy
import time
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types
import core_pipeline

class BioBankGrep:
#{
  def __init__(self):
  #{
    self.cfg = core_pipeline.load_config()
    self.df = pd.read_pickle(f"biobank_df.pkl")

    # Hash check
    with open("biobank_hash.json", "r") as f: meta = json.load(f)
    if core_pipeline.generate_state_hash(self.df) != meta["state_hash"]: raise Exception("Data registries desynchronized.")
      
    self.faiss = faiss.read_index(f"biobank_faiss.faiss")
    with open("biobank_bm25.pkl", "rb") as f: self.bm25 = pickle.load(f)

    try:
      with open("biobank_filters.json", "r") as f: self.filters = json.load(f)
    except FileNotFoundError:
      print("WARNING: filters_file not found. UI filters will be disabled.")
      self.filters = {}
    
    self.nlp = spacy.load(self.cfg.get("clinical_nlp_model"))
    self.bi_encoder = SentenceTransformer(self.cfg.get("bi_encoder_name"))
    self.cross_encoder = CrossEncoder(self.cfg.get("cross_encoder_name"))
    self.limit = len(self.df)
  #}

  def _apply_filter(self, idx: int, vals: list, fid: set) -> set:
  #{
    # Extract the schema definition for this specific filter
    # (Assuming manifest filters array matches the idx exactly)
    filter_schema = self.filters[idx]
    column = filter_schema["filter_column"]
    predicate = filter_schema["predicate"]
    
    # Optimization: Only run string matching on rows that have survived previous filters
    sub_df = self.df.loc[list(fid)]
    
    if predicate == "exact_match": match_idx = sub_df[sub_df[column].isin(vals)].index
    elif predicate == "contains_any":
      escaped_vals = [re.escape(str(v)) for v in vals]
      pattern = '|'.join(escaped_vals)
      match_idx = sub_df[sub_df[column].astype(str).str.contains(pattern, case=False, na=False)].index
    else: return fid # Fallback: if predicate is unknown, do not drop rows
      
    # Return the newly narrowed set of indices
    return fid.intersection(set(match_idx))
  #}

  def decompose_query(self, raw_query: str) -> dict:
  #{
    if not raw_query: return {"must_have_groups": [], "semantic_context": ""}

    prompt = f"""
    You are an expert at biobanks. Analyze the following biobank search query. 
    Identify the absolute mandatory constraints (e.g., specific environments like 'marine', specific species, anatomical origins like 'placenta', geographical locations).
    
    CRITICAL INSTRUCTIONS:
    1. Group distinct constraints into separate lists.
    2. Perform deep ontological expansion for each constraint. Include direct synonyms and common sub-categories.
    3. MORPHOLOGICAL EXPANSION: You MUST include grammatical variations (plurals, adjectives). If the constraint is a noun like 'placenta', you must explicitly include its adjective form 'placental', 'placentas', etc.
    4. ENVIRONMENTAL TAXONOMY: If the constraint is a broad biological group (e.g., 'animal', 'plant', 'microbe'), you MUST include the major environmental domains where they exist (e.g., 'marine', 'aquatic', 'terrestrial', 'ocean', 'sea').
    5. SEMANTIC PRESERVATION: Do NOT strip the constraints out of the 'semantic_context'. The semantic_context must be a rich, complete phrase containing both the constraints and the general concepts so the vector database has maximum context.
    
    Respond ONLY with a valid JSON object matching this schema:
    {{
        "must_have_groups": [
            ["placenta", "placental", "placentas", "chorion", "decidua", "trophoblast"], // Constraint 1 + expansions + adjectives
            ["animal", "marine", "avian", "aquatic", "fish"] // Example Constraint 2 + environmental domains
        ],
        "semantic_context": "placenta tissues" // DO NOT strip words. Leave the full descriptive phrase intact.
    }}
    
    Query: "{raw_query}"
    """

    try:
      client = genai.Client()
      response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.0)
      )
      return json.loads(response.text)
    except Exception as e:
      print(f"DEBUG: LLM Decomposition failed: {e}")
      return {"must_have_groups": [], "semantic_context": raw_query}
  #}

  def execute_query(self, dsl) -> pd.DataFrame:
  #{
    raw_query = dsl.get("nlp", "").strip().lower()
    if not raw_query: return pd.DataFrame(columns=self.df.columns)

    # PRE-FILTERING (The UI Predicate Dispatcher)
    surviving_indices = set(self.df.index)
    dsl_filters = dsl.get("filters", {})
    if dsl_filters:
      for idx, vals in dsl_filters.items():
        surviving_indices = self._apply_filter(int(idx), vals, surviving_indices)
        if not surviving_indices: return pd.DataFrame(columns=self.df.columns)

    # 2. LLM Query Decomposition
    start_time = time.time()
    llm_filters = self.decompose_query(raw_query)
    print(f"DBG Decomposed query in {time.time() - start_time:.2f}s: {llm_filters}")

    semantic_q = llm_filters.get("semantic_context", raw_query)
    if not semantic_q.strip(): semantic_q = raw_query

    # 3. FAISS Semantic Search
    semantic_qvec = self.bi_encoder.encode([semantic_q]).astype("float32")
    faiss.normalize_L2(semantic_qvec)
    faiss_scores, faiss_indices = self.faiss.search(semantic_qvec, len(self.df))
    print(f"DBG faiss_scores: {faiss_scores}")

    similarity_threshold = 0.3
    quality_faiss_mask = faiss_scores[0] >= similarity_threshold
    quality_faiss_indices = set(self.df.index[faiss_indices[0][i]] for i in range(len(faiss_indices[0])) if quality_faiss_mask[i])
    
    candidates_after_faiss = surviving_indices.intersection(quality_faiss_indices)
    if not candidates_after_faiss: return pd.DataFrame(columns=self.df.columns)

    # 4. BM25 Lexical Gate
    must_have_groups = llm_filters.get("must_have_groups", [])
    bm25_indices = candidates_after_faiss
    
    if must_have_groups:
      for group in must_have_groups:
        group_tokens = [t.lower() for t in group]
        print(f"DBG group: {group}")
        group_scores = self.bm25.get_scores(group_tokens)
        print(f"DBG group bm25: {group_scores}")
        group_df_indices = set(self.df.index[np.where(group_scores > 0)[0]])
        bm25_indices = bm25_indices.intersection(group_df_indices)
        if not bm25_indices: break
            
    if not bm25_indices: return pd.DataFrame(columns=self.df.columns)

    # 5. Cross-Encoder Ranking
    candidates = sorted(bm25_indices)
    # Create a super-query so the CE recognizes the sub-categories
    ce_inputs = [[raw_query, self.df.loc[idx, 'sem_context']] for idx in candidates]
    
    start_time = time.time()
    ce_scores = self.cross_encoder.predict(ce_inputs)
    print(f"DBG CE-ranked {len(candidates)} candidates in {time.time() - start_time:.2f}s")
    
    ce_probabilities = 1.0 / (1.0 + np.exp(-ce_scores))
    print(f"DBG CE probabilities: {ce_probabilities}")

    results_series = pd.Series(ce_probabilities, index=candidates)
    user_top_k = dsl.get("top_k", self.limit)
    
    top_results = results_series.sort_values(ascending=False).head(user_top_k)
    ce_threshold = 0.0001 
    top_results = top_results[top_results >= ce_threshold]
    print(f"Returning {len(top_results)} rows")
    
    if top_results.empty: return pd.DataFrame(columns=self.df.columns)
    return self.df.loc[top_results.index].assign(ce_score=top_results)
  #}
#}
