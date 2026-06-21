import re
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
    raw_query = dsl.get("nlp", "").strip().lower()
    
    # 1. Base Cleanup via your existing spaCy pipeline
    raw_tokens = core_pipeline.clean_clinical_text(raw_query, self.nlp)
    
    # 2. Comprehensive Domain Blacklist
    # Strips out words that exist uniformly across rows to prevent Cross-Encoder overfitting
    domain_blacklist = {
      "biobank", "biobanks", "repository", "repositories", 
      "sample", "samples", "data", "specimen", "specimens", 
      "cohort", "cohorts", "collection", "collections",
      "study", "studies", "protocol", "protocols", "patient", "patients",
      "donor", "donors", "human", "investigator", "investigators"
    }
    filtered_tokens = [tok for tok in raw_tokens if tok not in domain_blacklist]
    
    # Short-circuit if the query is now empty
    if not filtered_tokens:
      return pd.DataFrame(columns=self.df.columns)
      
    # 3. Target Clinical Synonym Expansion (The Quick Ontology)
    # Built from your complete vocabulary sweep to bridge the layperson-clinical gap
    quick_ontology = {
      # Demographics & Age Groups
      "child": "child pediatric paediatric children neonate newborn juvenile",
      "children": "children child pediatric paediatric neonate newborn juvenile",
      "baby": "baby infant newborn neonate pediatric",
      "infant": "infant baby newborn neonate pediatric",
      "adult": "adult mature grownup",
      "elderly": "elderly geriatric aged senior",
      "woman": "woman female maternal",
      "man": "man male paternal",
      
      # Oncology & Pathology
      "cancer": "cancer oncology malignant tumor carcinoma neoplasm malignancy",
      "tumor": "tumor oncology malignant cancer carcinoma neoplasm tumor",
      "tumour": "tumour oncology malignant cancer carcinoma neoplasm tumour",
      "leukemia": "leukemia leukaemia hematologic malignancy lymphoma",
      "leukaemia": "leukaemia leukemia hematologic malignancy lymphoma",
      
      # Fluid & Circulatory System
      "blood": "blood plasma serum hematology wholeblood",
      "plasma": "plasma blood serum",
      "serum": "serum blood plasma",
      "heart": "heart cardiac cardiovascular myocardium",
      
      # Internal Organs & Systems
      "liver": "liver hepatic hepatitis cirrhosis",
      "kidney": "kidney renal nephrology renal",
      "kidneys": "kidneys renal nephrology renal",
      "brain": "brain neural neurological cortex cerebrospinal",
      "lung": "lung pulmonary respiratory lungs",
      "lungs": "lungs lung pulmonary respiratory",
      "gut": "gut gastrointestinal gi bowel intestinal",
      "stomach": "stomach gastrointestinal gastric"
    }
    
    expanded_tokens = []
    for tok in filtered_tokens:
      # Expand known clinical terms, otherwise pass the clean lemma through safely
      expanded_tokens.append(quick_ontology.get(tok, tok))
      
    processed_q = " ".join(expanded_tokens)
  
    # 4. STAGE 1: Hybrid Retrieval (BM25 + FAISS)
    # We pass the expanded clinical string so BM25 can hit exact text values
    q_vec = self.bi_encoder.encode([processed_q]).astype("float32")
    faiss.normalize_L2(q_vec)
    
    subset_ids = self.df.index.tolist()
    _, faiss_indices = self.index.search(q_vec, min(self.limit, len(subset_ids)))
    
    bm25_scores = self.bm25.get_scores(processed_q.split())
    bm25_indices = np.argsort(-bm25_scores)[:self.limit]
    
    candidates = list(set([subset_ids[i] for i in faiss_indices[0]] + [subset_ids[i] for i in bm25_indices]))
  
    # 5. STAGE 2: Protected Semantic Re-Ranking (Cross-Encoder)
    # We wrap the perfectly isolated and expanded entities inside a standard grammatical sentence
    ce_query_context = f"Does this biobank repository contain samples, data, or resources related to {processed_q}?"
    
    inputs = [(ce_query_context, " ".join(self.df_sem.loc[idx].values.astype(str))) for idx in candidates]
    scores = self.cross_encoder.predict(inputs)
    
    # 6. Mathematical Guard Gate
    probabilities = 1.0 / (1.0 + np.exp(-scores))
    results_series = pd.Series(probabilities, index=candidates)
    
    relevance_threshold = 0.01  # Safe 1% baseline cutoff for context-wrapped queries
    valid_results = results_series[results_series >= relevance_threshold]
    
    if valid_results.empty: 
      return pd.DataFrame(columns=self.df.columns)
      
    user_top_k = dsl.get("top_k", self.limit)
    top_results = valid_results.sort_values(ascending=False).head(user_top_k)
    
    return self.df.loc[top_results.index].assign(ce_score=top_results)
  #}
#}
