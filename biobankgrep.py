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
        raw_query = dsl.get("nlp", "").strip().lower()

        # 1. Base Cleanup via your existing spaCy pipeline
        raw_tokens = core_pipeline.clean_clinical_text(raw_query, self.nlp)

        # 2. Comprehensive Domain Blacklist
        domain_blacklist = {
            "biobank", "biobanks", "repository", "repositories",
            "sample", "samples", "data", "specimen", "specimens",
            "cohort", "cohorts", "collection", "collections",
            "study", "studies", "protocol", "protocols", "patient", "patients",
            "donor", "donors", "human", "investigator", "investigators"
        }
        filtered_tokens = [tok for tok in raw_tokens if tok not in domain_blacklist]

        if not filtered_tokens:
            return pd.DataFrame(columns=self.df.columns)

        # 3. Target Clinical Synonym Expansion (The Quick Ontology)
        quick_ontology = {
            "child": "child pediatric paediatric children neonate newborn juvenile",
            "children": "children child pediatric paediatric neonate newborn juvenile",
            "baby": "baby infant newborn neonate pediatric",
            "infant": "infant baby newborn neonate pediatric",
            "adult": "adult mature grownup",
            "elderly": "elderly geriatric aged senior",
            "woman": "woman female maternal",
            "man": "man male paternal",
            "cancer": "cancer oncology malignant tumor carcinoma neoplasm malignancy",
            "tumor": "tumor oncology malignant cancer carcinoma neoplasm",
            "tumour": "tumour oncology malignant cancer carcinoma neoplasm",
            "leukemia": "leukemia leukaemia hematologic malignancy lymphoma",
            "leukaemia": "leukaemia leukemia hematologic malignancy lymphoma",
            "blood": "blood plasma serum hematology wholeblood",
            "plasma": "plasma blood serum",
            "serum": "serum blood plasma",
            "heart": "heart cardiac cardiovascular myocardium",
            "liver": "liver hepatic hepatitis cirrhosis",
            "kidney": "kidney renal nephrology",
            "kidneys": "kidneys renal nephrology",
            "brain": "brain neural neurological cortex cerebrospinal",
            "lung": "lung pulmonary respiratory lungs",
            "lungs": "lungs lung pulmonary respiratory",
            "gut": "gut gastrointestinal gi bowel intestinal",
            "stomach": "stomach gastrointestinal gastric"
        }

        expanded_tokens = []
        for tok in filtered_tokens:
            expanded_tokens.append(quick_ontology.get(tok, tok))
        print(f"DEBUG expanded tokens: {expanded_tokens}")

        # --- THE DECOUPLING CRITICAL STEP ---
        # Stage 1 gets the literal synonym explosion to maximize recall
        processed_q = " ".join(expanded_tokens)

        # Stage 2 gets the clean, unmodified natural language phrase
        clean_natural_phrase = " ".join(filtered_tokens)

        # 4. STAGE 1: Hybrid Retrieval (BM25 + FAISS)
        q_vec = self.bi_encoder.encode([processed_q]).astype("float32")
        faiss.normalize_L2(q_vec)

        subset_ids = self.df.index.tolist()
        _, faiss_indices = self.index.search(q_vec, min(self.limit, len(subset_ids)))

        bm25_scores = self.bm25.get_scores(processed_q.split())
        bm25_indices = np.argsort(-bm25_scores)[:self.limit]
        print(f"DBG bm25_scores: {bm25_scores}")

        # Force a stable, sorted order to guarantee identical PyTorch batching
        raw_candidates = set([subset_ids[i] for i in faiss_indices[0]] + [subset_ids[i] for i in bm25_indices])
        candidates = sorted(list(raw_candidates))

        # 5. STAGE 2: Protected Semantic Re-Ranking (Cross-Encoder)
        # Use a brutal, minimal scaffold so the model must evaluate the keyword itself
        ce_query_context = f"Search target: {clean_natural_phrase}"
        print(f"DEBUG ce_query_context: {ce_query_context}")

        inputs = [(ce_query_context, " ".join(self.df_sem.loc[idx].values.astype(str))) for idx in candidates]
        scores = self.cross_encoder.predict(inputs)
        print(f"DEBUG ce_scores: {scores}")

        # 6. Refined Mathematical Guard Gate
        probabilities = 1.0 / (1.0 + np.exp(-scores))
        print(f"DEBUG ce_probabilities: {probabilities}")
        results_series = pd.Series(probabilities, index=candidates)

        # Raised safely to 25%. Clean, targeted prompts prevent logit compression.
        relevance_threshold = 0.25
        valid_results = results_series[results_series >= relevance_threshold]

        if valid_results.empty:
            return pd.DataFrame(columns=self.df.columns)

        user_top_k = dsl.get("top_k", self.limit)
        top_results = valid_results.sort_values(ascending=False).head(user_top_k)

        return self.df.loc[top_results.index].assign(ce_score=top_results)
#}
