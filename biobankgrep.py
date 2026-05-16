import pandas as pd
import numpy as np
import faiss
import pickle
import json
import configparser
import re
from sentence_transformers import SentenceTransformer
import spacy
from nltk.stem.snowball import SnowballStemmer

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

        # Initialize the stemmer
        self.stemmer = SnowballStemmer("english")
    #}

    def clean_human_query(self, raw_query):
    #{
      # 1. Keep a small, tight list of domain-specific custom stop words, "garbage words"
      # Make sure you use their 'stems'
      custom_stop_stems = {
        "biobank", "repositor", "sampl", "specimen", "data", "databas"
      }

      # Process the query with scispaCy
      clean_stems = []
      doc = self.nlp(raw_query.lower())
      for token in doc:
      #{
        # Check the token's Part of Speech (POS) tag!
        # We ONLY allow Nouns, Proper Nouns, and Adjectives.
        # This instantly destroys verbs like 'have', 'contain', 'want', 'get'.
        if token.pos_ in {"NOUN", "PROPN", "ADJ"}:
          if not token.is_stop:
            stemmed_word = self.stemmer.stem(token.lemma_)
              if stemmed_word not in custom_stop_stems:
                clean_stems.append(stemmed_word)
      #}
      return clean_stems
    #}

    def execute_query(self, dsl):
        raw_query = dsl.get("nlp", "").strip()
        filters = dsl.get("filters", {})
        top_k = dsl.get("top_k", self.schema["default_top_k"])
        
        # 1. Clean query
        clean_query_words = self.clean_human_query(raw_query)
        processed_query = " ".join(clean_query_words)
        
        # 2. Apply Pandas Filters
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
                
        if f_df.empty or top_k <= 0 or not processed_query: 
            return f_df.head(max(0, top_k)).assign(rrf_score=0.0)

        # 3. Vector Score
        subset_ids = f_df.index.tolist()
        q_vec = self.model.encode([processed_query]).astype("float32")
        faiss.normalize_L2(q_vec)
        v_scores, _ = self.index.search(q_vec, self.index.ntotal)
        v_scores = v_scores[0][subset_ids]

        # 4. BM25 Score
        tokenized_q = processed_query.split()
        k_scores = np.array([self.bm25.get_scores(tokenized_q)[i] for i in subset_ids])

        # 5. RRF & Lexical Multiplier
        v_ranks, k_ranks = np.argsort(-v_scores), np.argsort(-k_scores)
        rrf_map = {idx: 0.0 for idx in subset_ids}
        
        for r, p in enumerate(v_ranks): rrf_map[subset_ids[p]] += 1.0/(self.rrf_k + r)
        for r, p in enumerate(k_ranks): rrf_map[subset_ids[p]] += 1.0/(self.rrf_k + r)


        # --- THE STEMMED NLP SLEDGEHAMMER ---
        for idx in subset_ids:
            row_text = " ".join(map(str, f_df.loc[idx].values)).lower()

            row_doc = self.nlp(row_text)
            # Match the database row by turning every word into its root stem
            clean_row_stems = set([self.stemmer.stem(token.lemma_) for token in row_doc if not token.is_punct])

            match_count = sum(1 for stem in clean_query_words if stem in clean_row_stems)

            if len(clean_query_words) > 0 and match_count == 0:
                rrf_map[idx] = 0.0 # Hard drop
            else:
                rrf_map[idx] *= (1 + match_count)
        # ------------------------------------

        final_s = pd.Series(rrf_map)
        final_s = final_s[final_s > 0.0] # Threshold drop
        
        winners = final_s.sort_values(ascending=False).index[:top_k]
        return f_df.loc[winners].assign(rrf_score=final_s[winners])
