# biobank_grep.py
import pandas as pd
import pyarrow.parquet as pq
import numpy as np
import faiss
import json
import hashlib
import configparser
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

class BioBankGrep:
#{
    def __init__(self, config_path="girl.cfg"):
    #{
        config = configparser.ConfigParser()
        config.read(config_path)
        self.cfg = config['GLOBAL']
        
        with open(self.cfg.get("manifest_file"), "r") as f: self.manifest = json.load(f)
        self._verify_integrity()
        
        m_files = self.manifest["files"]
        self.vector_index = faiss.read_index(m_files["faiss"]["name"])
        self.df = pq.read_table(m_files["parquet"]["name"]).to_pandas()
        with open(m_files["schema"]["name"], "r") as f: self.schema = json.load(f)
        
        search_text = [" | ".join(map(str, row)) for row in self.df.values]
        self.bm25 = BM25Okapi([doc.lower().split() for doc in search_text])
        self.model = SentenceTransformer(self.manifest["model_name"])
        self.rrf_k = 60
    #}

    def _verify_integrity(self):
    #{
        for info in self.manifest["files"].values():
        #{
            h = hashlib.sha256()
            with open(info["name"], "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
            if h.hexdigest() != info["hash"]: 
                raise PermissionError(f"Integrity Mismatch: {info['name']}")
        #}
    #}

    def execute_query(self, dsl):
    #{
        query, filters, top_k = dsl.get("nlp", "").strip(), dsl.get("filters", {}), dsl.get("top_k", self.schema["default_top_k"])
        f_df = self.df.copy()
        for col, val in filters.items():
        #{
            if val is None: continue
            if isinstance(val, list): f_df = f_df[f_df[col].isin(val)]
            elif isinstance(val, tuple): f_df = f_df[(f_df[col] >= val[0]) & (f_df[col] <= val[1])]
        #}
        if f_df.empty or top_k <= 0 or not query: return f_df.head(max(0, top_k)).assign(rrf_score=0.0)

        subset_ids = f_df.index.tolist()
        q_vec = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_vec)
        
        v_scores = np.dot(np.array([self.vector_index.reconstruct(i) for i in subset_ids]), q_vec.T).flatten()
        k_scores = self.bm25.get_batch_scores(query.lower().split(), subset_ids)

        v_ranks, k_ranks = np.argsort(-v_scores), np.argsort(-k_scores)
        rrf_map = {idx: 0.0 for idx in subset_ids}
        for r, p in enumerate(v_ranks): rrf_map[subset_ids[p]] += 1.0/(self.rrf_k + r)
        for r, p in enumerate(k_ranks): rrf_map[subset_ids[p]] += 1.0/(self.rrf_k + r)

        final_s = pd.Series(rrf_map)
        winners = final_s.sort_values(ascending=False).index[:top_k]
        return f_df.loc[winners].assign(rrf_score=final_s[winners])
    #}
#}