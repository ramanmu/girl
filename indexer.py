import pandas as pd
import faiss
import pickle
import json
import spacy
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import core_pipeline

def build_artifacts(df):
#{
  cfg = core_pipeline.load_config()
  base = cfg.get("index_file_basename")
  
  nlp = spacy.load(cfg.get("clinical_nlp_model"))
  bi_encoder = SentenceTransformer(cfg.get("bi_encoder_name"))
  vector_cols = [c.strip() for c in cfg.get("vector_columns").split(",")]
  
  print("Building BM25 Lexical Index...")
  # Create a corpus for BM25 and Vectors based on config vector_columns
  corpus = df[vector_cols].apply(lambda row: " ".join(row.astype(str)), axis=1).tolist()
  
  tokenized_corpus = [core_pipeline.clean_clinical_text(text, nlp) for text in corpus]
  bm25 = BM25Okapi(tokenized_corpus)
  
  with open(f"{base}_bm25.pkl", "wb") as f: pickle.dump(bm25, f)
  
  print("Building FAISS Vector Index...")
  vectors = bi_encoder.encode(corpus).astype("float32")
  faiss.normalize_L2(vectors)
  index = faiss.IndexFlatIP(vectors.shape[1])
  index.add(vectors)
  faiss.write_index(index, f"{base}.faiss")
  
  state_hash = core_pipeline.generate_state_hash(df)
  with open(f"{base}_meta.json", "w") as f:
  #{
    json.dump({"state_hash": state_hash}, f)
  #}
  
  df.to_pickle(f"{base}_df.pkl")
  print("Indexing Complete.")
#}
