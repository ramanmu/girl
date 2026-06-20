import pandas as pd
import faiss
import pickle
import json
import spacy
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import core_pipeline

def clean_repo_type(text):
#{
  # 1. Look for the label "RepositoryType:" (case-insensitive)
  match = re.search(r'(?i)repository\s*type\s*:\s*(.*)', str(text))
    
  if match:
  # Get the captured group (the content after the colon)
  # Strip away the common "garbage" characters from the end
  # This removes trailing '>', ']', etc.
  content = match.group(1)
  return content.strip(' >]')
    
  # If no match, return the original text
  return str(text)
#}


def build_artifacts():
#{
  cfg = core_pipeline.load_config()
  base = cfg.get("index_file_basename")
  nlp = spacy.load(cfg.get("clinical_nlp_model"))
  
  # 1. Gold Source
  df = pd.read_csv(cfg.get("csv_file")).fillna("").reset_index(drop=True)
  # Standardize display columns
  for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df[col] = df[col].str.replace(r"\s+([.,;:?!])", r"\1", regex=True)

  # Repository Type Sanitization
  if 'repository_type' in df.columns:
    df['repository_type'] = df['repository_type'].apply(clean_repo_type)

  
  # 2. Semantic Registry
  df_semantic = df.copy()
  vec_cols = [c.strip() for c in cfg.get("vector_columns").split(",")]
  for col in vec_cols:
  #{
    df_semantic[col] = df_semantic[col].apply(core_pipeline.get_indexable_text)
  #}
  
  # 3. Artifact Generation
  # BM25 & FAISS only use the semantic space
  corpus = df_semantic[vec_cols].apply(lambda row: " ".join(row.astype(str)), axis=1).tolist()
  tokenized = [core_pipeline.clean_clinical_text(t, nlp) for t in corpus]
  
  # Save BM25
  with open(f"{base}_bm25.pkl", "wb") as f: pickle.dump(BM25Okapi(tokenized), f)
  
  # Save FAISS
  bi_encoder = SentenceTransformer(cfg.get("bi_encoder_name"))
  vectors = bi_encoder.encode(corpus).astype("float32")
  faiss.normalize_L2(vectors)
  index = faiss.IndexFlatIP(vectors.shape[1])
  index.add(vectors)
  faiss.write_index(index, f"{base}.faiss")
  
  # 4. Atomic Sync
  df.to_pickle(f"{base}_display_df.pkl")
  df_semantic.to_pickle(f"{base}_semantic_df.pkl")
  
  state_hash = core_pipeline.generate_state_hash(df)
  with open(f"{base}_meta.json", "w") as f: json.dump({"state_hash": state_hash}, f)
  print("Indexing Complete.")
#}

if __name__ == "__main__": build_artifacts()
