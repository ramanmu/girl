import re
import pandas as pd
import faiss
import pickle
import json
import spacy
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import core_pipeline

def compile_filters(df: pd.DataFrame, output_path: str) -> pd.DataFrame:
#{
  """
  Extracts unique values, creates strictly named proxy '_filter' columns 
  for fast querying, and outputs a dynamic, schema-driven manifest.
  """
  filters = []

  # filter for fees
  col = 'fees'
  if col in df.columns:
    proxy = 'fees_filter'
    raw_fees = df[col].fillna('n/a').astype(str).str.strip().str.lower()
    no_fee_keywords = ['0', 'none', 'no', 'free', 'n/a', 'false', 'na']
    df[col] = raw_fees.apply(lambda x: 'No' if x in no_fee_keywords else 'Yes')
    df[proxy] = df['fees']
        
    filters.append({
      "original_column": col,
      "filter_column": proxy,
      "type": "multi",
      "predicate": "exact_match",
      "ui_name": "Fees",
      "options": ["Yes", "No"]
    })

  # filter for addresses
  col = 'address'
  proxy = 'address_filter'
  if col in df.columns:
    clean_addresses = df[col].fillna('').astype(str).str.strip()
    df[proxy] = clean_addresses.apply(lambda x: x.split(',')[-1].strip() if ',' in x else x)
    unique_countries = sorted([c for c in df[proxy].unique() if c and len(c) > 1 and c.lower() != 'n/a'])
        
    filters.append({
      "original_column": col,
      "filter_column": proxy,
      "type": "multi",
      "predicate": "exact_match",
      "ui_name": "Country",
      "options": unique_countries
    })

  # filter for repository types
  col = 'repository_type'
  proxy = 'repository_type_filter'
  if col in df.columns:
    clean_types = df[col].dropna().astype(str).str.strip()
    df[proxy] = clean_types
    all_types = clean_types.str.split(',').explode().str.strip()
    unique_types = sorted([x for x in all_types.unique() if x and x.lower() != 'n/a'])
        
    filters.append({
      "original_column": col,
      "filter_column": proxy,
      "type": "multi",
      "predicate": "contains_any",
      "ui_name": "Repository Type",
      "options": unique_types
    })

  with open(output_path, "w") as f: json.dump(filters, f, indent=2)
  print(f"Successfully compiled and saved filters to {output_path}.")
  return df
#}

def build_artifacts():
#{
  cfg = core_pipeline.load_config()
  nlp = spacy.load(cfg.get("clinical_nlp_model"))
  
  # Golden Source
  df = pd.read_csv("biobank_data.csv").fillna("").reset_index(drop=True)
  # Standardize display columns
  for col in df.select_dtypes(include=["object", "string"]).columns:
    df[col] = df[col].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    df[col] = df[col].str.replace(r"\s+([.,;:?!])", r"\1", regex=True)

  # Repository Type Sanitization
  if 'repository_type' in df.columns:
    def clean_repo_type(text):
      matches = re.findall(r'RepositoryType:\s*([^>\]]+)', str(text))
      return ", ".join(matches) if matches else str(text)
    df['repository_type'] = df['repository_type'].apply(clean_repo_type)

  # Create the semantic context column for the vectorization
  vec_cols = [c.strip() for c in cfg.get("vector_columns").split(",")]
  semantic_context = pd.Series("", index=df.index)
  for col in vec_cols:
    if col not in df.columns: continue
    clean_text = df[col].fillna("").apply(core_pipeline.get_indexable_text)
    if col == 'description': clean_text = clean_text.astype(str).str.slice(0, 3000)
    else: clean_text = clean_text.astype(str)
    semantic_context += clean_text + " | "
  df['sem_context'] = semantic_context.str.rstrip(" | ")
  
  # Compute and save the BM25 scores based on the unified semantic context
  corpus = df['sem_context'].tolist()
  tokenized = [core_pipeline.clean_clinical_text(t, nlp) for t in corpus]
  with open(f"biobank_bm25.pkl", "wb") as f: pickle.dump(BM25Okapi(tokenized), f)
  
  # Save FAISS
  bi_encoder = SentenceTransformer(cfg.get("bi_encoder_name"))
  vectors = bi_encoder.encode(corpus).astype("float32")
  faiss.normalize_L2(vectors)
  index = faiss.IndexFlatIP(vectors.shape[1])
  index.add(vectors)
  faiss.write_index(index, f"biobank_faiss.faiss")
  
  # Compute filter, filter columns, and save golden source dataframe
  fdf = compile_filters(df, 'biobank_filters.json')
  fdf.to_pickle(f"biobank_df.pkl")

  # Compute and save data md5 digest
  state_hash = core_pipeline.generate_state_hash(fdf)
  with open(f"biobank_hash.json", "w") as f: json.dump({"state_hash": state_hash}, f)
  print("Indexing Complete.")
#}

if __name__ == "__main__": build_artifacts()
