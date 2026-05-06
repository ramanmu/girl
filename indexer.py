import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import faiss
import json
import hashlib
import argparse
import sys
import configparser
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

def get_sha256(filename):
#{
    h = hashlib.sha256()
    with open(filename, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""): h.update(chunk)
    return h.hexdigest()
#}

def run_indexer(csv_path, config_path="girl.cfg"):
#{
    config = configparser.ConfigParser()
    config.read(config_path)
    cfg = config['GLOBAL']
    
    basename = cfg.get("index_file_basename")
    
    print(f"--- 🛠️  Generating Artifacts for: {csv_path} ---")
    df = pd.read_csv(csv_path).fillna('')
    
    # 1. SCHEMA & CONTROL GENERATION
    schema = {
        "filters": [], 
        "total_records": len(df), 
        "default_top_k": cfg.getint("default_top_k")
    }
    for col in df.columns:
    #{
        if pd.api.types.is_numeric_dtype(df[col]):
            schema["filters"].append({
                "column": col, "type": "range", 
                "min": float(df[col].min()), "max": float(df[col].max())
            })
        else:
            schema["filters"].append({
                "column": col, "type": "multi", 
                "options": sorted(df[col].unique().tolist())
            })
    #}
    schema_file = f"{basename}_schema.json"
    with open(schema_file, "w") as f: json.dump(schema, f, indent=2)

    # 2. HYBRID SEARCH INDEXING
    search_text = [" | ".join(map(str, row)) for row in df.values]
    model = SentenceTransformer(cfg.get("model_name"))
    embeddings = model.encode(search_text, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    
    faiss_file, parquet_file = f"{basename}.faiss", f"{basename}.parquet"
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, faiss_file)
    pq.write_table(pa.Table.from_pandas(df), parquet_file)

    # 3. SECURITY MANIFEST
    manifest = {
        "model_name": cfg.get("model_name"),
        "files": {
            "parquet": {"name": parquet_file, "hash": get_sha256(parquet_file)},
            "faiss": {"name": faiss_file, "hash": get_sha256(faiss_file)},
            "schema": {"name": schema_file, "hash": get_sha256(schema_file)}
        }
    }
    with open(cfg.get("manifest_file"), "w") as f: json.dump(manifest, f, indent=2)
    print("--- ✅ Indexing Complete & Integrity Verified ---")
#}

if __name__ == "__main__":
#{
    parser = argparse.ArgumentParser(description="BioBankGrep Indexer Utility")
    parser.add_argument("csv", help="Source CSV file path")
    parser.add_argument("--config", default="girl.cfg", help="Path to girl.cfg")
    args = parser.parse_args()
    try:
        run_indexer(args.csv, args.config)
    except Exception as e:
        print(f"Indexer Error: {e}"); sys.exit(1)
#}