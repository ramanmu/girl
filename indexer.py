import pandas as pd
import numpy as np
import faiss
import pickle
import json
import configparser
import re
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import spacy

nlp = spacy.load("en_core_sci_sm");

def build_index():
    print("--- 🛠️  Generating Artifacts ---")
    cfg = configparser.ConfigParser()
    cfg.read("girl.cfg")
    cfg = cfg["GLOBAL"]

    # 1. LOAD & CLEAN DATA
    df = pd.read_csv(cfg.get("csv_file")).fillna("")

    if 'fees' in df.columns:
        df['fees'] = df['fees'].astype(str).str.strip()

    if 'repository_type' in df.columns:
        def clean_repo_type(text):
            matches = re.findall(r'RepositoryType:\s*([^>\]]+)', str(text))
            return ", ".join(matches) if matches else str(text)
        df['repository_type'] = df['repository_type'].apply(clean_repo_type)

    # 2. BUILD SCHEMA
    schema = {"filters": [], "total_records": len(df), "default_top_k": cfg.getint("default_top_k")}
    cat_filters = [c.strip() for c in cfg.get("category_filters", "").split(",") if c.strip()]
    txt_filters = [c.strip() for c in cfg.get("text_filters", "").split(",") if c.strip()]

    for col in df.columns:
        if col in cat_filters:
            schema["filters"].append({"column": col, "type": "multi", "options": sorted(df[col].unique().tolist())})
        elif col in txt_filters:
            schema["filters"].append({"column": col, "type": "substring"})

    # 3. BUILD SEMANTIC DOCUMENTS
    vector_cols = [c.strip() for c in cfg.get("vector_columns", "").split(",") if c.strip()]
    documents = []
    for idx, row in df.iterrows():
        doc_parts = [str(row[col]) for col in vector_cols if col in df.columns]
        documents.append(" ".join(doc_parts))

    # 4. VECTORIZATION (FAISS)
    print(f"Loading {cfg.get('model_name')}...")
    model = SentenceTransformer(cfg.get("model_name"))
    embeddings = model.encode(documents, show_progress_bar=True).astype("float32")
    
    index = faiss.IndexFlatIP(embeddings.shape[1])
    faiss.normalize_L2(embeddings)
    index.add(embeddings)

    # 5. KEYWORD INDEX (BM25) via Lemmatization
    tokenized_docs = [doc.lower().split() for doc in documents]
    print("Lemmatizing documents for BM25...");
    tokenized_docs = [];

    # process documents in bulk for performance
    for doc in nlp.pipe(documents, disable=["ner", "parser"]):
    # Disabling 'ner' and 'parser' makes this run 10x faster as we only need the
    # dictionary roots, not full sentence diagramming.
    #{
      #Extract the lemma (root word), ignore punctuation and spaces
      tokens = [token.lemma_.lower() for token in doc if not token.is_punct and not token.is_space];
      tokenized_docs.append(tokens);
    #}
    bm25 = BM25Okapi(tokenized_docs)

    # 6. SAVE ARTIFACTS
    base = cfg.get("index_file_basename")
    faiss.write_index(index, f"{base}.faiss")
    with open(f"{base}_bm25.pkl", "wb") as f: pickle.dump(bm25, f)
    df.to_pickle(f"{base}_df.pkl")
    with open(cfg.get("manifest_file"), "w") as f: json.dump(schema, f, indent=2)
    print("--- ✅ Indexing Complete ---")

if __name__ == "__main__":
    build_index()
