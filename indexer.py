import configparser
import json
import faiss
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import os
import pandas as pd
import pickle
import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import spacy
from types import SimpleNamespace

def clean_repository_type(t):
#{
  m = re.findall(r'RepositoryType:\s*([^>\]]+)', str(t))
  return ", ".join(m) if m else str(t)
#}


def main(cfg_file="girl.cfg"):
#{
  # ---- LOAD APP CONFIG ------------------
  with open(cfg_file, "r") as f: r = json.load(f);
  cfg = json.loads(json.dumps(r), object_hook=lambda d: SimpleNamespace(**d));

  # ---- LOAD, SCRUB, AND SAVE  BIOBANK DATA -------------
  data_file = cfg.data_file;
  base  = os.path.splitext(os.path.basename(data_file))[0];
  df = pd.read_csv(data_file).fillna("");
  df['docs'] = df.astype(str).agg(" ".join, axis=1);

  c = 'fees'; # fix the fees column data
  if c in df.columns: df[c] = df[c].astype(str).str.strip()

  c = 'repository_type';  # fix the repository type column data
  if c in df.columns: df[c] = df[c].apply(clean_repository_type);

  df_file = f"{base}_df.pkl";
  df.to_pickle(df_file)

  # ---- EXTRACT AND SAVE THE FILTERS --------------------------
  filter_rules = [];
  for f in cfg.filters:
  #{
    if f.column not in df.columns: continue;
    if f.type in ['mono', 'multi']: f.options = sorted(df[c].unique().tolist());
    filter_rules.append(f.__dict__);
  #}

  filters_file = f"{base}_filters.json";
  with open(filters_file, "w") as f: json.dump(filter_rules, f, indent=2);
    
  # ------ COMPUTE AND SAVE BM25 ---------------------------
  sdl = df['docs'].tolist();
  nlp = spacy.load(cfg.nlp_model.strip("'\""));
  sbs = SnowballStemmer("english");
  tokens = [];
  for d in nlp.pipe([w.lower() for w in sdl], disable=['ner', 'parser']):
  #{
    s = [sbs.stem(t.lemma_) for t in d if not t.is_punct and not t.is_space];
    tokens.append(s);
  #}
  bm25 =  BM25Okapi(tokens);

  bm25_file = f"{base}_bm25.pkl";
  with open(bm25_file, "wb") as f: pickle.dump(bm25, f)

  # ---- VECTORIZE, INDEX, AND SAVE BIOBANK EMBEDDINGS ------------------
  bi_encoder = SentenceTransformer(cfg.bi_encoder);
  embeddings = bi_encoder.encode(sdl, show_progress_bar=True, convert_to_numpy=True);
  faiss.normalize_L2(embeddings);

  # ---- INDEX THE VECTOR EMBEDDINGS
  dim = embeddings.shape[1];
  faiss_index = faiss.IndexFlatL2(dim);
  faiss_index.add(embeddings);

  faiss_file = f"{base}.faiss";
  faiss.write_index(faiss_index, faiss_file)

  # ------ SET UP THE ARTIFACT MANIFEST --------------
  print("--- 🛠️  Generating and saving Artifacts ---")
  manifest = {
    "df_file": df_file,
    "filters_file": filters_file,
    "bm25_file": bm25_file,
    "faiss_file": faiss_file,
    "total_records": len(df)
  };
  with open(cfg.manifest_file, "w") as f: json.dump(manifest, f, indent=2)
 
  print("--- ✅ Indexing Complete ---")
#}

if __name__ == "__main__": main();
