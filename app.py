from biobankgrep import BioBankGrep
import json
import streamlit as st
from types import SimpleNamespace

# Initialize engine once (caches the models in memory)
st.set_page_config(page_title="ISBER BioBank Search", layout="wide")
st.title("🧬 ISBER BioBank IRL - BioBankAI Discovery Engine")

@st.cache_resource
def load_engine(): return BioBankGrep()

# ---- Load global config -------------------
cfg_file = "girl.cfg";
with open(cfg_file, "r") as f: rc = json.load(f);
cfg = json.loads(json.dumps(rc), object_hook=lambda d: SimpleNamespace(**d));

# ---- Load manifest file --------------------------
man_file = cfg.manifest_file;
with open(man_file, "r") as f: rc = json.load(f);
man = json.loads(json.dumps(rc), object_hook=lambda d: SimpleNamespace(**d));

# ---- Load filters -------------------------------
fil_file = man.filters_file;
with open(fil_file, "r") as f: filters = json.load(f);

engine = load_engine()

# --- UI: Sidebar Filters ---
st.sidebar.header("Data Search Filters")
active_filters = {}
for f in filters:
#{
  c = f["column"]
  if f["type"] == "multi":
    selection = st.sidebar.multiselect(f"Select one or values for {c}", options=f["options"])
    if selection: active_filters[c] = [selection];
  elif f["type"] == "substring": # Text box for fuzzy matching (like Address)
    selection = st.sidebar.text_input(f"Enter filter terms for {c} ")
    if selection: active_filters[c] = selection;
#}

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Max Results", min_value=5, max_value=100, value=cfg.default_top_k);

# --- UI: Main Search ---
query = st.text_input("Describe the biobank data you are looking for:", 
                      placeholder="e.g., pediatric samples in australia")

if st.button("Search", type="primary") or query:
  if not query:
    st.warning("Please enter a search term.")
  else:
    with st.spinner("Searching ISBER Bio-bank database..."):
      results = engine.execute_query(query, active_filters, top_k);
            
    if results.empty:
      st.error("No biobanks matched your exact criteria.")
    else:
      st.success(f"Found {len(results)} biobanks.")
      desired_column_order = [
        "name",
        "repository_type",
        "description",
        "fees",
        "url",
        "email",
        "phone",
        "address",
        "Ranking",
      ]

      # Filter and re-order the dataframe rows dynamically
      # (We use errors='ignore' just in case a column name has a typo)
      ordered_results = results.reindex(columns=desired_column_order, fill_value="N/A")


      # 2. RENDER THE UPGRADED DATAFRAME WITH CONFIG
      st.dataframe(
        ordered_results,
        use_container_width=True,
        hide_index=True,
        row_height=100,
        column_config={
          "name": st.column_config.TextColumn("Name", width="medium"),
          "repository_type": st.column_config.TextColumn("Type", width="medium"),
          "description": st.column_config.TextColumn("Description", width="large"),
          "fees": st.column_config.TextColumn("Fees?", width="small"),
          "url": st.column_config.TextColumn("URL", width="large"),
          "email": st.column_config.TextColumn("Email", width="large"),
          "phone": st.column_config.TextColumn("Tel", width="medium"),
          "address": st.column_config.TextColumn("Address", width="large"),
          "Ranking": st.column_config.NumberColumn("Rank Score", format="%.4f")
        }
      )
