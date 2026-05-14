import streamlit as st
import json
from biobankgrep import BioBankGrep

# Initialize engine once (caches the models in memory)
@st.cache_resource
def load_engine():
    return BioBankGrep()

engine = load_engine()
schema = engine.schema

st.set_page_config(page_title="BioBank Search", layout="wide")
st.title("🧬 BioBank Discovery Engine")

# --- UI: Sidebar Filters ---
st.sidebar.header("Data Filters")
active_filters = {}

for f in schema["filters"]:
    col = f["column"]
    if f["type"] == "multi":
        selection = st.sidebar.multiselect(f"Select {col.title()}", options=f["options"])
        if selection: active_filters[col] = selection
    elif f["type"] == "substring":
        # Text box for fuzzy matching (like Address)
        selection = st.sidebar.text_input(f"Search {col.title()} (Contains)")
        if selection: active_filters[col] = [selection]

st.sidebar.markdown("---")
top_k = st.sidebar.slider("Max Results", min_value=5, max_value=100, value=schema["default_top_k"])

# --- UI: Main Search ---
query = st.text_input("Describe the biobank data you are looking for:", 
                      placeholder="e.g., pediatric samples in australia")

if st.button("Search", type="primary") or query:
    if not query:
        st.warning("Please enter a search term.")
    else:
        with st.spinner("Searching vectors and keywords..."):
            dsl = {
                "nlp": query,
                "filters": active_filters,
                "top_k": top_k
            }
            results = engine.execute_query(dsl)
            
            if results.empty:
                st.error("No biobanks matched your exact criteria.")
            else:
                st.success(f"Found {len(results)} highly relevant biobanks.")
                st.dataframe(
                    results.drop(columns=['description'], errors='ignore'), # Hide giant text blocks in the table
                    use_container_width=True,
                    hide_index=True
                )
