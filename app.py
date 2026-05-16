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

                # 1. ---- DEFINE YOUR DESIRED COLUMN ORDER ------------
                # Columns will be displayed in this order from left-to-right
                desired_column_order = [
                  "name",
                  "repository_type",
                  "description",
                  "fees",
                  "url",
                  "email",
                  "phone",
                  "address",
                  "rrf_score",
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
                        "rrf_score": st.column_config.NumberColumn("Rank Score", format="%.4f")
                    }
                )
