import streamlit as st
import pandas as pd
import urllib.parse
from biobankgrep import BioBankGrep

# SEARCH ENGINE CACHE: Initialize engine (caches models in memory)
@st.cache_resource
def load_engine():
    return BioBankGrep()

engine = load_engine()
schema = engine.manifest

st.set_page_config(page_title="BioBank Discovery Engine", layout="wide")
st.title("🧬 BioBank Discovery Engine")

# Form Gate with Custom UI Layout ---
# By wrapping your columns in a form, we kill the callback race condition
# without destroying the split-pane and grid logic below it.
with st.form(key="search_form"):
    q_col, b_col = st.columns([85, 15], vertical_alignment="bottom")

    with q_col:
        query = st.text_input("Enter search:", placeholder="e.g., placental tissue", key="user_query_input")

    with b_col:
        submit_button = st.form_submit_button("Search", use_container_width=True)

# PROCESS QUERY: Inline execution (Only runs when button is explicitly clicked)
if submit_button:
    active_filters = st.session_state.get("active_filters", {})
    # Safely pull top_k, defaulting to 100 to ensure full 84-row sweep
    top_k = st.session_state.get("top_k", schema.get("default_top_k", 100))
    
    dsl = {"nlp": query.strip(), "filters": active_filters, "top_k": top_k}
    
    with st.spinner("Searching..."):
        # We push the DataFrame and row index back into session_state here 
        # so your grid list and split-pane code below this block works perfectly.
        st.session_state.search_results = engine.execute_query(dsl)
        st.session_state.selected_row_idx = 0

# --- UI: Sidebar Filters ---
st.sidebar.header("Data Filters")
if "active_filters" not in st.session_state:
    st.session_state.active_filters = {}

for f in schema["filters"]:
    col = f["column"]
    if f["type"] == "multi":
        selection = st.sidebar.multiselect(f"Select {col.title()}", options=f["options"])
        if selection: st.session_state.active_filters[col] = selection
    elif f["type"] == "substring":
        selection = st.sidebar.text_input(f"Search {col.title()} (Contains)")
        if selection: st.session_state.active_filters[col] = selection

st.sidebar.markdown("---")
st.sidebar.slider("Max Results", 5, 100, schema["default_top_k"], key="top_k")

st.divider()

def select_preview_row(row_index):
    st.session_state.selected_row_idx = row_index

def display_as_split_pane(ordered_results):
    list_pane, preview_pane = st.columns([40, 60], gap="xxsmall")

    with list_pane:
        st.markdown("### 📥 Repository")
        with st.container(height=450):
            for idx, row in ordered_results.reset_index(drop=True).iterrows():
                is_selected = (idx == st.session_state.selected_row_idx)
                # Show CE Score for visual ranking
                label = f"🧬 {row['name']}"
                st.button(
                    label, 
                    key=f"row_{idx}", 
                    type="primary" if is_selected else "secondary",
                    use_container_width=True, 
                    on_click=select_preview_row, 
                    args=(idx,)
                )

    with preview_pane:
        st.markdown("### 🔍 Overview")
        current_idx = st.session_state.selected_row_idx
        if current_idx < len(ordered_results):
            active_record = ordered_results.iloc[current_idx]
            with st.container(border=True, height=450):
                st.subheader(f"🧬 {active_record['name']}")
                st.caption(f"**Type:** {active_record['repository_type']} | **Fees:** {active_record['fees']}")
                st.divider()
                st.write(active_record['description'])
                st.divider()
                
                st.markdown("#### 📍 Contact")
                if active_record['url'] and active_record['url'] != "N/A":
                    clean_url = active_record['url'] if active_record['url'].startswith("http") else f"https://{active_record['url']}"
                    st.markdown(f"🔗 **Website:** [{active_record['name']}]({clean_url})")
                if active_record['email'] and active_record['email'] != "N/A":
                    st.markdown(f"✉️ **Email:** [{active_record['email']}](mailto:{active_record['email']})")
                if active_record['phone'] and active_record['phone'] != "N/A":
                    st.markdown(f"📞 **Tel:** {active_record['phone']}")
                if active_record['address'] and active_record['address'] != "N/A":
                    enc = urllib.parse.quote(active_record['address'])
                    maps_url = f"https://www.google.com/maps/search/?api=1&query={enc}"
                    st.markdown(f"🏢 **Address:** [{active_record['address']}]({maps_url})")
        else:
            st.info("Select a repository to inspect clinical metadata.")

if st.session_state.get("search_results") is not None:
    results = st.session_state.search_results
    if not results.empty:
        st.success(f"Found {len(results)} biobanks.")
        # Column order updated to use ce_score
        desired_cols = ["name", "repository_type", "description", "fees", "url", "email", "phone", "address", "ce_score"]
        ordered_results = results.reindex(columns=desired_cols, fill_value="N/A")
        display_as_split_pane(ordered_results)
    else:
        st.error("No biobanks matched your criteria.")
