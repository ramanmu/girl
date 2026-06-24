import streamlit as st
import pandas as pd
import urllib.parse
from biobankgrep import BioBankGrep

@st.cache_resource
def load_engine():
#{
  return BioBankGrep()
#}

engine = load_engine()
ui_filters = engine.filters

st.set_page_config(page_title="BioBank Discovery Engine", layout="wide")
st.title("🧬 BioBank Discovery Engine")

# 2. Session State Initialization
if "search_results" not in st.session_state: st.session_state.search_results = None
if "selected_row_idx" not in st.session_state: st.session_state.selected_row_idx = 0
if "current_query" not in st.session_state: st.session_state.current_query = ""

# 3. Master Execution Callback
def perform_search():
#{
  # Defensively fetch state variables to prevent Cold Start crashes
  current_q = st.session_state.get("current_query", "")
  top_k_val = st.session_state.get("top_k_slider", 100)

  dsl = {
    "nlp": current_q.strip(),
    "filters": {},
    "top_k": top_k_val
  }

  # Dynamically map the UI selections to the DSL using the schema index
  for idx, f in enumerate(ui_filters):
    selected_values = st.session_state.get(f"filter_{idx}", [])
    if selected_values: dsl["filters"][str(idx)] = selected_values

  if dsl["nlp"]: 
    with st.spinner("Executing semantic search..."):
      try:
        st.session_state.search_results = engine.execute_query(dsl)
        st.session_state.selected_row_idx = 0
      except RuntimeError as re:
        # Graceful, user-friendly UI integration for the API bottleneck
        st.warning(f"**AI Taxonomy Engine Failure**:  {str(re)}", icon="⚠️")
        st.info("**Please wait a moment and try your search again.**", icon="🔄")
        st.session_state.search_results = pd.DataFrame(columns=engine.df.columns) 
      except Exception as e:
        # Hard application crashes
        st.error(f"System Error: {str(e)}", icon="🚨")
        st.session_state.search_results = pd.DataFrame(columns=engine.df.columns) 
#}

# 4. UI: Sidebar Filters (Reactive)
st.sidebar.header("Data Filters")

for idx, f in enumerate(ui_filters):
  # Leverage the pre-computed ui_name from indexer.py
  display_name = f.get("ui_name", f.get("original_column", "").title())
  st.sidebar.multiselect(
    f"Select {display_name}", 
    options=f["options"],
    key=f"filter_{idx}",
    on_change=perform_search
  )

st.sidebar.markdown("---")
st.sidebar.slider("Max Results", 5, 100, 100, key="top_k_slider", on_change=perform_search)

# 5. Form Gate for Text Input
with st.form(key="search_form"):
  q_col, b_col = st.columns([85, 15], vertical_alignment="bottom")
  with q_col: st.text_input("Enter search:", placeholder="e.g., placental tissue", key="current_query")
  with b_col: submit_button = st.form_submit_button("Search", use_container_width=True)

if submit_button: perform_search()
st.divider()

# 6. Split-Pane Display Logic
def select_preview_row(row_index):
#{
  st.session_state.selected_row_idx = row_index
#}

def display_as_split_pane(ordered_results):
#{
  list_pane, preview_pane = st.columns([40, 60], gap="small")

  with list_pane:
    st.markdown("### 📥 Repository")
    with st.container(height=450):
      for idx, row in ordered_results.reset_index(drop=True).iterrows():
        is_selected = (idx == st.session_state.selected_row_idx)
        st.button(f"🧬 {row['name']}", key=f"row_{idx}", type="primary" if is_selected else "secondary", use_container_width=True, on_click=select_preview_row, args=(idx,))

  with preview_pane:
    st.markdown("### 🔍 Overview")
    current_idx = st.session_state.selected_row_idx
    
    if current_idx < len(ordered_results):
      active_record = ordered_results.iloc[current_idx]
      with st.container(border=True, height=450):
        st.subheader(f"🧬 {active_record['name']}")
        st.caption(f"**Type:** {active_record['repository_type']}")
        st.caption(f"**Fees:** {active_record['fees']}")
        st.divider()
        st.write(active_record['description'])
        st.divider()
        st.markdown("#### 📍 Contact")
        
        if active_record['url'] and active_record['url'] != "N/A":
          clean_url = active_record['url'] if str(active_record['url']).startswith("http") else f"https://{active_record['url']}"
          st.markdown(f"🔗 **Website:** [{active_record['name']}]({clean_url})")
        
        if active_record['email'] and active_record['email'] != "N/A":
          st.markdown(f"✉️ **Email:** [{active_record['email']}](mailto:{active_record['email']})")
        
        if active_record['phone'] and active_record['phone'] != "N/A":
          st.markdown(f"📞 **Tel:** {active_record['phone']}")
        
        if active_record['address'] and active_record['address'] != "N/A":
          enc = urllib.parse.quote(str(active_record['address']))
          maps_url = f"https://www.google.com/maps/search/?api=1&query={enc}" 
          st.markdown(f"🏢 **Address:** [{active_record['address']}]({maps_url})")
    else:
      st.info("Select a repository to inspect clinical metadata.")
#}

# 7. Render State
if st.session_state.search_results is not None:
  results = st.session_state.search_results
  if not results.empty:
    st.success(f"Found {len(results)} biobanks.")
    desired_cols = ["name", "repository_type", "description", "fees", "url", "email", "phone", "address", "ce_score"]
    available_cols = [c for c in desired_cols if c in results.columns]
    ordered_results = results.reindex(columns=available_cols, fill_value="N/A")
    display_as_split_pane(ordered_results)
  else:
    st.error("No biobanks matched your criteria.")
