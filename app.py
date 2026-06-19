import json
import pandas as pd
import streamlit as st
from biobankgrep import BioBankGrep

st.set_page_config(page_title="BioBank Search", layout="wide")
st.title("🧬 BioBank Discovery Engine")

# Initialize engine once (caches the models in memory)
@st.cache_resource
def load_engine():
  return BioBankGrep()

engine = load_engine()
schema = engine.schema

def execute_search (query, active_filters, top_k):
#{
  dsl = { "nlp": query, "filters": active_filters, "top_k": top_k }
  with st.spinnger("Searching..."):
    st.session_state.search_results = engine.execute_query(dsl);
    st.session_state.selected_row_idx = 0;
#}

def select_preview_row (row_index):
#{
  st.session_state.selected_row_idx = row_index;
#}

def display_as_split_pane (ordered_results):
#{
  # CONSTRUCT THE SPLIT-PANES: 40% LIST, 60% PREVIEW CARD
  list_pane, preview_pane = st.columns([40, 60], gap="medium");

  # LEFT-PANE (40%): HIGH-DENSITY SEARCH RESULTS LIST
  # --- LEFT PANE: INTERACTIVE LISTING ITEMS ---
  with list_pane:
    st.markdown("### 📥 Matching Repositories")
      
    # Create a scrollable wrapper frame for the list items
    with st.container(height=450):
      for idx, row in ordered_results.reset_index(drop=True).iterrows():
        is_selected = (idx == st.session_state.selected_row_idx)
        btn_type = "primary" if is_selected else "secondary"
        label = f"🧬 {row['name']} | Score: {row['rrf_score']:.3f}"
        st.button(
          label,
          key=f"row_{idx}",
          type=btn_type,
          use_container_width=True,
          on_click=select_preview_row,
          args=(idx,)
        )

  # --- RIGHT PANE: RICH PREVIEW CARD ---
  with preview_pane:
    st.markdown("### 🔍 Preview")
    current_row_num = st.session_state.selected_row_idx
    if current_row_num < len(ordered_results):
    #{
      active_record = ordered_results.iloc[current_row_num]

      # Render the untruncated card layout using beautiful Markdown containers
      with st.container(border=True, height=450, key=f"preview_scroll_ctx_{current_row_num}"):
      #{
        st.subheader(f"🧬 {active_record['name']}")
        st.caption(f"**Type:** {active_record['repository_type']}")
        st.caption(f"**Fees:** {active_record['fees']}")

        st.divider()

        # This native markdown block handles full word-wrapping dynamically without bugs
        st.markdown("#### 📋 Clinical Meta Data")
        st.write(active_record['description'])

        st.divider()

        # Operational Metadata Fields grouped tightly
        st.markdown("#### 📍 Contact")
        #st.write(f"💰 **Fees:** {active_record['fees']}")
        st.write(f"🏢 **Address:** {active_record['address']}")

        # Active asset anchor tags
        if active_record['url'] and active_record['url'] != "N/A":
        #{
          raw_url = active_record['url'].strip();

          # Force absolute path so streamlit won't treat it as a relative/local page.
          if raw_url.startswith(("http://", "https://")):
            clean_url = raw_url;
          else:
            clean_url = f"https://{raw_url}"; 

          # Display the URL hyperlink 
          st.markdown(f"🔗 [Repository Website]({clean_url})")
        #}
      #}
    #}
    else: st.info("Select a repository from the left panel listing to inspect its complete clinical metadata sheet.");
#}

# --- UI: Sidebar Filters ---
st.sidebar.header("Data Filters")
active_filters = {}

for f in schema["filters"]:
  col = f["column"]
  if f["type"] == "multi":
    selection = st.sidebar.multiselect(f"Select {col.title()}", options=f["options"], on_change=execute_search)
    if selection: active_filters[col] = selection
  elif f["type"] == "substring": # Text box for fuzzy matching (like Address)
    selection = st.sidebar.text_input(f"Search {col.title()} (Contains)", on_change=execute_search)
    if selection: active_filters[col] = [selection]

st.sidebar.markdown("---")

top_k = st.sidebar.slider("Max Results", min_value=5, max_value=100, value=schema["default_top_k"], on_change=execute_search)

# INITIALIZE SESSION STATE ON STARTUP
if "search_results" not in st.session_state:
  st.session_state.search_results = None;
if "selected_row_idx" not in st.session_state:
  st.session_state.selected_row_idx = 0;

# MAIN SEARCH
query = st.text_input("Enter search criteria", placeholder="e.g., pediatric samples", on_change=execute_search)


st.button("Search", type="primary", on_click=execute_search);
st.divider()

if st.session_state.search_results is not None:
  results = st.session_state.search_results;
  if not results.empty:
    st.success(f"Found {len(results)} biobanks.")
    desired_column_order = [
      "name", "repository_type", "description", "fees",
      "url", "email", "phone", "address", "rrf_score",
    ]
    ordered_results = results.reindex(columns=desired_column_order, fill_value="N/A")
    display_as_split_pane(ordered_results);
  else:
    st.error("No biobanks matched your exact criteria.")
                  
#                # 2. RENDER THE UPGRADED DATAFRAME WITH CONFIG
#                st.dataframe(
#                    ordered_results,
#                    use_container_width=True,
#                    hide_index=True,
#                    #row_height=100,
#                    column_config={
#                        "name": st.column_config.TextColumn("Name", width="medium"),
#                        "repository_type": st.column_config.TextColumn("Type", width="medium"),
#                        "description": st.column_config.TextColumn("Description", width="large"),
#                        "fees": st.column_config.TextColumn("Fees?", width="small"),
#                        "url": st.column_config.TextColumn("URL", width="large"),
#                        "email": st.column_config.TextColumn("Email", width="large"),
#                        "phone": st.column_config.TextColumn("Tel", width="medium"),
#                        "address": st.column_config.TextColumn("Address", width="large"),
#                        "rrf_score": st.column_config.NumberColumn("Rank Score", format="%.4f")
#                    }
#                )
#
#        # Inside app.py - Render as high-fidelity result cards
#        for idx, row in ordered_results.iterrows():
#        # Creates a beautiful visually isolated card container
#          with st.container(border=True):
#            # Title bar shows name and type clearly
#            st.subheader(f"🧬 {row['name']}")
#            st.caption(f"**Type:** {row['repository_type']} | **Fees:** {row['fees']} | **Match Score:** {row['rrf_score']:.4f}")
#            st.markdown(row['description'])
#            if row['url'] and row['url'] != "N/A":
#              raw_url = row['url'].strip();
#              # Force absolute path so streamlit won't treat it as a relative/local page.
#              if raw_url.startswith(("http://", "https://")):
#                clean_url = raw_url;
#              else:
#                clean_url = f"https://{raw_url}"; 
#              # Display the URL hyperlink 
#              st.markdown(f"🔗 [Repository Website]({clean_url})")

