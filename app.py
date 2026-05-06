import streamlit as st
import json
import os
from pm_search import BioBankGrep

# Load config for Title
import configparser
config = configparser.ConfigParser()
config.read('girl.cfg')
TITLE = os.environ.get("APP_TITLE", "🧬 BioBankGrep Portal")

st.set_page_config(page_title=TITLE, layout="wide")

@st.cache_resource
def load_grep(): return BioBankGrep()

def main():
#{
    st.title(TITLE)
    try:
        grep = load_grep()
    except Exception as e:
        st.error(f"Critical Integrity Fail: {e}"); return

    # Sidebar: Data Filter Controls
    st.sidebar.header("Instance Filters")
    dsl_filters = {}
    for f in grep.schema["filters"]:
    #{
        if f["type"] == "multi":
            sel = st.sidebar.multiselect(f"Select {f['column']}", f["options"])
            if sel: dsl_filters[f["column"]] = sel
        elif f["type"] == "range":
            sel = st.sidebar.slider(f"Range: {f['column']}", f["min"], f["max"], (f["min"], f["max"]))
            if sel[0] > f["min"] or sel[1] < f["max"]: dsl_filters[f["column"]] = sel
    #}
    
    top_k = st.sidebar.number_input("Limit", 0, grep.schema["total_records"], grep.schema["default_top_k"])
    query = st.text_input("Semantic Clinical Search:")

    if query:
        results = grep.execute_query({"nlp": query, "filters": dsl_filters, "top_k": top_k})
        if not results.empty:
            st.dataframe(results, use_container_width=True)
            st.download_button("Export Results", results.to_csv(index=False), "results.csv")
        else: st.info("No matching records found.")
    else:
        st.write("Awaiting query. Filtered data preview:")
        st.dataframe(grep.df.head(top_k))
#}

if __name__ == "__main__":
    main()