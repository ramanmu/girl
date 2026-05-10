import streamlit as st
import configparser
import os
from biobank_grep import BioBankGrep

config = configparser.ConfigParser()
config.read('girl.cfg')
TITLE = os.environ.get("APP_TITLE", "🧬 BioBankGrep Explorer")

st.set_page_config(page_title=TITLE, layout="wide")

@st.cache_resource
def load_grep(): 
    return BioBankGrep()

def main():
#{
    st.title(TITLE)
    
    # UI Staleness check
    cfg = config['GLOBAL']
    csv_file = cfg.get("csv_file")
    manifest_file = cfg.get("manifest_file")
    if not os.path.exists(manifest_file) or (os.path.exists(csv_file) and os.path.getmtime(csv_file) > os.path.getmtime(manifest_file)):
        st.warning("⚠️ The source CSV has been updated. Please run `python girl.py -i` in your terminal to refresh the artifacts for the Web UI.")

    try:
        grep = load_grep()
    except Exception as e:
        st.error(f"Engine Load Failed: Run `python girl.py -i` first. Detail: {e}"); return

    st.sidebar.header("Metadata Filters")
    dsl_filters = {}
    for f in grep.schema["filters"]:
    #{
        if f["type"] == "multi":
            sel = st.sidebar.multiselect(f"Filter {f['column']}", f["options"])
            if sel: dsl_filters[f["column"]] = sel
        elif f["type"] == "range":
            sel = st.sidebar.slider(f"Range: {f['column']}", f["min"], f["max"], (f["min"], f["max"]))
            if sel[0] > f["min"] or sel[1] < f["max"]: dsl_filters[f["column"]] = sel
    #}
    
    top_k = st.sidebar.number_input("Display Limit", 1, grep.schema["total_records"], grep.schema["default_top_k"])
    query = st.text_input("Enter synopsis search terms (e.g., 'pediatric oncology biobank'):")

    if query:
        results = grep.execute_query({"nlp": query, "filters": dsl_filters, "top_k": top_k})
        if not results.empty:
            st.dataframe(results, use_container_width=True)
            st.download_button("Export results to CSV", results.to_csv(index=False), "biobank_search.csv")
        else: 
            st.info("No matching biobanks found.")
    else:
        st.write("Displaying filtered preview:")
        st.dataframe(grep.df.head(top_k))
#}

if __name__ == "__main__": 
    main()