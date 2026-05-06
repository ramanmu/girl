import argparse
import sys
from pm_search import BioBankGrep
from indexer import run_indexer

def is_non_negative(value):
#{
    ivalue = int(value)
    if ivalue < 0: raise argparse.ArgumentTypeError(f"{value} is negative.")
    return ivalue
#}

def run_girl():
#{
    # 1. BOOTSTRAP: Detect Indexing vs Search
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("-i", "--index", help="CSV path to re-index")
    pre_parser.add_argument("query", nargs="?", default=None, help="NLP query")
    pre_args, remaining = pre_parser.parse_known_args()

    if pre_args.index:
    #{
        try:
            run_indexer(pre_args.index)
        except Exception as e:
            print(f"Indexer Short-circuit Failed: {e}"); sys.exit(1)
        
        if pre_args.query is None:
            print("Indexing complete. Search skipped."); return 
    #}

    # 2. INITIALIZE BIOBANKGREP
    try:
        grep = BioBankGrep()
        schema = grep.schema
    except Exception as e:
        print(f"Error: Run with -i first to index data. {e}"); sys.exit(1)

    # 3. DYNAMIC DUAL-MODE PARSER
    parser = argparse.ArgumentParser(description="girl: The BioBankGrep Utility")
    parser.add_argument("query", help="NLP clinical search string")
    parser.add_argument("-k", "--top_k", type=is_non_negative, default=schema["default_top_k"])
    parser.add_argument("-i", "--index", help="Handled in bootstrap")

    for f in schema["filters"]:
    #{
        if f["type"] == "multi":
            parser.add_argument(f"--{f['column']}", nargs="*", help=f"Choices: {f['options']}")
        elif f["type"] == "range":
            parser.add_argument(f"--{f['column']}", nargs=2, type=float, help=f"Bounds: {f['min']} to {f['max']}")
    #}

    args = parser.parse_args(sys.argv[1:]) 
    
    # 4. CONSTRUCT DSL & EXECUTE
    dsl_filters = {}
    for f in schema["filters"]:
    #{
        val = getattr(args, f["column"])
        if val: dsl_filters[f["column"]] = tuple(val) if f["type"] == "range" else val
    #}

    results = grep.execute_query({"nlp": args.query, "filters": dsl_filters, "top_k": args.top_k})

    if not results.empty:
        print(f"\n--- BioBankGrep Results ({len(results)}) ---")
        print(results.to_string(index=False))
    else:
        print("No matches found.")
#}

if __name__ == "__main__":
    run_girl()