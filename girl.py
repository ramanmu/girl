import argparse
import sys
import json
from biobank_grep import BioBankGrep

def main():
    try:
        with open("manifest.json", "r") as f: schema = json.load(f)
    except FileNotFoundError:
        print("Error: manifest.json not found. Run 'python indexer.py' first.")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="BioBank Search CLI")
    parser.add_argument("query", type=str, help="Natural language search query")
    parser.add_argument("-k", "--top_k", type=int, default=schema["default_top_k"], help="Number of results")

    schema_map = {f["column"]: f for f in schema["filters"]}
    for col, f_schema in schema_map.items():
        if f_schema["type"] in ["multi", "substring"]:
            parser.add_argument(f"--{col}", nargs="+", help=f"Filter by {col} ({f_schema['type']})")

    args = parser.parse_args()
    
    dsl_filters = {}
    for col, f_schema in schema_map.items():
        user_vals = getattr(args, col, None)
        if user_vals:
            if f_schema["type"] == "multi":
                invalid = [v for v in user_vals if v not in f_schema["options"]]
                if invalid:
                    print(f"Error: Invalid options for --{col}: {invalid}\nValid: {f_schema['options']}")
                    sys.exit(1)
            dsl_filters[col] = user_vals

    dsl = {"nlp": args.query, "filters": dsl_filters, "top_k": args.top_k}
    
    engine = BioBankGrep()
    results = engine.execute_query(dsl)
    
    if results.empty:
        print("\nNo matching biobanks found.")
    else:
        print(f"\n--- BioBankGrep Results ({len(results)}) ---")
        #cols_to_show = [c for c in ['name', 'repository_type', 'description', 'country', 'address', 'rrf_score'] if c in results.columns]
        cols_to_show = [c for c in ['name', 'repository_type', 'country', 'address', 'rrf_score'] if c in results.columns]
        if 'rrf_score' not in cols_to_show: cols_to_show.append('rrf_score')
        print(results[cols_to_show].to_csv())

if __name__ == "__main__":
    main()
