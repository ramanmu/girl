import argparse
from biobankgrep import BioBankGrep
import json
import sys
from types import SimpleNamespace

def main():
#{
  # ---- Load config file ----------------------
  cfg_file = "girl.cfg";
  try:
    with open(cfg_file, "r") as f: r = json.load(f);
  except FileNotFoundError:
    print(f"Error: Config file {cfg_file} not found. Run 'python indexer.py' first.");
    sys.exit(1)
  cfg = json.loads(json.dumps(r), object_hook=lambda x: SimpleNamespace(**x));

  # ---- Load manifest file ----------------------
  man_file = cfg.manifest_file;
  try:
    with open(man_file, "r") as f: r = json.load(f);
  except FileNotFoundError:
    print(f"Error: Manifest file {man_file} not found. Run 'python indexer.py' first.");
    sys.exit(1)
  man = json.loads(json.dumps(r), object_hook=lambda x: SimpleNamespace(**x));

  # ---- Load filters file ------------------------
  fil_file = man.filters_file;
  try:
    with open(fil_file, "r") as f: r = json.load(f);
  except FileNotFoundError:
    print(f"Error: Filters file {fil_file} not found. Run 'python indexer.py' first.");
    sys.exit(1)
  filters = json.loads(json.dumps(r), object_hook=lambda x: SimpleNamespace(**x));

  # ---- Configure command line parser --------------
  parser = argparse.ArgumentParser(description="BioBank Search CLI")
  parser.add_argument("query", type=str, help="Natural language search query")
  parser.add_argument("-k", "--top_k", type=int, default=cfg.default_top_k, help="Top k to show");

  for f in filters: 
  #{
    c = f.column;
    arg = [f"--{c}"];
    kwargs = {
      "help": f"Option to filter results by {c}",
      "type": str,
      "required": False
    };
    if f.type == 'multi': kwargs["nargs"] = "+";
    print(f"about to add argument {arg} to the parser");
    parser.add_argument(*arg, **kwargs);
  #}

  # ---- Parse arguments and initialize ----------------
  args = parser.parse_args()
  filter_asks = {};
  for f in filters:
  #{
    c = f.column;
    vals = getattr(args, c, None)
    if not vals: continue;
    if f.type in ['mono', 'multi']:
      badv = [v for v in vals if v not in f.options];
      if badv:
        print(f"Error: Invalid value(s) {badv} for --{c}. Valid values are: {f.options}")
        sys.exit(1)
    filter_asks[c] = vals
  #}

  # ---- Execute search and display results ---------------------
  engine = BioBankGrep()
  results = engine.execute_query(args.query, filter_asks, args.top_k);
    
  if results.empty: print("\nNo matching biobanks found.")
  else:
    print(f"\n--- BioBankGrep Results ({len(results)}) ---")
    cols_to_show = ['name', 'repository_type', 'address', 'Ranking'];
    print(results[cols_to_show].to_csv(index=False))

if __name__ == "__main__":
  main()
