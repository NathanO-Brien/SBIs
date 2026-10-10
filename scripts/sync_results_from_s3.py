"""Pull runs from the flat s3://sbi-optimization-runs/runs/ prefix into a local flat folder.

    python scripts/sync_results_from_s3.py [--folder results] [--all-products]

Only files missing locally are downloaded.  Browse them with scripts/run_browser.py.
(The old nested country/ and prespecified/ layout was archived under archive-nonj2-20261009/.)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi_coverage.runs.index import PRODUCTS, S3_BUCKET, sync_from_s3  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="results")
ap.add_argument("--all-products", action="store_true", help="also fetch Gurobi logs")
a = ap.parse_args()
new = sync_from_s3(a.folder, products=PRODUCTS if a.all_products else ("result.json", "result.csv", "config.json"),
                   progress=lambda n: print("  +", n))
print(f"{len(new)} run(s) downloaded from s3://{S3_BUCKET}/runs/ into {a.folder}")
