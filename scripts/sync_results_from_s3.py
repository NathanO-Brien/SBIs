"""Pull runs from the flat s3://sbi-optimization-runs/runs/ prefix into a local flat folder.

    python scripts/sync_results_from_s3.py [--folder results] [--scenario NK] [--all-products] [--overwrite]

Only files missing locally are downloaded (unless --overwrite). Browse them with scripts/sbi_gui.py.
(The old nested country/ and prespecified/ layout was archived under archive-nonj2-20261009/.)
"""
import argparse

from sbi_coverage.runs.config import S3_BUCKET
from sbi_coverage.runs.index import PRODUCTS, DEFAULT_SYNC_PRODUCTS, list_s3_runs, sync_from_s3
from sbi_coverage.runs.names import decode_run_name

ap = argparse.ArgumentParser()
ap.add_argument("--folder", default="results")
ap.add_argument("--scenario", help="only runs of this scenario code, e.g. NK")
ap.add_argument("--all-products", action="store_true", help="also fetch Gurobi logs")
ap.add_argument("--overwrite", action="store_true", help="re-download files that already exist locally")
a = ap.parse_args()
names = [n for n in list_s3_runs() if not a.scenario or decode_run_name(n)["scenario"] == a.scenario.upper()]
new = sync_from_s3(a.folder, names=names, products=PRODUCTS if a.all_products else DEFAULT_SYNC_PRODUCTS,
                   overwrite=a.overwrite, progress=lambda n: print("  +", n))
print(f"{len(new)} run(s) updated from s3://{S3_BUCKET}/runs/ into {a.folder}")
