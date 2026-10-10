"""Run bookkeeping: short decodable run names (names.py) and the index/S3 sync helpers (index.py).

Deliberately kept out of core/ and Lee/: nothing in the optimizer physics imports this package;
only the launch/worker/analysis scripts do. ``names`` is stdlib-only so it can be shared as a
single file; ``index`` needs pandas/boto3 (imported lazily).
"""
