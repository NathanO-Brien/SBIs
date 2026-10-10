# SBIs
Interrogating the technical details behind Golden Dome.

Optimizes space-based-interceptor (SBI) constellations: given a set of target points, an interceptor
performance model and a salvo size, find the fewest satellites whose coverage meets the requirement at
every target at every time. The optimizer follows Lee et al. (2020): repeat-ground-track (RGT) seed
orbits with circulant visibility matrices, solved as a binary integer program with Gurobi, using the exact
secular-J2 propagation of that paper. Orbit altitude is an **output**: seed families are anchored
top-down from the target latitudes (`src/sbi_coverage/Lee/seed_families.py`).

The last version before the J2 work (altitude-input, J2-off) is the `legacy-no-j2` branch / tag
`snapshot-no-j2-20261009`.

## Install
```
pip install -e .                 # numpy, scipy, numba, gurobipy (needs a licence), matplotlib, geopandas, ...
pip install -e ".[browser]"      # + Streamlit run browser
pip install -e ".[cloud]"        # + boto3 for AWS
```

## Layout
```
src/sbi_coverage/
  core/         configs, target shell, coverage kernels, propagation driver, plotting/report
  propagators/  secular-J2 propagator (Lee 2020, Eqs. 3-5)
  Lee/          seed families, seed pipeline, circulant matrix, MILP, run_pipeline.run_sclp (one full run)
  runs/         run names, run configuration, result index/S3 sync, replay helpers
  legacy/       frozen pre-J2 constellation generators (not used by the optimizer)
scripts/        entry points (below)
data/targets/   example target-point CSVs
```

## Running things
| Goal | Command |
|---|---|
| One local run from constants | `python scripts/lee_test2.py` |
| One cloud run | `python scripts/cloud_run.py --target-points-csv data/targets/pacific_blob1.csv --salvo 5 --test-run` |
| A parameter grid in the cloud | `python scripts/cloud_sweep.py --target-points-csv <csv> --scenario NK --grid grid.json` |
| The same grid locally (to test) | `python scripts/cloud_sweep.py ... --local --max-jobs 2 --time-limit 60` |
| **GUI**: launch many runs, watch jobs, pull/plot/export results (cloud only) | `streamlit run scripts/sbi_gui.py` |
| Pull cloud runs locally | `python scripts/sync_results_from_s3.py --folder results` |
| Re-simulate a result and check coverage | `python scripts/check_result_coverage.py <run name or S3 URI>` |
| Coverage plots / video, gap diagnosis | `scripts/coverage_from_sclp.py`, `scripts/diagnose_coverage_gaps.py` |
| Self-checks | `python scripts/verify_lee_j2.py`, `scripts/verify_runnames.py`, `scripts/verify_gui_logic.py` |

Launchers accept `--dry-run` (print identities only) and `--skip-existing` (skip combinations that already have
a finished result). Country mode (target a country's shell points) is deprecated; use prespecified target points.

## Run names and files
Every run has a short, decodable name and **all** its files live flat (no subfolders) in `results/` locally or
`s3://sbi-optimization-runs/runs/` in the cloud:

```
CNp-s046-d15-v6-g10-b170-k0-i1-261009a-3f9c_result.json     (also _result.csv, _gurobi.log, _config.json, _failed.json)
CN p  salvo detect vbo accel burnout decide ips  date seq hash
```
Scenario code (lowercase `p` = principal set), salvo size, detection time (s), interceptor burnout velocity
(km/s), max acceleration (g), target burnout time (s), decision time (s), interceptors per satellite, UTC date,
same-day sequence letter, and a 4-hex hash of every other result-affecting setting (target points, intercept
altitude, time step, N_D, margins, ...). `TEST-` prefix = test run (hidden in the browser by default). Solver
limits/gap are not part of a run's identity; they are recorded in the JSON. Decode with
`python -m sbi_coverage.runs.names decode <name>` (the module is stdlib-only and can be shared as one file).

## Result JSON (schema_version 2)
`result` (satellites, status, gap, timings), `cov` / `sim` (replay inputs), `targets`, `run_config` (inputs,
seed families with altitude/inclination, seed-generation and filter statistics, solver settings), and
`provenance` (git commit, library versions, host/Batch job). Older (schema 1) files are read through
`sbi_coverage.runs.index.normalize_sidecar`.

## Cloud
`docker build -f docker/Dockerfile --build-arg GIT_COMMIT=$(git rev-parse HEAD) -t <ecr-uri>:latest .`, push to ECR, and
the launchers submit AWS Batch jobs (queue `sbi-queue`, definition `sbi-optimizer-job`) that run
`scripts/batch_worker.py` with Gurobi WLS credentials from Secrets Manager. Constants: `sbi_coverage/runs/config.py`.

# Authorship Statement
This repository was authored by Nathan O'Brien as part of funded graduate research for the Engineering Space Policy Laboratory at the Georgia Institute of Technology. Professor Thomas González Roberts served as the sole advisor throughout the development process. Fernando Miranda Espana acted as an international affairs co-contributor.

# Contributors
Special thanks is due to Professor Hang Woon Lee for sharing his novel approach to solving the MILP constellation optimization problem. All materials under the src/sbi_coverage/Lee folder were directly inspired by his PhD dissertation and code repository.
