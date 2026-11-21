# Vendored: sierra-research/tau2-bench @ c06b07dd7c4d4ec61ba6ff9bcdf0b15dd7ef36d4 (MIT, see LICENSE)
- **Contents:** `src/tau2/**/*.py` (from the commit tarball) and `data/tau2/domains/airline/{db.json,tasks.json,policy.md,split_tasks.json}`.
- **Not modified.**
- **Loaded without running `tau2/__init__.py`.** That file imports the LLM runner stack, which the airline domain does not need; see `scripts/tau2_airline_feasibility.py`.
- **Runtime deps for the airline domain only:** pydantic ≥ 2.12, loguru, docstring-parser, deepdiff, addict, python-dotenv, toml.
