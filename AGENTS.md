# Agent Instructions

## Virtual Environment Setup
- Always use the virtual environment when working on this project
- If the virtual environment is not present, create it by running: `./setup_bottlemod_env.sh`
- Note: Running `setup_bottlemod_env.sh` recreates `.venv`; only rerun when a rebuild is intended.

## Scope Guardrails
- Prefer editing core code in `bottlemod/`.
- Treat `bottlemod_vanilla/` as a reference/baseline unless explicitly requested.

## Dependency/Install Fallback
- If `requirements.txt` is missing, document the install issue and proceed with minimal required packages only.

## Reproducibility and Output Hygiene
- Do not modify historical experiment outputs under `thesis_experiment/`; write new results to a new timestamped folder.
- Do not edit generated artifacts such as `__pycache__/`.

## Logging Work
- After completing work, always log what you have done in `logbook.md`
- Include a timestamp and summary in bulletpoints.
- Each entry must include: timestamp, files changed, tests run, and a short bullet summary.

