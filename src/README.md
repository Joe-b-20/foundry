# src/

All experiment code, deliberately flat: files import each other as siblings
(`expJ_selfdiscover` imports `expG_controller`; `gpu_exp1b_ca2d` imports
`gpu_exp1_novelty`; `gpu_avida_loop` imports `gpu_avida_oe`), so one directory keeps
every import working.

**Run from the repo root:** `bash run.sh src/<file>.py [args]` — `run.sh` activates the
conda env and executes from root, so relative artifact paths (`runs/...`) resolve.

The full map (what each file does, the import graph, configs that matter, repro
commands): [`consolidation/06_code_map.md`](../consolidation/06_code_map.md). Paths
there predate the `src/` move — prepend `src/` to any `.py` it names.

Rough groups (details in the code map):

| group | files |
|---|---|
| foundation | `core_data.py`, `expA_mealy.py`, `fsm_extract.py` |
| the recipe + breadth (sessions 1–7) | `expB…expU`, `expM_*`, `expN_*`, `expJ_*`, `expK/O/P/Q/R/S/T` |
| moonshot / walls + bridge (8–9) | `expV…expGG` |
| GPU campaigns (10+) | `gpu_*` |
| identity hunt (phase 2–3) | `gpu_pcf_hunt.py`, `pcf_quadmine.py` |
| recognition (phase 4) | `fn_telescope.py` |
| interpretability / audit | `interp_*`, `audit.py`, `analyze_phase2.py` (in `scripts/`) |
