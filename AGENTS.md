# AGENTS.md

## Mandatory Startup (Every Session)
Before doing analysis, coding, or running experiments, read these files in order:
1. `docs/HANDOFF_2026-02-26.md` (or newest `docs/HANDOFF_*.md` if a newer one exists)
2. `PROJECT_LOG.md`
3. `docs/temporal_mosaic_v1.md` (for temporal mosaic tasks)
4. `README.md` (commands and run patterns)

Then provide a short 3-6 bullet summary of:
- Current pipeline status
- Active defaults/parameters
- Latest outputs/artifacts
- Immediate next steps for the user request

## Session-End Update Rule
If code, defaults, workflow, or outputs changed, update:
- `PROJECT_LOG.md` (decisions + session note)
- `docs/HANDOFF_2026-02-26.md` (or latest handoff file)
- Any impacted technical doc (for example `docs/temporal_mosaic_v1.md`)

## Scope
- Keep exports targeting Google Drive with explicit folder names.
- Prefer small AOI tests before large runs when changing masking or assignment logic.
- For diagnostics changes, regenerate at least one single-site plot before batch runs.
