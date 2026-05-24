# Fugue

Research scaffold for an AI-assisted fugue generator.

Target task: given a key/mode, a subject, and either three or four voices,
generate a short tonal fugue and export it as MIDI.

Current state:

- Research notes and feasibility plan live in `docs/`.
- Resource download helpers live in `scripts/`.
- Raw downloaded data should be kept under `data/raw/` and is ignored by git.
- The local fallback environment is `.venv-system/`, created with system site-packages because
  direct PyPI access from the sandbox is blocked.

Planned CLI shape:

```powershell
fugue generate --key C --mode minor --voices 4 --subject examples/subject.mid --out out/fugue.mid
```

Local setup used in this workspace:

```powershell
python -m venv --system-site-packages .venv-system
.\.venv-system\Scripts\python -m pip install -e . --no-deps --no-build-isolation
python scripts\fetch_resources.py
```

