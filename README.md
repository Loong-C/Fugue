# Fugue

Research scaffold for an AI-assisted fugue generator.

Target task: given a key/mode, a subject, and either three or four voices,
generate a short tonal fugue and export it as MIDI.

Current state:

- Research notes and feasibility plan live in `docs/`.
- Resource download helpers live in `scripts/`.
- Raw downloaded data should be kept under `data/raw/` and is ignored by git.
- The working environment is `.venv/`, configured with system site-packages so it can see
  both the locally added solver/tokenization packages and the system music/ML packages.
- The generator is implemented as a hybrid symbolic/statistical engine: form planning and
  entry placement are explicit, while free counterpoint and episodes are sampled from a
  corpus-trained Markov style profile and then rule-scored. The profile is trained from
  JSB chorales plus the downloaded WTC fugue voice data when available.

Planned CLI shape:

```powershell
.\.venv\Scripts\fugue generate `
  --key C `
  --mode minor `
  --voices 4 `
  --subject examples\subjects\c_minor_subject.theme `
  --out out\fugue.mid `
  --report out\fugue_report.json `
  --variants 8 `
  --temperature 0.85
```

Local setup used in this workspace:

```powershell
python -m venv .venv
# Set include-system-site-packages = true in .venv\pyvenv.cfg if PyPI is unavailable.
.\.venv\Scripts\python -m pip install -e . --no-deps --no-build-isolation
python scripts\fetch_resources.py
```

The subject can be a MIDI/MusicXML file or a compact text theme such as:

```text
C4:0.5 D4:0.5 Eb4:0.5 G4:0.5 F4:0.5 Eb4:0.5 D4:0.5 C4:1.0
```

Useful commands:

```powershell
.\.venv\Scripts\fugue inspect-subject examples\subjects\c_minor_subject.theme
.\.venv\Scripts\fugue build-style-profile
.\.venv\Scripts\python scripts\verify_generator.py --variants 8
.\.venv\Scripts\pytest
```

The verifier writes MIDI examples and a JSON quality report under `out/verification/`.
It checks parseability, entry count, zero parallel fifths/octaves in the selected best
candidates, range safety, and candidate diversity.
