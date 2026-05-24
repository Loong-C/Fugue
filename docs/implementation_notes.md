# Implementation Notes

Updated: 2026-05-24.

## Implemented Engine

The current generator follows the hybrid route proposed in `feasibility_plan.md`:

- `SubjectAnalyzer`: `subject.py` loads MIDI, MusicXML, or compact text themes.
- `AnswerPlanner`: `theory.py` creates tonal answers by diatonic fifth mapping.
- `FormPlanner`: `generator.py` builds exposition, two middle-entry regions, final entry,
  stretto answer, and cadence.
- `Statistical style model`: `style.py` learns interval and duration weights from
  `Jsb16thSeparated.json` when the dataset is available, with a built-in fallback.
- `Counterpoint generator`: `generator.py` fills free spans by stochastic sampling, then
  penalizes range errors, strong-beat dissonance, voice crossing, and parallel perfects.
- `Evaluator`: `evaluate.py` reports entries, parallel fifths/octaves, crossings,
  range violations, strong dissonances, total duration, seed, and style source.
- `Exporter`: `export.py` writes MIDI and optional MusicXML through `music21`.

This is not a neural end-to-end model. It is a corpus-trained symbolic generator, which is
better aligned with the small-data risk described in the feasibility plan.

## Quality Gate Used During Development

Smoke command:

```powershell
.\.venv\Scripts\fugue generate --key C --mode minor --voices 4 --subject examples\subjects\c_minor_subject.theme --out out\demo_c_minor_fugue.mid --report out\demo_c_minor_report.json --seed 100 --variants 16 --temperature 0.8
```

Observed diagnostic from the current implementation:

- Entries: 8.
- Parallel fifths: 0.
- Parallel octaves: 0.
- Voice crossings: 5.
- Range violations: 0.
- Strong dissonances: nonzero but limited; these include accented suspensions and rough spots
  from the stochastic free counterpoint layer.

Three-voice smoke command:

```powershell
.\.venv\Scripts\fugue generate --key D --mode minor --voices 3 --subject examples\subjects\c_minor_subject.theme --out out\demo_d_minor_3voice.mid --report out\demo_d_minor_3voice_report.json --seed 17 --variants 8 --temperature 1.1
```

Observed diagnostic:

- Entries: 7.
- Parallel fifths: 0.
- Parallel octaves: 0.
- Voice crossings: 0.
- Range violations: 0.

## Known Limitations

- The style layer is statistical rather than neural. A future `M3` milestone can add a
  small inpainting/ranking network, but the present architecture already leaves a clean
  insertion point for that.
- The generator produces a short school fugue, not a Bach-level fugue.
- Some generated candidates can still have voice crossings or accented dissonances; use
  higher `--variants` and lower `--temperature` for stricter results.
- The default text subject format is convenient for tests and examples. MIDI/MusicXML
  subjects are supported through `music21`.
