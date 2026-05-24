# Implementation Notes

Updated: 2026-05-24.

## Implemented Engine

The current generator follows the hybrid route proposed in `feasibility_plan.md`:

- `SubjectAnalyzer`: `subject.py` loads MIDI, MusicXML, or compact text themes.
- `AnswerPlanner`: `theory.py` creates tonal answers by diatonic fifth mapping.
- `FormPlanner`: `generator.py` builds exposition, two middle-entry regions, final entry,
  stretto answer, and cadence.
- `Corpus Markov style model`: `style.py` learns interval/duration distributions and
  interval/duration transition tables from `Jsb16thSeparated.json`, then augments them
  with all 48 downloaded WTC fugue Humdrum files when available.
- `Counterpoint generator`: `generator.py` fills free spans by stochastic sampling, then
  penalizes range errors, strong-beat dissonance, voice crossing, and parallel perfects.
  Candidate fragments are ranked by both rule cost and style negative log-likelihood.
  Harmony candidates are kept local to the current melodic position, fragment boundaries
  are scored against neighboring fixed entries, and parallel-perfect motion is checked on
  the same half-beat grid used by the evaluator.
  Adjacent identical free-note fragments are merged before export so voice lines sustain
  naturally instead of stuttering.
- `Evaluator`: `evaluate.py` reports entries, parallel fifths/octaves, crossings,
  range violations, strong dissonances, monophonic overlaps, rhythmic grid errors,
  sub-half-beat notes, melody issues, vertical clusters, total duration, seed, and
  style source.
- `Exporter`: `export.py` writes MIDI and optional MusicXML through `music21`.

This is not a neural end-to-end model. It is a corpus-trained probabilistic symbolic
generator, which is better aligned with the small-data risk described in the feasibility
plan. The architecture still leaves room for a future neural inpainting/ranking module.

## Quality Gate Used During Development

Smoke command:

```powershell
.\.venv\Scripts\fugue generate --key C --mode minor --voices 4 --subject examples\subjects\c_minor_subject.theme --out out\demo_c_minor_fugue.mid --report out\demo_c_minor_report.json --seed 100 --variants 16 --temperature 0.8
```

Observed diagnostic from the current implementation:

- Entries: 8.
- Parallel fifths: 0.
- Parallel octaves: 0.
- Voice crossings: 0.
- Range violations: 0.
- Monophonic overlaps: 0.
- Rhythmic grid violations: 0.
- Sub-half-beat notes: 0.
- Melody issues: 0.
- Vertical clusters: 0.
- Strong dissonances: nonzero but limited; these include accented suspensions and rough spots
  from the stochastic free counterpoint layer.
- Candidate diversity from `scripts/verify_generator.py --variants 16`: 16 entry plans,
  16 score values, and 16 top-voice contours.

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
- Monophonic overlaps: 0.
- Rhythmic grid violations: 0.
- Sub-half-beat notes: 0.
- Melody issues: 0.
- Vertical clusters: 0.
- Candidate diversity from `scripts/verify_generator.py --variants 16`: 16 entry plans,
  14 score values, and 13 top-voice contours.

## Verification Script

Run:

```powershell
.\.venv\Scripts\python scripts\verify_generator.py --variants 16
```

The script generates 4-voice and 3-voice fugues, parses the MIDI outputs with `music21`,
and fails if the best candidate has parallel fifths, parallel octaves, range violations,
monophonic overlaps, rhythmic grid instability, sub-half-beat notes, weak voice-line
melody metrics, vertical clusters, too few entries, or insufficient candidate diversity.
It writes outputs to `out/verification/`.

## Known Limitations

- The style layer is Markov/statistical rather than neural. A future `M3` milestone can
  add a small inpainting/ranking network, but the present architecture already leaves a
  clean insertion point for that.
- The generator produces a short school fugue, not a Bach-level fugue.
- Some non-selected candidates can still have voice crossings or accented dissonances;
  the selected best candidate is checked by the verifier. Use higher `--variants` and
  lower `--temperature` for stricter results.
- The default text subject format is convenient for tests and examples. MIDI/MusicXML
  subjects are supported through `music21`.
