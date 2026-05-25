# Implementation Notes

Updated: 2026-05-25.

## Implemented Engine

The current generator follows the hybrid route proposed in `feasibility_plan.md`:

- `SubjectAnalyzer`: `subject.py` loads MIDI, MusicXML, or compact text themes.
- `AnswerPlanner`: `theory.py` creates tonal answers by diatonic fifth mapping.
- `FormPlanner`: `generator.py` builds exposition, two middle-entry regions, final entry,
  stretto answer, and cadence.
- `Corpus Markov style model`: `style.py` learns interval/duration distributions,
  interval/duration transition tables, metrical duration profiles, and short
  melodic-rhythmic cells from `Jsb16thSeparated.json`, then augments them with all 48
  downloaded WTC fugue Humdrum files when available.
- `Counterpoint generator`: `generator.py` fills free spans by stochastic sampling, then
  penalizes range errors, strong-beat dissonance, voice crossing, and parallel perfects.
  Candidate fragments are ranked by both rule cost and style negative log-likelihood.
  Free counterpoint is generated in corpus-learned cells rather than isolated notes.
  Each cell is decoded with a small beam that balances the learned contour, metrical
  rhythm likelihood, vertical sonority, and parallel-perfect avoidance. Fragment
  boundaries are scored against neighboring fixed entries, and parallel-perfect motion is
  checked on the same half-beat grid used by the evaluator.
- `Evaluator`: `evaluate.py` reports entries, parallel fifths/octaves, crossings,
  range violations, strong dissonances, monophonic overlaps, rhythmic grid errors,
  sub-half-beat notes, melody issues, free-counterpoint stagnation, unstable
  free-counterpoint rhythm, vertical clusters, total duration, seed, and style source.
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
- Voice crossings: 7.
- Range violations: 0.
- Monophonic overlaps: 0.
- Rhythmic grid violations: 0.
- Sub-half-beat notes: 0.
- Melody issues: 0.
- Free-counterpoint stagnation issues: 0.
- Free-counterpoint rhythm issues: 0.
- Vertical clusters: 0.
- Strong dissonances: nonzero but limited; these include accented suspensions and rough spots
  from the stochastic free counterpoint layer.
- Candidate diversity from `scripts/verify_generator.py --variants 16`: 16 entry plans,
  16 score values, and 16 top-voice contours.
- Audibility observations from the verification report: maximum repeated free run is
  3 notes / 3 beats, minimum unique pitches per voice is 12, average active voices is
  3.585, and strong-beat dissonance rate is 0.059.

Three-voice smoke command:

```powershell
.\.venv\Scripts\fugue generate --key D --mode minor --voices 3 --subject examples\subjects\c_minor_subject.theme --out out\demo_d_minor_3voice.mid --report out\demo_d_minor_3voice_report.json --seed 17 --variants 16 --temperature 1.1
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
- Free-counterpoint stagnation issues: 0.
- Free-counterpoint rhythm issues: 0.
- Vertical clusters: 0.
- Candidate diversity from `scripts/verify_generator.py --variants 16`: 16 entry plans,
  15 score values, and 12 top-voice contours.
- Audibility observations from the verification report: maximum repeated free run is
  3 notes / 3 beats, minimum unique pitches per voice is 15, average active voices is
  2.781, and strong-beat dissonance rate is 0.039.

## Verification Script

Run:

```powershell
.\.venv\Scripts\python scripts\verify_generator.py --variants 16
```

The script generates 4-voice and 3-voice fugues, parses the MIDI outputs with `music21`,
and fails if the best candidate has parallel fifths, parallel octaves, range violations,
monophonic overlaps, rhythmic grid instability, sub-half-beat notes, weak voice-line
melody metrics, static free-counterpoint spans, unstable free-counterpoint rhythm,
vertical clusters, too few entries, or insufficient candidate diversity. It also writes
per-voice horizontal rhythm/melody observations and vertical texture observations:
duration histograms, maximum repeated free-note runs, pitch spans, active-voice density,
strong-beat dissonance rate, and low-spacing samples. It writes outputs to
`out/verification/`.

## Root Cause Fix for Static Free Counterpoint

The repeated-note problem was not caused by Bach data preferring repeated notes. The
learned corpus profile has nonzero melodic motion as its dominant local pattern; zero
intervals are a small minority. The original generator let the statistical interval
sample propose a pitch, then a local harmony/spacing selector often replaced that pitch
with the nearest safe chord tone. In dense four-voice passages this made repeated safe
tones a local optimum.

The fix is structural:

- learn short melodic-rhythmic cells from the corpus instead of sampling every note in
  isolation;
- decode each cell as a small beam so learned contour, metrical rhythm, and vertical
  harmony are optimized together;
- add a cell-level contour loss so the selected pitches preserve the cumulative shape of
  the learned cell;
- keep repeated free notes as separate attacks instead of merging them into hidden long
  tones;
- rank 16 candidates by the same diagnostics used for verification so clear, human-like
  candidates win over merely safe static ones.

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
