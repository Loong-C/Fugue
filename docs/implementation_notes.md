# Implementation Notes

Updated: 2026-05-26.

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
- `Harmony inference`: after fixed subjects, answers, countersubjects, and cadence tones
  are placed, `generator.py` infers a measure-level triadic harmonic target from the
  already-fixed voices. Free counterpoint is then generated and locally polished against
  that inferred target, rather than imposing a blind abstract progression over the
  contrapuntal material.
- `Evaluator`: `evaluate.py` reports entries, parallel fifths/octaves, crossings,
  range violations, strong dissonances, monophonic overlaps, rhythmic grid errors,
  sub-sixteenth notes, melody issues, rapid repeated free-counterpoint attacks,
  free-counterpoint stagnation, unstable free-counterpoint rhythm, vertical clusters,
  harmonic clarity issues, total duration, seed, and style source.
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
- Voice crossings: 16.
- Range violations: 0.
- Monophonic overlaps: 0.
- Rhythmic grid violations: 0.
- Sub-sixteenth notes: 0.
- Melody issues: 0.
- Rapid repeated free-counterpoint attacks: 0.
- Free-counterpoint stagnation issues: 0.
- Free-counterpoint rhythm issues: 0.
- Vertical clusters: 0.
- Strong dissonances: nonzero but limited; these include accented suspensions and rough spots
  from the stochastic free counterpoint layer.
- Candidate diversity from `scripts/verify_generator.py --variants 16`: 16 entry plans,
  16 score values, and 16 top-voice contours.
- Audibility observations from the verification report: free counterpoint now contains
  sixteenth notes, dotted or odd-quarter-grid durations, syncopated starts, and long
  notes. In the current default 16-candidate run the 4-voice winner has 83 free
  sixteenth notes, 3 dotted/odd-grid notes, 33 long notes, an aligned-start fraction
  of 0.811, maximum repeated free run of 2 notes / 4 beats, average active voices of
  3.585, and strong-beat dissonance rate of 0.061.
- Harmony observations from the verification report: 87.5% of sampled strong-beat
  sonorities contain at least two notes from the inferred target triad, 55.4% contain
  all three triad pitch classes, the non-chord-tone rate is 20.0%, and the bass supports
  the inferred root or fifth in 68.8% of samples.

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
- Sub-sixteenth notes: 0.
- Melody issues: 0.
- Rapid repeated free-counterpoint attacks: 0.
- Free-counterpoint stagnation issues: 0.
- Free-counterpoint rhythm issues: 0.
- Vertical clusters: 0.
- Candidate diversity from `scripts/verify_generator.py --variants 16`: 16 entry plans,
  14 score values, and 13 top-voice contours.
- Audibility observations from the verification report: the 3-voice winner has 70 free
  sixteenth notes, 10 dotted/odd-grid notes, 28 long notes, an aligned-start fraction
  of 0.659, maximum repeated free run of 2 notes / 4 beats, average active voices of
  2.781, and strong-beat dissonance rate of 0.036.
- Harmony observations from the verification report: 74.1% of sampled strong-beat
  sonorities contain at least two notes from the inferred target triad, 40.7% contain
  all three triad pitch classes, the non-chord-tone rate is 24.1%, and the bass supports
  the inferred root or fifth in 57.4% of samples.

## Verification Script

Run:

```powershell
.\.venv\Scripts\python scripts\verify_generator.py --variants 16
```

The script generates 4-voice and 3-voice fugues, parses the MIDI outputs with `music21`,
and fails if the best candidate has parallel fifths, parallel octaves, range violations,
monophonic overlaps, rhythmic grid instability, sub-sixteenth notes, weak voice-line
melody metrics, static free-counterpoint spans, unstable free-counterpoint rhythm,
vertical clusters, rapid repeated free-counterpoint attacks, too few entries, or
insufficient candidate diversity. It also writes per-voice horizontal rhythm/melody
observations, vertical texture observations, and inferred-harmony observations:
duration histograms, rhythm profiles for sixteenth/dotted/long/syncopated material,
maximum repeated free-note runs, pitch spans, active-voice density, aligned-start
fraction, strong-beat dissonance rate, low-spacing samples, target-triad coverage,
non-chord-tone rate, bass root/fifth support, and compact measure-level harmonic
progression. It writes outputs to `out/verification/`.

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
- distinguish sustained same-pitch material from rapid repeated attacks, so long tones
  are represented as sustains while short same-pitch rearticulation remains visible to
  evaluation;
- rank 16 candidates by the same diagnostics used for verification so clear, human-like
  candidates win over merely safe static ones.

## Root Cause Fix for Rhythm Collapse

The later hymn-like rhythm problem was caused by representation collapse rather than
missing weights. The first cell model fixed repeated-note stagnation by learning
melodic-rhythmic cells, but the extractor then discarded every duration below a half
beat and quantized all cell durations and phases to the half-beat grid. The decoder
reapplied the same half-beat quantization. As a result, the learned cell table contained
only plain eighth/quarter material plus a few long values; generated free counterpoint
could not produce sixteenth runs, dotted values, or independent off-beat starts even
though the raw corpora contained them.

The fix is structural:

- melodic cells are extracted and cached on the quarter-beat grid;
- each cell stores a learned rhythm class: `plain`, `sixteenth`, `dotted`,
  `syncopated`, or `long`;
- top-cell selection is stratified by rhythm class so rarer but real fugue gestures are
  not pushed out by globally frequent plain cells;
- rhythm-class priors are corrected by duration density and blended with the raw
  duration distribution, so sixteenth-note windows become local episodes rather than
  the default texture;
- cell decoding keeps quarter-grid durations instead of snapping them back to half
  beats;
- interval realization uses directional scale snapping, preserving upward and downward
  learned motion instead of letting chromatic intervals collapse back to the same scale
  tone;
- weak-grid vertical scoring allows passing and neighboring dissonances more than
  strong beats, while fixed-entry/cadence boundaries are checked for parallel perfects;
- adjacent same-pitch long free-counterpoint events are tied into sustained notes, so
  long tones are represented as long tones instead of repeated attacks.

## Root Cause Fix for Rapid Repetition After Sixteenth Notes

After sixteenth-note cells were restored, rapid same-pitch attacks reappeared in a few
outputs. A corpus probe showed that this was not a real learned Bach/WTC preference:
after contiguous same-pitch note runs are merged before learning, zero-interval
sixteenth cells remain a small minority. The failures came from two implementation
effects:

- the corpus extractor had previously treated split tied notes as separate attacks,
  so the style cache overcounted some zero intervals;
- the local vertical polisher only visited events that covered a strong beat. A weak
  sixteenth or off-beat eighth could have a very high repeated-attack cost but never be
  considered for repair.

The fix is again structural, not a hard "ban repeated notes" rule:

- style extraction now merges contiguous same-pitch runs before updating melodic
  interval and cell counters;
- cell decoding preserves the intended learned direction when a harmony repair snaps a
  pitch to the scale or a target chord;
- the vertical/melodic polisher also visits weak events that are part of a rapid
  same-pitch free-counterpoint pair, letting the existing learned melodic and harmonic
  costs choose a neighboring scale or chord tone;
- `evaluate.py` and `scripts/verify_generator.py` now make rapid repeated
  free-counterpoint attacks an explicit zero-tolerance quality gate.

## Root Cause Harmony Upgrade

The first harmonic-control experiment failed when it imposed the broad form plan
directly on already-fixed entries. Subjects, answers, and countersubjects often imply a
different local sonority than the broad plan's current degree. Forcing every free voice
toward that broad target made some verticals worse, especially when a dominant seventh
was used as the default target without preparation/resolution logic.

The current solution reads harmony from the fixed contrapuntal material first:

- each measure chooses a target degree by fitting candidate triads to the already-placed
  voices, while still paying a transition cost toward classical progressions such as
  tonic, predominant, dominant, tonic;
- target sonorities are triads for now, avoiding unprepared default sevenths;
- generated free cells see the inferred target as a soft preference, weighted more on
  strong beats than on passing weak subdivisions;
- a local vertical polisher improves strong-beat chord membership, bass root/fifth
  support, spacing, and rapid repeated attacks without changing fixed subject material.

The result is not yet a full Roman-numeral harmonic planner, but the selected default
outputs now show clearer strong-beat triadic support while preserving melodic
independence and zero parallel fifths/octaves.

The current audition set is in `out/audition_2026-05-26_harmony/`:

- `c_minor_4voice_harmony.mid`
- `d_minor_3voice_harmony.mid`

Both parse with the expected voice count and report zero parallel fifths/octaves, zero
rapid repeated free-counterpoint attacks, zero rhythmic grid violations, zero
sub-sixteenth notes, zero free-counterpoint stagnation, zero unstable free-counterpoint
rhythm issues, and zero vertical clusters.

## Known Limitations

- The style layer is Markov/statistical rather than neural. A future `M3` milestone can
  add a small inpainting/ranking network, but the present architecture already leaves a
  clean insertion point for that.
- The generator produces a short school fugue, not a Bach-level fugue.
- Some non-selected candidates can still have voice crossings or accented dissonances;
  the selected best candidate is checked by the verifier. Use higher `--variants` and
  lower `--temperature` for stricter results.
- The harmonic model is still a soft triadic target inferred per measure. It does not
  yet parse full Roman-numeral syntax with suspensions, secondary dominants, cadential
  six-four treatment, or prepared/resolved sevenths.
- The default text subject format is convenient for tests and examples. MIDI/MusicXML
  subjects are supported through `music21`.
