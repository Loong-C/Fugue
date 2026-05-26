from __future__ import annotations

import argparse
import json
from pathlib import Path

from music21 import converter

from fugue_generator.export import write_midi
from fugue_generator.generator import FugueGenerator, FugueRequest
from fugue_generator.models import GeneratedFugue
from fugue_generator.theory import KeyContext


DEFAULT_SUBJECT = Path("examples/subjects/c_minor_subject.theme")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and verify fugue quality gates.")
    parser.add_argument("--subject", type=Path, default=DEFAULT_SUBJECT)
    parser.add_argument("--out-dir", type=Path, default=Path("out/verification"))
    parser.add_argument("--variants", type=int, default=16)
    args = parser.parse_args()

    root = Path.cwd()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    generator = FugueGenerator(root)
    runs = [
        ("four_voice_c_minor", FugueRequest("C", "minor", 4, args.subject, 100, 0.8, args.variants)),
        ("three_voice_d_minor", FugueRequest("D", "minor", 3, args.subject, 17, 1.1, args.variants)),
    ]

    results = []
    failures = []
    for name, request in runs:
        candidates = generator.generate_candidates(request)
        best = max(candidates, key=lambda candidate: candidate.diagnostics.score)
        output = write_midi(best, args.out_dir / f"{name}.mid", KeyContext(request.key, request.mode))
        parsed = converter.parse(output)
        diversity = _candidate_diversity(candidates)
        result = {
            "name": name,
            "output": str(output),
            "parsed_parts": len(parsed.parts),
            "parsed_duration": float(parsed.highestTime),
            "diversity": diversity,
            "audibility": _audibility_observations(best),
            "best": best.diagnostics.to_dict(),
            "candidate_scores": [candidate.diagnostics.score for candidate in candidates],
        }
        results.append(result)
        failures.extend(_quality_failures(request, best, len(parsed.parts), diversity))

    report = {"results": results, "failures": failures}
    report_path = args.out_dir / "verification_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 1 if failures else 0


def _quality_failures(
    request: FugueRequest,
    fugue: GeneratedFugue,
    parsed_parts: int,
    diversity: dict[str, int],
) -> list[str]:
    failures = []
    diag = fugue.diagnostics
    if parsed_parts != request.voices:
        failures.append(f"{request.voices}-voice MIDI parsed as {parsed_parts} parts")
    if len(fugue.entries) < request.voices + 3:
        failures.append(f"{request.voices}-voice fugue has too few entries: {len(fugue.entries)}")
    if diag.parallel_fifths != 0:
        failures.append(f"{request.voices}-voice fugue has parallel fifths: {diag.parallel_fifths}")
    if diag.parallel_octaves != 0:
        failures.append(f"{request.voices}-voice fugue has parallel octaves: {diag.parallel_octaves}")
    if diag.range_violations != 0:
        failures.append(f"{request.voices}-voice fugue has range violations: {diag.range_violations}")
    if diag.monophonic_overlaps != 0:
        failures.append(f"{request.voices}-voice fugue has overlapping notes in a voice")
    if diag.rhythmic_grid_violations != 0:
        failures.append(f"{request.voices}-voice fugue has unstable rhythmic grid positions")
    if diag.short_note_count != 0:
        failures.append(f"{request.voices}-voice fugue has sub-sixteenth-note values")
    if diag.melody_issues != 0:
        failures.append(f"{request.voices}-voice fugue has weak voice-line melody metrics")
    if diag.free_stagnation_issues != 0:
        failures.append(f"{request.voices}-voice fugue has static free-counterpoint spans")
    if diag.free_rhythm_issues != 0:
        failures.append(f"{request.voices}-voice fugue has unstable free-counterpoint rhythm")
    if diag.vertical_clusters != 0:
        failures.append(f"{request.voices}-voice fugue has vertical tone clusters")
    if request.voices == 3 and diag.voice_crossings != 0:
        failures.append(f"3-voice fugue has voice crossings: {diag.voice_crossings}")
    if diversity["entry_plans"] < min(3, request.variants):
        failures.append(f"{request.voices}-voice candidates lack entry-plan diversity")
    if diversity["score_values"] < min(3, request.variants):
        failures.append(f"{request.voices}-voice candidates lack score diversity")
    return failures


def _candidate_diversity(candidates: list[GeneratedFugue]) -> dict[str, int]:
    entry_signatures = {
        tuple((entry.voice_index, entry.kind, entry.key_name) for entry in candidate.entries)
        for candidate in candidates
    }
    score_values = {candidate.diagnostics.score for candidate in candidates}
    top_voice_contours = {_top_voice_contour(candidate) for candidate in candidates}
    return {
        "entry_plans": len(entry_signatures),
        "score_values": len(score_values),
        "top_voice_contours": len(top_voice_contours),
    }


def _top_voice_contour(candidate: GeneratedFugue) -> tuple[int, ...]:
    events = [
        event
        for event in sorted(candidate.voices[0].events, key=lambda item: item.offset)
        if event.pitch is not None
    ]
    return tuple(event.pitch for event in events[:16])


def _audibility_observations(fugue: GeneratedFugue) -> dict[str, object]:
    per_voice = []
    for index, voice in enumerate(fugue.voices):
        events = sorted([event for event in voice.events if event.pitch is not None], key=lambda item: item.offset)
        free = [event for event in events if event.label == "free counterpoint"]
        durations = _histogram(event.duration for event in free)
        max_run_notes, max_run_beats = _max_repeated_run(free)
        pitches = [event.pitch for event in events]
        per_voice.append(
            {
                "voice": index,
                "notes": len(events),
                "free_notes": len(free),
                "unique_pitches": len(set(pitches)),
                "pitch_span": 0 if not pitches else max(pitches) - min(pitches),
                "free_duration_histogram": durations,
                "free_rhythm_profile": _rhythm_profile(free),
                "max_free_repeated_run_notes": max_run_notes,
                "max_free_repeated_run_beats": round(max_run_beats, 3),
            }
        )

    vertical_samples = _vertical_samples(fugue)
    rhythm_independence = _rhythm_independence(fugue)
    return {
        "per_voice": per_voice,
        "vertical": vertical_samples,
        "rhythm_independence": rhythm_independence,
        "summary": {
            "max_free_repeated_run_notes": max(item["max_free_repeated_run_notes"] for item in per_voice),
            "max_free_repeated_run_beats": max(item["max_free_repeated_run_beats"] for item in per_voice),
            "minimum_unique_pitches": min(item["unique_pitches"] for item in per_voice),
            "average_active_voices": vertical_samples["average_active_voices"],
            "strong_dissonance_rate": vertical_samples["strong_dissonance_rate"],
            "vertical_cluster_times": fugue.diagnostics.vertical_clusters,
            "free_sixteenth_notes": sum(item["free_rhythm_profile"]["sixteenth_notes"] for item in per_voice),
            "free_dotted_notes": sum(item["free_rhythm_profile"]["dotted_notes"] for item in per_voice),
            "free_long_notes": sum(item["free_rhythm_profile"]["long_notes"] for item in per_voice),
            "aligned_start_fraction": rhythm_independence["aligned_start_fraction"],
        },
    }


def _vertical_samples(fugue: GeneratedFugue) -> dict[str, object]:
    total_duration = fugue.diagnostics.total_duration
    times = [round(i * 0.5, 6) for i in range(int(total_duration / 0.5) + 1)]
    active_counts = []
    strong_checks = 0
    strong_dissonances = 0
    close_low_spacing = 0
    full_texture_after_exposition = 0
    after_exposition_checks = 0
    exposition_end = max((entry.start for entry in fugue.entries[: len(fugue.voices)]), default=0.0)
    for t in times:
        pitches = [voice.active_pitch(t) for voice in fugue.voices]
        active = [pitch for pitch in pitches if pitch is not None]
        active_counts.append(len(active))
        if t >= exposition_end:
            after_exposition_checks += 1
            if len(active) >= len(fugue.voices) - 1:
                full_texture_after_exposition += 1
        if abs(t % 1.0) < 1e-6:
            for i in range(len(pitches)):
                for j in range(i + 1, len(pitches)):
                    if pitches[i] is None or pitches[j] is None:
                        continue
                    strong_checks += 1
                    if abs(pitches[i] - pitches[j]) % 12 in {1, 2, 6, 10, 11}:
                        strong_dissonances += 1
        ordered = sorted(active)
        if len(ordered) >= 2 and ordered[1] - ordered[0] < 5:
            close_low_spacing += 1
    return {
        "average_active_voices": round(sum(active_counts) / max(1, len(active_counts)), 3),
        "minimum_active_voices": min(active_counts) if active_counts else 0,
        "full_texture_ratio_after_exposition": round(
            full_texture_after_exposition / max(1, after_exposition_checks),
            3,
        ),
        "strong_dissonance_rate": round(strong_dissonances / max(1, strong_checks), 3),
        "close_low_spacing_samples": close_low_spacing,
    }


def _max_repeated_run(events) -> tuple[int, float]:
    run_pitch = None
    run_count = 0
    run_duration = 0.0
    best_count = 0
    best_duration = 0.0
    for event in events:
        if event.pitch == run_pitch:
            run_count += 1
            run_duration += event.duration
        else:
            best_count = max(best_count, run_count)
            best_duration = max(best_duration, run_duration)
            run_pitch = event.pitch
            run_count = 1
            run_duration = event.duration
    best_count = max(best_count, run_count)
    best_duration = max(best_duration, run_duration)
    return best_count, best_duration


def _rhythm_profile(events) -> dict[str, int]:
    sixteenth_notes = 0
    dotted_notes = 0
    long_notes = 0
    syncopated_onsets = 0
    beat_crossing_syncopations = 0
    for event in events:
        units = round(event.duration / 0.25)
        if event.duration < 0.5 - 1e-6:
            sixteenth_notes += 1
        if units % 2 == 1 and units > 1:
            dotted_notes += 1
        if event.duration >= 2.0 - 1e-6:
            long_notes += 1
        beat_phase = round(event.offset % 1.0, 6)
        if beat_phase in {0.25, 0.75}:
            syncopated_onsets += 1
        if abs(beat_phase - 0.5) < 1e-6 and int(event.offset) < int(event.end - 1e-6):
            beat_crossing_syncopations += 1
    return {
        "sixteenth_notes": sixteenth_notes,
        "dotted_notes": dotted_notes,
        "long_notes": long_notes,
        "syncopated_onsets": syncopated_onsets,
        "beat_crossing_syncopations": beat_crossing_syncopations,
    }


def _rhythm_independence(fugue: GeneratedFugue) -> dict[str, float]:
    starts_by_time: dict[float, int] = {}
    starts = 0
    for voice in fugue.voices:
        for event in voice.events:
            if event.pitch is None or event.label != "free counterpoint":
                continue
            starts += 1
            key = round(event.offset, 6)
            starts_by_time[key] = starts_by_time.get(key, 0) + 1
    aligned_starts = sum(count for count in starts_by_time.values() if count >= 2)
    return {
        "free_note_starts": starts,
        "shared_start_times": sum(1 for count in starts_by_time.values() if count >= 2),
        "aligned_start_fraction": round(aligned_starts / max(1, starts), 3),
    }


def _histogram(values) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(round(float(value), 3))
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: float(item[0])))


if __name__ == "__main__":
    raise SystemExit(main())
