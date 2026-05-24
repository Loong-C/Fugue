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
    parser.add_argument("--variants", type=int, default=8)
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
        failures.append(f"{request.voices}-voice fugue has sub-half-beat notes")
    if diag.melody_issues != 0:
        failures.append(f"{request.voices}-voice fugue has weak voice-line melody metrics")
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


if __name__ == "__main__":
    raise SystemExit(main())
