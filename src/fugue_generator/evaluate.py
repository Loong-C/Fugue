from __future__ import annotations

from dataclasses import replace

from .models import EntryPlan, FugueDiagnostics, VoiceLine
from .theory import interval_class


def evaluate_voice_lines(
    voices: list[VoiceLine],
    entries: list[EntryPlan],
    total_duration: float,
    seed: int,
    style_source: str,
    grid: float = 0.5,
) -> FugueDiagnostics:
    parallel_fifths = 0
    parallel_octaves = 0
    voice_crossings = 0
    range_violations = 0
    strong_dissonances = 0

    for voice in voices:
        for event in voice.events:
            if event.pitch is not None and not (voice.spec.low <= event.pitch <= voice.spec.high):
                range_violations += 1

    times = [round(i * grid, 6) for i in range(int(total_duration / grid) + 1)]
    for idx in range(1, len(times)):
        prev_t = times[idx - 1]
        t = times[idx]
        prev = [voice.active_pitch(prev_t) for voice in voices]
        current = [voice.active_pitch(t) for voice in voices]

        for upper, lower in zip(current, current[1:]):
            if upper is not None and lower is not None and upper < lower:
                voice_crossings += 1

        strong = abs(t % 1.0) < 1e-6
        for i in range(len(voices)):
            for j in range(i + 1, len(voices)):
                a0, b0 = prev[i], prev[j]
                a1, b1 = current[i], current[j]
                if None in {a0, b0, a1, b1}:
                    continue
                previous_interval = interval_class(a0, b0)
                current_interval = interval_class(a1, b1)
                motion_a = a1 - a0
                motion_b = b1 - b0
                if current_interval == 7 and previous_interval == 7 and motion_a * motion_b > 0:
                    parallel_fifths += 1
                if current_interval == 0 and previous_interval == 0 and motion_a * motion_b > 0:
                    parallel_octaves += 1
                if strong and current_interval in {1, 2, 6, 10, 11}:
                    strong_dissonances += 1

    score = (
        1000.0
        - 55.0 * parallel_fifths
        - 65.0 * parallel_octaves
        - 8.0 * voice_crossings
        - 100.0 * range_violations
        - 5.0 * strong_dissonances
        + 20.0 * len(entries)
    )
    return FugueDiagnostics(
        score=round(score, 3),
        entries=entries,
        parallel_fifths=parallel_fifths,
        parallel_octaves=parallel_octaves,
        voice_crossings=voice_crossings,
        range_violations=range_violations,
        strong_dissonances=strong_dissonances,
        total_duration=total_duration,
        seed=seed,
        style_source=style_source,
    )


def with_output_path(diagnostics: FugueDiagnostics, output_path) -> FugueDiagnostics:
    return replace(diagnostics, output_path=output_path)

