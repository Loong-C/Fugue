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
    monophonic_overlaps = 0
    rhythmic_grid_violations = 0
    short_note_count = 0
    melody_issues = 0
    free_stagnation_issues = 0
    free_rhythm_issues = 0
    vertical_clusters = 0

    for voice in voices:
        sorted_events = sorted(voice.events, key=lambda item: item.offset)
        note_events = [event for event in sorted_events if event.pitch is not None]
        for event in voice.events:
            if event.pitch is not None and not (voice.spec.low <= event.pitch <= voice.spec.high):
                range_violations += 1
            if not _is_on_grid(event.offset, 0.25) or not _is_on_grid(event.duration, 0.25):
                rhythmic_grid_violations += 1
            if event.pitch is not None and event.duration < 0.5 - 1e-6:
                short_note_count += 1
        for first, second in zip(sorted_events, sorted_events[1:]):
            if first.end > second.offset + 1e-6:
                monophonic_overlaps += 1
        melody_issues += _voice_melody_issues(note_events)
        stagnation, rhythm = _free_counterpoint_issues(note_events)
        free_stagnation_issues += stagnation
        free_rhythm_issues += rhythm

    cluster_times = [round(i * 0.25, 6) for i in range(int(total_duration / 0.25) + 1)]
    for t in cluster_times:
        pitches = [voice.active_pitch(t) for voice in voices]
        active = sorted(pitch for pitch in pitches if pitch is not None)
        if _has_vertical_cluster(active):
            vertical_clusters += 1

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
        - 420.0 * parallel_fifths
        - 480.0 * parallel_octaves
        - 16.0 * voice_crossings
        - 100.0 * range_violations
        - 5.0 * strong_dissonances
        - 200.0 * monophonic_overlaps
        - 60.0 * rhythmic_grid_violations
        - 25.0 * short_note_count
        - 35.0 * melody_issues
        - 45.0 * free_stagnation_issues
        - 30.0 * free_rhythm_issues
        - 80.0 * vertical_clusters
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
        monophonic_overlaps=monophonic_overlaps,
        rhythmic_grid_violations=rhythmic_grid_violations,
        short_note_count=short_note_count,
        melody_issues=melody_issues,
        free_stagnation_issues=free_stagnation_issues,
        free_rhythm_issues=free_rhythm_issues,
        vertical_clusters=vertical_clusters,
        total_duration=total_duration,
        seed=seed,
        style_source=style_source,
    )


def with_output_path(diagnostics: FugueDiagnostics, output_path) -> FugueDiagnostics:
    return replace(diagnostics, output_path=output_path)


def _is_on_grid(value: float, grid: float) -> bool:
    return abs((value / grid) - round(value / grid)) < 1e-6


def _voice_melody_issues(events) -> int:
    if len(events) < 8:
        return 1
    pitches = [event.pitch for event in events if event.pitch is not None]
    issues = 0
    if len(set(pitches)) < 4:
        issues += 1
    repeated_run = 1
    for previous, current in zip(pitches, pitches[1:]):
        if current == previous:
            repeated_run += 1
            if repeated_run >= 5:
                issues += 1
        else:
            repeated_run = 1
        if abs(current - previous) > 12:
            issues += 1
    return issues


def _free_counterpoint_issues(events) -> tuple[int, int]:
    free_events = [event for event in events if event.label == "free counterpoint"]
    if not free_events:
        return 0, 0
    stagnation = 0
    rhythm = 0
    run_pitch = None
    run_duration = 0.0
    run_count = 0
    stable_long_values = {3.0, 4.0}
    for event in free_events:
        if event.pitch == run_pitch:
            run_count += 1
            run_duration += event.duration
        else:
            if run_count >= 4 or run_duration >= 4.0 - 1e-6:
                stagnation += 1
            run_pitch = event.pitch
            run_count = 1
            run_duration = event.duration

        if event.duration > 4.0 + 1e-6:
            stagnation += 1
        if event.duration > 2.0 + 1e-6 and round(event.duration, 6) not in stable_long_values:
            rhythm += 1
        if event.duration > 2.0 + 1e-6 and not _is_on_grid(event.offset, 1.0):
            rhythm += 1
    if run_count >= 4 or run_duration >= 4.0 - 1e-6:
        stagnation += 1
    return stagnation, rhythm


def _has_vertical_cluster(pitches: list[int]) -> bool:
    if len(pitches) < 3:
        return False
    return any(window[-1] - window[0] <= 4 for window in _windows(pitches, 3))


def _windows(values: list[int], size: int):
    for index in range(0, len(values) - size + 1):
        yield values[index : index + size]
