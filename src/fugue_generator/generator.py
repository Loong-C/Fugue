from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .evaluate import evaluate_voice_lines, with_output_path
from .models import EntryPlan, GeneratedFugue, HarmonySection, MusicalEvent, NoteSequence, VoiceLine
from .style import CorpusStyleModel
from .subject import load_subject
from .theory import (
    KeyContext,
    fit_sequence_to_voice,
    relative_degree_for_mode,
    tonal_answer,
    transpose_sequence_to_key,
    voice_specs,
)


@dataclass(frozen=True)
class FugueRequest:
    key: str
    mode: str
    voices: int
    subject_path: Path
    seed: int = 1
    temperature: float = 1.0
    variants: int = 5
    measures: int | None = None


class FugueGenerator:
    def __init__(self, project_root: Path, style_model: CorpusStyleModel | None = None) -> None:
        self.project_root = project_root
        self.style_model = style_model or CorpusStyleModel.load(project_root)

    def generate(self, request: FugueRequest) -> GeneratedFugue:
        candidates = self.generate_candidates(request)
        return max(candidates, key=lambda candidate: candidate.diagnostics.score)

    def generate_candidates(self, request: FugueRequest) -> list[GeneratedFugue]:
        return [
            self._generate_one(request, request.seed + index * 101)
            for index in range(max(1, request.variants))
        ]

    def write_report(
        self,
        fugue: GeneratedFugue,
        path: str | Path,
        candidates: list[GeneratedFugue] | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        data = fugue.diagnostics.to_dict()
        if candidates:
            data["candidates"] = [candidate.diagnostics.to_dict() for candidate in candidates]
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return target

    def _generate_one(self, request: FugueRequest, seed: int) -> GeneratedFugue:
        rng = random.Random(seed)
        key_context = KeyContext(request.key, request.mode)
        subject = load_subject(request.subject_path)
        subject = _normalize_subject_length(subject)
        specs = list(voice_specs(request.voices))
        entry_spacing = max(4.0, _ceil_to_measure(subject.duration * 0.9))
        total_duration = float((request.measures or _default_measures(subject.duration, request.voices)) * 4)

        lines = [VoiceLine(spec) for spec in specs]
        entries, harmony = self._plan_form(request, key_context, subject, entry_spacing, total_duration, rng)
        answer = tonal_answer(subject, key_context)
        countersubject = self._make_countersubject(subject, key_context, rng)

        for entry in entries:
            entry_key = _parse_key_name(entry.key_name)
            motif = subject if entry.kind == "subject" else answer
            if entry_key.name != key_context.name:
                motif = transpose_sequence_to_key(motif, key_context, entry_key)
            motif = fit_sequence_to_voice(motif.with_label(entry.label), specs[entry.voice_index])
            self._place_sequence(lines[entry.voice_index], motif, entry.start, entry.label)

        self._place_countersubjects(lines, entries, countersubject, key_context, specs)
        self._place_cadence(lines, key_context, total_duration)
        self._fill_all_gaps(lines, harmony, subject, request.temperature, rng, total_duration)
        _coalesce_adjacent_events(lines)

        diagnostics = evaluate_voice_lines(
            lines,
            entries,
            total_duration,
            seed,
            self.style_model.source,
        )
        return GeneratedFugue(lines, entries, harmony, diagnostics)

    def _plan_form(
        self,
        request: FugueRequest,
        key_context: KeyContext,
        subject: NoteSequence,
        entry_spacing: float,
        total_duration: float,
        rng: random.Random,
    ) -> tuple[list[EntryPlan], list[HarmonySection]]:
        count = request.voices
        if count == 4:
            possible_orders = ([1, 0, 2, 3], [0, 1, 3, 2], [2, 1, 0, 3])
        else:
            possible_orders = ([1, 0, 2], [0, 1, 2], [2, 1, 0])
        order = list(rng.choice(possible_orders))

        entries: list[EntryPlan] = []
        for index, voice_index in enumerate(order):
            kind = "subject" if index % 2 == 0 else "answer"
            entries.append(
                EntryPlan(
                    voice_index=voice_index,
                    start=index * entry_spacing,
                    kind=kind,
                    key_name=key_context.name,
                    label=f"exposition {kind}",
                )
            )

        exposition_end = len(order) * entry_spacing + subject.duration
        is_minorish = request.mode.lower() in {"minor", "natural_minor", "aeolian", "dorian", "phrygian"}
        dominant_key = key_context.related(5, "minor" if is_minorish else "major")
        relative_key = key_context.related(
            relative_degree_for_mode(request.mode),
            "major" if is_minorish else "minor",
        )
        subdominant_key = key_context.related(4, "minor" if is_minorish else "major")
        middle_keys = [dominant_key, relative_key, subdominant_key]
        rng.shuffle(middle_keys)

        cursor = _ceil_to_measure(exposition_end + 4.0)
        for idx, target_key in enumerate(middle_keys[:2]):
            cursor += 8.0
            voice_index = rng.randrange(count)
            kind = "subject" if idx % 2 == 0 else "answer"
            entries.append(
                EntryPlan(
                    voice_index=voice_index,
                    start=cursor,
                    kind=kind,
                    key_name=target_key.name,
                    label=f"middle {idx + 1} {kind}",
                )
            )
            cursor += _ceil_to_measure(subject.duration)

        final_start = min(total_duration - 16.0, _ceil_to_measure(cursor + 8.0))
        final_voice = rng.choice([0, count - 1])
        entries.append(
            EntryPlan(
                voice_index=final_voice,
                start=final_start,
                kind="subject",
                key_name=key_context.name,
                label="final subject",
            )
        )
        if subject.duration >= 3.0:
            stretto_voice = 0 if final_voice != 0 else min(1, count - 1)
            entries.append(
                EntryPlan(
                    voice_index=stretto_voice,
                    start=_round_to_grid(final_start + max(2.0, subject.duration * 0.5), 0.5),
                    kind="answer",
                    key_name=key_context.name,
                    label="stretto answer",
                )
            )

        harmony = [
            HarmonySection(
                0.0,
                _ceil_to_measure(exposition_end),
                key_context.name,
                (1, 5, 1, 5),
                "exposition",
            ),
            HarmonySection(
                _ceil_to_measure(exposition_end),
                final_start,
                dominant_key.name,
                (1, 4, 2, 5, 1),
                "episodes",
            ),
            HarmonySection(
                final_start,
                total_duration - 8.0,
                key_context.name,
                (1, 4, 5, 1),
                "final entries",
            ),
            HarmonySection(total_duration - 8.0, total_duration, key_context.name, (2, 5, 1, 1), "cadence"),
        ]
        return entries, harmony

    def _make_countersubject(
        self,
        subject: NoteSequence,
        key_context: KeyContext,
        rng: random.Random,
    ) -> NoteSequence:
        subject_pitches = subject.pitches
        if not subject_pitches:
            return subject
        start = key_context.snap_to_scale(subject_pitches[0] + rng.choice([3, 4, 8, 9]))
        events: list[MusicalEvent] = []
        previous_subject_pitch = subject_pitches[0]
        current = start
        for event in subject.events:
            if event.pitch is None:
                events.append(MusicalEvent(event.offset, event.duration, None, "countersubject"))
                continue
            motion = event.pitch - previous_subject_pitch
            if motion:
                current -= max(-5, min(5, motion))
            else:
                current += rng.choice([-2, 1, 2])
            current = key_context.snap_to_scale(current)
            events.append(MusicalEvent(event.offset, event.duration, current, "countersubject"))
            previous_subject_pitch = event.pitch
        return NoteSequence(events, "countersubject")

    def _place_countersubjects(
        self,
        lines: list[VoiceLine],
        entries: list[EntryPlan],
        countersubject: NoteSequence,
        key_context: KeyContext,
        specs,
    ) -> None:
        for entry in entries:
            start = entry.start + countersubject.duration
            if start >= max(e.start for e in entries) + countersubject.duration:
                continue
            voice = lines[entry.voice_index]
            sequence = fit_sequence_to_voice(
                countersubject.with_label("countersubject"),
                specs[entry.voice_index],
            )
            events = [
                MusicalEvent(
                    event.offset + start,
                    event.duration,
                    None
                    if event.pitch is None
                    else _fold_into_range(
                        event.pitch,
                        specs[entry.voice_index].low,
                        specs[entry.voice_index].high,
                    ),
                    "countersubject",
                )
                for event in sequence.events
            ]
            if all(not voice.overlaps(event) for event in events) and self._segment_penalty(
                lines,
                entry.voice_index,
                events,
            ) < 16:
                voice.add_many(events)

    def _place_cadence(self, lines: list[VoiceLine], key_context: KeyContext, total_duration: float) -> None:
        start = total_duration - 4.0
        if len(lines) == 3:
            final_degrees = [1, 5, 1]
            dominant_plan = ["leading", "4", "5"]
        else:
            final_degrees = [1, 3, 5, 1]
            dominant_plan = ["leading", "2", "4", "5"]
        for index, line in enumerate(lines):
            spec = line.spec
            target_degree = final_degrees[min(index, len(final_degrees) - 1)]
            final_pitch = _nearest_scale_degree_pitch(
                key_context,
                target_degree,
                spec.center,
                spec.low,
                spec.high,
            )
            plan = dominant_plan[min(index, len(dominant_plan) - 1)]
            if plan == "leading":
                dominant_pitch = final_pitch - 1
                if dominant_pitch < spec.low:
                    dominant_pitch += 12
            else:
                dominant_pitch = _nearest_scale_degree_pitch(
                    key_context,
                    int(plan),
                    final_pitch,
                    spec.low,
                    spec.high,
                )
            for event in [
                MusicalEvent(start - 4.0, 2.0, dominant_pitch, "cadential dominant"),
                MusicalEvent(start - 2.0, 2.0, dominant_pitch, "cadential dominant"),
                MusicalEvent(start, 4.0, final_pitch, "final tonic"),
            ]:
                if not line.overlaps(event):
                    line.add(event)

    def _fill_all_gaps(
        self,
        lines: list[VoiceLine],
        harmony: list[HarmonySection],
        subject: NoteSequence,
        temperature: float,
        rng: random.Random,
        total_duration: float,
    ) -> None:
        for voice_index, line in enumerate(lines):
            first_entry = min(
                (event.offset for event in line.events if event.pitch is not None),
                default=total_duration,
            )
            spans = line.free_spans(total_duration)
            for start, end in spans:
                if end <= first_entry + 1e-6:
                    continue
                if end - start < 0.5:
                    continue
                generated = self._generate_segment(
                    lines,
                    voice_index,
                    start,
                    end,
                    harmony,
                    subject,
                    temperature,
                    rng,
                )
                line.add_many(generated)

    def _generate_segment(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        start: float,
        end: float,
        harmony: list[HarmonySection],
        subject: NoteSequence,
        temperature: float,
        rng: random.Random,
    ) -> list[MusicalEvent]:
        best: list[MusicalEvent] = []
        best_score = math.inf
        attempts = 10
        line = lines[voice_index]
        start_previous = line.previous_pitch(start)
        end_next = line.next_pitch(end)
        for _ in range(attempts):
            candidate = self._sample_segment(
                lines,
                voice_index,
                start,
                end,
                harmony,
                subject,
                temperature,
                rng,
            )
            score = self._segment_penalty(lines, voice_index, candidate, start_previous, end_next)
            score += 0.35 * self._style_penalty(candidate)
            if score < best_score:
                best = candidate
                best_score = score
        return best

    def _sample_segment(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        start: float,
        end: float,
        harmony: list[HarmonySection],
        subject: NoteSequence,
        temperature: float,
        rng: random.Random,
    ) -> list[MusicalEvent]:
        line = lines[voice_index]
        spec = line.spec
        events: list[MusicalEvent] = []
        t = start
        previous = line.previous_pitch(t)
        if previous is None:
            previous = spec.center
        previous_interval: int | None = None
        previous_duration: float | None = None
        subject_intervals = _subject_intervals(subject)
        while t < end - 1e-6:
            remaining = end - t
            duration = _fit_duration_to_grid(
                self.style_model.sample_duration(
                    rng,
                    remaining,
                    temperature,
                    previous_duration=previous_duration,
                ),
                remaining,
            )
            section = _section_at(harmony, t)
            key_context = _parse_key_name(section.key_name)
            chord_degree = section.progression[int(((t - section.start) // 4) % len(section.progression))]
            chord = key_context.chord_pitches(
                chord_degree,
                spec.low,
                spec.high,
                include_seventh=chord_degree == 5,
            )
            strong = abs(t % 1.0) < 1e-6
            if strong and chord and rng.random() < 0.9:
                pitch = min(chord, key=lambda item: abs(item - previous))
            else:
                interval = (
                    rng.choice(subject_intervals)
                    if subject_intervals and rng.random() < 0.25
                    else self.style_model.sample_interval(
                        rng,
                        temperature,
                        previous_interval=previous_interval,
                    )
                )
                pitch = key_context.snap_to_scale(previous + int(interval))
                pitch = _fold_into_range(pitch, spec.low, spec.high)
            pitch = self._choose_local_pitch(
                lines,
                voice_index,
                t,
                duration,
                pitch,
                chord,
                key_context,
                previous,
                strong,
                rng,
            )
            events.append(MusicalEvent(t, duration, pitch, "free counterpoint"))
            previous_interval = int(pitch - previous)
            previous_duration = duration
            previous = pitch
            t = round(t + duration, 6)
        return events

    def _choose_local_pitch(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        t: float,
        duration: float,
        proposed: int,
        chord: list[int],
        key_context: KeyContext,
        previous: int,
        strong: bool,
        rng: random.Random,
    ) -> int:
        spec = lines[voice_index].spec
        candidates = {
            proposed,
            key_context.snap_to_scale(proposed + 1),
            key_context.snap_to_scale(proposed - 1),
        }
        candidates.update(_local_chord_candidates(chord, previous))
        for raw in range(previous - 9, previous + 10):
            snapped = key_context.snap_to_scale(raw)
            if spec.low <= snapped <= spec.high:
                candidates.add(snapped)
        candidates = {_fold_into_range(candidate, spec.low, spec.high) for candidate in candidates}
        scored = []
        strong_times = _covered_strong_times(t, duration)
        if not strong_times:
            strong_times = [t]
        crossing_times = _covered_grid_times(t, duration, 0.5)
        for pitch in candidates:
            penalty = _melodic_continuity_cost(previous, pitch)
            if strong and chord:
                penalty += 0 if pitch in chord else 2.5
            for other_index, other_line in enumerate(lines):
                if other_index == voice_index:
                    continue
                for check_t in strong_times:
                    other = other_line.active_pitch(check_t)
                    if other is None:
                        continue
                    interval = abs(pitch - other) % 12
                    if interval in {1, 2, 6, 10, 11}:
                        penalty += 35
                    if interval == 0:
                        penalty += 30
                    other_previous = other_line.active_pitch(max(0.0, check_t - 0.5))
                    if other_previous is not None:
                        previous_interval = abs(previous - other_previous) % 12
                        current_interval = abs(pitch - other) % 12
                        if previous_interval in {0, 7} and current_interval in {0, 7}:
                            if (pitch - previous) * (other - other_previous) > 0:
                                penalty += 1000
                for check_t in crossing_times:
                    active_by_voice = [line.active_pitch(check_t) for line in lines]
                    active_by_voice[voice_index] = pitch
                    penalty += _sonority_spacing_cost(active_by_voice, voice_index)
                    other = other_line.active_pitch(check_t)
                    other_previous = other_line.active_pitch(max(0.0, check_t - 0.5))
                    if other is not None and other_previous is not None:
                        self_previous = previous if check_t - 0.5 < t - 1e-6 else pitch
                        if _is_parallel_perfect_motion(self_previous, pitch, other_previous, other):
                            penalty += 1000
            penalty += rng.random() * 0.5
            scored.append((penalty, pitch))
        return min(scored)[1]

    def _segment_penalty(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        segment: list[MusicalEvent],
        start_previous: int | None = None,
        end_next: int | None = None,
    ) -> float:
        penalty = 0.0
        previous = start_previous
        for event in segment:
            if event.pitch is None:
                continue
            if previous is not None:
                penalty += _melodic_continuity_cost(previous, event.pitch)
            for other_index, line in enumerate(lines):
                if other_index == voice_index:
                    continue
                check_times = set(_covered_strong_times(event.offset, event.duration))
                check_times.update(_covered_grid_times(event.offset, event.duration, 0.5))
                for check_t in sorted(check_times):
                    other = line.active_pitch(check_t)
                    if other is None:
                        continue
                    interval = abs(event.pitch - other) % 12
                    if interval in {1, 2, 6, 10, 11}:
                        penalty += 7
                    if interval == 0:
                        penalty += 10
                    active_by_voice = [line.active_pitch(check_t) for line in lines]
                    active_by_voice[voice_index] = event.pitch
                    penalty += 0.35 * _sonority_spacing_cost(active_by_voice, voice_index)
                    if previous is not None:
                        other_previous = line.active_pitch(max(0.0, check_t - 0.5))
                        if other_previous is not None:
                            self_previous = previous if check_t - 0.5 < event.offset - 1e-6 else event.pitch
                            if _is_parallel_perfect_motion(self_previous, event.pitch, other_previous, other):
                                penalty += 200
            previous = event.pitch
        if previous is not None and end_next is not None:
            penalty += _melodic_continuity_cost(previous, end_next)
        return penalty

    def _style_penalty(self, segment: list[MusicalEvent]) -> float:
        pitches = [event.pitch for event in segment if event.pitch is not None]
        durations = [event.duration for event in segment if event.pitch is not None]
        if len(pitches) < 2:
            return 0.0
        return self.style_model.style_penalty(pitches, durations)

    def _place_sequence(self, line: VoiceLine, sequence: NoteSequence, start: float, label: str) -> None:
        for event in sequence.events:
            placed = event.shifted(start, label=label)
            if placed.pitch is not None:
                placed = MusicalEvent(
                    placed.offset,
                    placed.duration,
                    _fold_into_range(placed.pitch, line.spec.low, line.spec.high),
                    placed.label,
                )
                line.add(placed)


def _normalize_subject_length(subject: NoteSequence) -> NoteSequence:
    if subject.duration >= 2.0:
        return subject
    events = []
    for event in subject.events:
        events.append(MusicalEvent(event.offset * 2, event.duration * 2, event.pitch, event.label))
    return NoteSequence(events, subject.name)


def _coalesce_adjacent_events(lines: list[VoiceLine]) -> None:
    for line in lines:
        merged: list[MusicalEvent] = []
        for event in sorted(line.events, key=lambda item: (item.offset, item.duration)):
            if (
                merged
                and event.pitch == merged[-1].pitch
                and event.label == merged[-1].label
                and abs(event.offset - merged[-1].end) < 1e-6
            ):
                previous = merged[-1]
                merged[-1] = MusicalEvent(
                    previous.offset,
                    previous.duration + event.duration,
                    previous.pitch,
                    previous.label,
                )
            else:
                merged.append(event)
        line.events = merged


def _default_measures(subject_duration: float, voices: int) -> int:
    return max(28, min(56, int(math.ceil((subject_duration * (voices + 7) + 48) / 4))))


def _ceil_to_measure(value: float) -> float:
    return math.ceil(value / 4.0) * 4.0


def _fit_duration_to_grid(duration: float, remaining: float) -> float:
    if remaining < 0.5:
        return max(0.25, round(remaining * 4) / 4)
    value = max(0.5, round(duration * 2) / 2)
    return min(value, remaining)


def _round_to_grid(value: float, grid: float) -> float:
    return round(value / grid) * grid


def _parse_key_name(name: str) -> KeyContext:
    pieces = name.split()
    if len(pieces) == 1:
        return KeyContext(pieces[0], "minor")
    return KeyContext(pieces[0], pieces[1])


def _section_at(sections: list[HarmonySection], t: float) -> HarmonySection:
    for section in sections:
        if section.start <= t < section.end:
            return section
    return sections[-1]


def _subject_intervals(subject: NoteSequence) -> list[int]:
    pitches = subject.pitches
    return [b - a for a, b in zip(pitches, pitches[1:]) if abs(b - a) <= 7]


def _fold_into_range(pitch: int, low: int, high: int) -> int:
    while pitch < low:
        pitch += 12
    while pitch > high:
        pitch -= 12
    return max(low, min(high, pitch))


def _covered_strong_times(start: float, duration: float) -> list[float]:
    first = math.ceil(start - 1e-6)
    last = math.floor(start + duration - 1e-6)
    return [float(t) for t in range(first, last + 1) if t >= start - 1e-6]


def _covered_grid_times(start: float, duration: float, grid: float) -> list[float]:
    first = math.ceil((start - 1e-6) / grid)
    last = math.floor((start + duration - 1e-6) / grid)
    return [round(i * grid, 6) for i in range(first, last + 1)]


def _local_chord_candidates(chord: list[int], previous: int, span: int = 9) -> set[int]:
    local = {pitch for pitch in chord if abs(pitch - previous) <= span}
    if local or not chord:
        return local
    return {min(chord, key=lambda pitch: abs(pitch - previous))}


def _melodic_continuity_cost(previous: int, current: int) -> float:
    leap = abs(current - previous)
    if leap == 0:
        return 4.0
    cost = leap * 0.25
    if leap > 7:
        cost += (leap - 7) * 4.0
    if leap > 12:
        cost += (leap - 12) * 80.0
    return cost


def _is_parallel_perfect_motion(
    previous: int,
    current: int,
    other_previous: int,
    other_current: int,
) -> bool:
    previous_interval = abs(previous - other_previous) % 12
    current_interval = abs(current - other_current) % 12
    return (
        previous_interval in {0, 7}
        and current_interval in {0, 7}
        and (current - previous) * (other_current - other_previous) > 0
    )


def _sonority_spacing_cost(
    active_by_voice: list[int | None],
    voice_index: int,
) -> float:
    active = sorted((pitch for pitch in active_by_voice if pitch is not None), reverse=True)
    if len(active) < 2:
        return 0.0
    cost = 0.0
    for upper, lower in zip(active, active[1:]):
        gap = upper - lower
        if gap < 3:
            cost += (3 - gap) * 80
        elif gap > 19:
            cost += (gap - 19) * 4

    proposed_pitch = active_by_voice[voice_index]
    if proposed_pitch is not None:
        for other_index, other in enumerate(active_by_voice):
            if other_index == voice_index or other is None:
                continue
            if other_index < voice_index and proposed_pitch >= other - 1:
                cost += 90
            if other_index > voice_index and proposed_pitch <= other + 1:
                cost += 90
    return cost


def _windows(values: list[int], size: int):
    for index in range(0, len(values) - size + 1):
        yield values[index : index + size]


def _nearest_scale_degree_pitch(
    key_context: KeyContext,
    degree: int,
    center: int,
    low: int,
    high: int,
) -> int:
    candidates = []
    degree_index = degree - 1
    for octave in range(-4, 5):
        pitch = key_context.pitch_from_diatonic_index(degree_index + octave * 7)
        if low <= pitch <= high:
            candidates.append(pitch)
    if not candidates:
        return _fold_into_range(key_context.pitch_from_diatonic_index(degree_index), low, high)
    return min(candidates, key=lambda pitch: abs(pitch - center))
