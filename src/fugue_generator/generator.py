from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

from .evaluate import evaluate_voice_lines, with_output_path
from .models import EntryPlan, GeneratedFugue, HarmonySection, MusicalEvent, NoteSequence, VoiceLine, VoiceSpec
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
    variants: int = 16
    measures: int | None = None


def selection_key(candidate: GeneratedFugue) -> tuple[int, float]:
    diagnostics = candidate.diagnostics
    hard_issues = (
        diagnostics.parallel_fifths
        + diagnostics.parallel_octaves
        + diagnostics.range_violations
        + diagnostics.monophonic_overlaps
        + diagnostics.rhythmic_grid_violations
        + diagnostics.short_note_count
        + diagnostics.melody_issues
        + diagnostics.repeated_attack_issues
        + diagnostics.free_stagnation_issues
        + diagnostics.free_rhythm_issues
        + diagnostics.vertical_clusters
    )
    if len(candidate.voices) == 3:
        hard_issues += diagnostics.voice_crossings
    return (-hard_issues, diagnostics.score)


class FugueGenerator:
    def __init__(self, project_root: Path, style_model: CorpusStyleModel | None = None) -> None:
        self.project_root = project_root
        self.style_model = style_model or CorpusStyleModel.load(project_root)

    def generate(self, request: FugueRequest) -> GeneratedFugue:
        candidates = self.generate_candidates(request)
        return max(candidates, key=selection_key)

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
        harmony = self._infer_harmony_from_fixed_lines(lines, harmony, total_duration)
        self._fill_all_gaps(lines, harmony, subject, request.temperature, rng, total_duration)
        _merge_sustained_free_repetitions(lines)
        self._polish_vertical_sonorities(lines, harmony)
        _merge_sustained_free_repetitions(lines)

        diagnostics = evaluate_voice_lines(
            lines,
            entries,
            total_duration,
            seed,
            self.style_model.source,
            harmony=harmony,
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

    def _infer_harmony_from_fixed_lines(
        self,
        lines: list[VoiceLine],
        base_harmony: list[HarmonySection],
        total_duration: float,
    ) -> list[HarmonySection]:
        inferred: list[HarmonySection] = []
        previous_degree: int | None = None
        measure_count = int(math.ceil(total_duration / 4.0))
        for measure in range(measure_count):
            start = float(measure * 4)
            end = min(total_duration, start + 4.0)
            base_section = _section_at(base_harmony, start)
            key_context = _parse_key_name(base_section.key_name)
            broad_degree = base_section.progression[
                int(((start - base_section.start) // 4) % len(base_section.progression))
            ]
            candidates = _candidate_harmony_degrees(broad_degree, base_section.label)
            degree = min(
                candidates,
                key=lambda candidate: _harmony_fit_cost(
                    lines,
                    key_context,
                    candidate,
                    start,
                    end,
                    broad_degree,
                    base_section.label,
                    previous_degree,
                ),
            )
            inferred.append(
                HarmonySection(
                    start,
                    end,
                    key_context.name,
                    (degree,),
                    f"inferred {base_section.label}",
                )
            )
            previous_degree = degree
        return inferred

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
        attempts = 1
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
                end_next,
            )
            score = self._segment_penalty(lines, voice_index, candidate, start_previous, end_next, harmony)
            score += 1.05 * self._style_penalty(candidate)
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
        end_next: int | None = None,
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
            cell_candidates: list[list[MusicalEvent]] = []
            for _ in range(4):
                cell = self.style_model.sample_cell(
                    rng,
                    remaining,
                    temperature,
                    phase=t,
                    previous_interval=previous_interval,
                    previous_duration=previous_duration,
                )
                candidate = self._events_from_cell(
                    lines,
                    voice_index,
                    t,
                    end,
                    cell.durations,
                    cell.intervals,
                    harmony,
                    spec,
                    previous,
                    previous_interval,
                    previous_duration,
                    subject_intervals,
                    rng,
                )
                if candidate:
                    cell_candidates.append(candidate)

            if not cell_candidates:
                duration = _fit_duration_to_grid(
                    self.style_model.sample_duration(
                        rng,
                        remaining,
                        temperature,
                        previous_duration=previous_duration,
                        phase=t,
                    ),
                    remaining,
                )
                key_context = _parse_key_name(_section_at(harmony, t).key_name)
                interval = self.style_model.sample_interval(
                    rng,
                    temperature,
                    previous_interval=previous_interval,
                )
                pitch = _fold_into_range(
                    _directional_scale_snap(key_context, previous, int(interval)),
                    spec.low,
                    spec.high,
                )
                cell_candidates.append([MusicalEvent(t, duration, pitch, "free counterpoint")])

            best_cell = min(
                cell_candidates,
                key=lambda candidate: self._cell_penalty(
                    lines,
                    voice_index,
                    candidate,
                    harmony,
                    previous,
                    previous_interval,
                    previous_duration,
                    end_next if candidate[-1].end >= end - 1e-6 else None,
                ),
            )
            events.extend(best_cell)
            for event in best_cell:
                previous_interval = int(event.pitch - previous) if event.pitch is not None else previous_interval
                previous_duration = event.duration
                if event.pitch is not None:
                    previous = event.pitch
            t = round(best_cell[-1].end, 6)
        return events

    def _events_from_cell(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        start: float,
        end: float,
        durations: tuple[float, ...],
        intervals: tuple[int, ...],
        harmony: list[HarmonySection],
        spec: VoiceSpec,
        previous: int,
        previous_interval: int | None,
        previous_duration: float | None,
        subject_intervals: list[int],
        rng: random.Random,
    ) -> list[MusicalEvent]:
        planned: list[tuple[float, float, int, int, list[int], set[int], int, bool, KeyContext]] = []
        t = start
        pitch_cursor = previous
        for raw_duration, raw_interval in zip(durations, intervals):
            if t >= end - 1e-6:
                break
            remaining = end - t
            duration = _fit_duration_to_grid(raw_duration, remaining)
            if duration < 0.25 - 1e-6:
                break
            interval = int(raw_interval)
            if subject_intervals and rng.random() < 0.12:
                interval = rng.choice(subject_intervals)
            key_context = _parse_key_name(_section_at(harmony, t).key_name)
            pitch = _directional_scale_snap(key_context, pitch_cursor, interval)
            pitch = _fold_into_range(pitch, spec.low, spec.high)
            if abs(pitch - pitch_cursor) > 12:
                pitch = _directional_scale_snap(key_context, pitch_cursor, -interval)
                pitch = _fold_into_range(pitch, spec.low, spec.high)
            section = _section_at(harmony, t)
            chord_degree = section.progression[int(((t - section.start) // 4) % len(section.progression))]
            root_pc = key_context.pitch_from_diatonic_index(chord_degree - 1) % 12
            chord = key_context.chord_pitches(
                chord_degree,
                spec.low,
                spec.high,
                include_seventh=False,
            )
            chord_pcs = {chord_pitch % 12 for chord_pitch in chord}
            planned.append(
                (
                    t,
                    duration,
                    pitch,
                    interval,
                    chord,
                    chord_pcs,
                    root_pc,
                    abs(t % 1.0) < 1e-6,
                    key_context,
                )
            )
            pitch_cursor = pitch
            t = round(t + duration, 6)
        if not planned:
            return []
        return self._decode_cell_path(
            lines,
            voice_index,
            planned,
            spec.low,
            spec.high,
            previous,
            previous_interval,
            previous_duration,
        )

    def _decode_cell_path(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        planned: list[tuple[float, float, int, int, list[int], set[int], int, bool, KeyContext]],
        low: int,
        high: int,
        previous: int,
        previous_interval: int | None,
        previous_duration: float | None,
    ) -> list[MusicalEvent]:
        states: list[tuple[float, list[int], int, int | None, float | None]] = [
            (0.0, [], previous, previous_interval, previous_duration)
        ]
        for t, duration, proposed, intended_interval, chord, chord_pcs, root_pc, strong, key_context in planned:
            next_states: list[tuple[float, list[int], int, int | None, float | None]] = []
            for cost, path, prev_pitch, prev_interval, prev_duration in states:
                for pitch in _cell_pitch_candidates(
                    proposed,
                    chord,
                    key_context,
                    low,
                    high,
                    prev_pitch,
                    intended_interval,
                ):
                    interval = max(-12, min(12, int(pitch - prev_pitch)))
                    pitch_cost = cost
                    pitch_cost += 0.5 * _melodic_continuity_cost(prev_pitch, pitch)
                    target = max(-12, min(12, int(intended_interval)))
                    pitch_cost += 3.0 * self.style_model.interval_penalty(interval, prev_interval)
                    pitch_cost += 4.2 * abs(interval - target)
                    pitch_cost += 0.6 * self.style_model.duration_penalty(duration, prev_duration, phase=t)
                    if strong and chord:
                        pitch_cost += 0 if pitch in chord else 2.5
                    pitch_cost += _target_harmony_pitch_cost(
                        pitch,
                        chord_pcs,
                        root_pc,
                        strong,
                        voice_index,
                        len(lines),
                    )
                    pitch_cost += self._vertical_pitch_penalty(
                        lines,
                        voice_index,
                        t,
                        duration,
                        pitch,
                        prev_pitch,
                    )
                    next_states.append((pitch_cost, path + [pitch], pitch, interval, duration))
            next_states.sort(key=lambda item: item[0])
            states = next_states[:4]
        intended_intervals = [interval for _, _, _, interval, _, _, _, _, _ in planned]
        best_path = min(
            states,
            key=lambda item: item[0] + _cell_contour_loss(previous, item[1], intended_intervals),
        )[1]
        return [
            MusicalEvent(t, duration, pitch, "free counterpoint")
            for (t, duration, _, _, _, _, _, _, _), pitch in zip(planned, best_path)
        ]

    def _vertical_pitch_penalty(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        t: float,
        duration: float,
        pitch: int,
        previous: int,
    ) -> float:
        penalty = 0.0
        strong_times = _covered_strong_times(t, duration)
        vertical_times = strong_times if strong_times else [t]
        crossing_times = _covered_grid_times(t, duration, 0.25)
        for other_index, other_line in enumerate(lines):
            if other_index == voice_index:
                continue
            for check_t in vertical_times:
                other = other_line.active_pitch(check_t)
                if other is None:
                    continue
                interval = abs(pitch - other) % 12
                strong_metric = _is_strong_metric_time(check_t)
                if interval in {1, 2, 6, 10, 11}:
                    penalty += 35 if strong_metric else 7
                if interval == 0:
                    penalty += 30 if strong_metric else 9
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
        return penalty

    def _cell_penalty(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        candidate: list[MusicalEvent],
        harmony: list[HarmonySection],
        previous: int | None,
        previous_interval: int | None,
        previous_duration: float | None,
        end_next: int | None,
    ) -> float:
        penalty = 0.0
        local_previous = previous
        local_previous_interval = previous_interval
        local_previous_duration = previous_duration
        for event in candidate:
            if event.pitch is None or local_previous is None:
                continue
            interval = int(event.pitch - local_previous)
            penalty += 0.5 * _melodic_continuity_cost(local_previous, event.pitch)
            penalty += 2.0 * self.style_model.interval_penalty(interval, local_previous_interval)
            penalty += 0.8 * self.style_model.duration_penalty(
                event.duration,
                local_previous_duration,
                phase=event.offset,
            )
            penalty += self._vertical_pitch_penalty(
                lines,
                voice_index,
                event.offset,
                event.duration,
                event.pitch,
                local_previous,
            )
            penalty += self._event_harmony_penalty(lines, voice_index, event, harmony)
            local_previous = event.pitch
            local_previous_interval = interval
            local_previous_duration = event.duration
        if local_previous is not None and end_next is not None:
            penalty += _melodic_continuity_cost(local_previous, end_next)
            penalty += self._next_boundary_penalty(lines, voice_index, candidate[-1], end_next)
        return penalty

    def _segment_penalty(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        segment: list[MusicalEvent],
        start_previous: int | None = None,
        end_next: int | None = None,
        harmony: list[HarmonySection] | None = None,
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
                check_times.update(_covered_grid_times(event.offset, event.duration, 0.25))
                for check_t in sorted(check_times):
                    other = line.active_pitch(check_t)
                    if other is None:
                        continue
                    interval = abs(event.pitch - other) % 12
                    strong_metric = _is_strong_metric_time(check_t)
                    if interval in {1, 2, 6, 10, 11}:
                        penalty += 7 if strong_metric else 1.5
                    if interval == 0:
                        penalty += 10 if strong_metric else 3
                    active_by_voice = [line.active_pitch(check_t) for line in lines]
                    active_by_voice[voice_index] = event.pitch
                    penalty += 0.35 * _sonority_spacing_cost(active_by_voice, voice_index)
                    if previous is not None:
                        other_previous = line.active_pitch(max(0.0, check_t - 0.5))
                        if other_previous is not None:
                            self_previous = previous if check_t - 0.5 < event.offset - 1e-6 else event.pitch
                            if _is_parallel_perfect_motion(self_previous, event.pitch, other_previous, other):
                                penalty += 200
            if harmony is not None:
                penalty += self._event_harmony_penalty(lines, voice_index, event, harmony)
            previous = event.pitch
        if previous is not None and end_next is not None and segment:
            penalty += _melodic_continuity_cost(previous, end_next)
            penalty += self._next_boundary_penalty(lines, voice_index, segment[-1], end_next)
        return penalty

    def _style_penalty(self, segment: list[MusicalEvent]) -> float:
        pitches = [event.pitch for event in segment if event.pitch is not None]
        durations = [event.duration for event in segment if event.pitch is not None]
        if len(pitches) < 2:
            return 0.0
        return self.style_model.style_penalty(pitches, durations)

    def _event_harmony_penalty(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        event: MusicalEvent,
        harmony: list[HarmonySection],
    ) -> float:
        if event.pitch is None:
            return 0.0
        check_times = set(_covered_strong_times(event.offset, event.duration))
        if not check_times:
            check_times.add(event.offset)
        penalty = 0.0
        for check_t in sorted(check_times):
            chord_pcs, root_pc = _target_chord_context(harmony, check_t)
            active_by_voice = [line.active_pitch(check_t) for line in lines]
            active_by_voice[voice_index] = event.pitch
            penalty += _harmonic_sonority_cost(
                active_by_voice,
                chord_pcs,
                root_pc,
                voice_index,
                _is_strong_metric_time(check_t),
            )
        return penalty

    def _polish_vertical_sonorities(
        self,
        lines: list[VoiceLine],
        harmony: list[HarmonySection],
        passes: int = 2,
    ) -> None:
        for _ in range(passes):
            changed = False
            for voice_index, line in enumerate(lines):
                for event_index, event in enumerate(list(line.events)):
                    if event.pitch is None or event.label != "free counterpoint":
                        continue
                    check_times = _covered_strong_times(event.offset, event.duration)
                    has_repeated_neighbor = _has_rapid_repeated_neighbor(line.events, event_index)
                    if not check_times and not has_repeated_neighbor:
                        continue
                    current_cost = self._polish_event_cost(
                        lines,
                        voice_index,
                        event_index,
                        event.pitch,
                        harmony,
                    )
                    best_pitch = event.pitch
                    best_cost = current_cost
                    for pitch in _polish_pitch_candidates(event, line.spec, harmony):
                        if pitch == event.pitch:
                            continue
                        candidate_cost = self._polish_event_cost(
                            lines,
                            voice_index,
                            event_index,
                            pitch,
                            harmony,
                        )
                        if candidate_cost < best_cost - 6.0:
                            best_pitch = pitch
                            best_cost = candidate_cost
                    if best_pitch != event.pitch:
                        line.events[event_index] = MusicalEvent(
                            event.offset,
                            event.duration,
                            best_pitch,
                            event.label,
                        )
                        line.events.sort(key=lambda item: (item.offset, item.duration))
                        changed = True
            if not changed:
                break

    def _polish_event_cost(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        event_index: int,
        pitch: int,
        harmony: list[HarmonySection],
    ) -> float:
        line = lines[voice_index]
        event = line.events[event_index]
        previous_pitch = _previous_event_pitch(line.events, event_index)
        next_pitch = _next_event_pitch(line.events, event_index)
        cost = 0.0
        if previous_pitch is not None:
            interval = int(pitch - previous_pitch)
            cost += 0.8 * _melodic_continuity_cost(previous_pitch, pitch)
            cost += 1.2 * self.style_model.interval_penalty(interval)
            if (
                previous_pitch == pitch
                and event.duration <= 0.5 + 1e-6
            ):
                cost += 90.0
        if next_pitch is not None:
            cost += 0.8 * _melodic_continuity_cost(pitch, next_pitch)
            if next_pitch == pitch and event.duration <= 0.5 + 1e-6:
                cost += 90.0

        for check_t in _covered_strong_times(event.offset, event.duration):
            chord_pcs, root_pc = _target_chord_context(harmony, check_t)
            active_by_voice = [voice.active_pitch(check_t) for voice in lines]
            active_by_voice[voice_index] = pitch
            for other_index, other in enumerate(active_by_voice):
                if other_index == voice_index or other is None:
                    continue
                interval_class = abs(pitch - other) % 12
                if interval_class in {1, 2, 6, 10, 11}:
                    cost += 60.0
                if interval_class == 0:
                    cost += 35.0
                if previous_pitch is not None:
                    other_previous = lines[other_index].active_pitch(max(0.0, check_t - 0.5))
                    if other_previous is not None and _is_parallel_perfect_motion(
                        previous_pitch,
                        pitch,
                        other_previous,
                        other,
                    ):
                        cost += 1500.0
                if next_pitch is not None:
                    other_next = lines[other_index].active_pitch(check_t + 0.5)
                    if other_next is not None and _is_parallel_perfect_motion(
                        pitch,
                        next_pitch,
                        other,
                        other_next,
                    ):
                        cost += 1500.0
            cost += 1.4 * _sonority_spacing_cost(active_by_voice, voice_index)
            cost += 2.2 * _harmonic_sonority_cost(
                active_by_voice,
                chord_pcs,
                root_pc,
                voice_index,
                True,
            )
        return cost

    def _next_boundary_penalty(
        self,
        lines: list[VoiceLine],
        voice_index: int,
        final_event: MusicalEvent,
        next_pitch: int,
    ) -> float:
        if final_event.pitch is None:
            return 0.0
        boundary = final_event.end
        penalty = 0.0
        for other_index, line in enumerate(lines):
            if other_index == voice_index:
                continue
            other_previous = line.active_pitch(max(0.0, boundary - 0.5))
            other_next = line.active_pitch(boundary)
            if other_previous is None or other_next is None:
                continue
            if _is_parallel_perfect_motion(final_event.pitch, next_pitch, other_previous, other_next):
                penalty += 1200.0
        return penalty

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


def _merge_sustained_free_repetitions(lines: list[VoiceLine]) -> None:
    for line in lines:
        merged: list[MusicalEvent] = []
        for event in sorted(line.events, key=lambda item: (item.offset, item.duration)):
            if (
                merged
                and event.label == "free counterpoint"
                and merged[-1].label == "free counterpoint"
                and event.pitch is not None
                and event.pitch == merged[-1].pitch
                and abs(merged[-1].end - event.offset) < 1e-6
                and _should_tie_free_repetition(merged[-1], event)
            ):
                previous = merged[-1]
                merged[-1] = MusicalEvent(
                    previous.offset,
                    round(previous.duration + event.duration, 6),
                    previous.pitch,
                    previous.label,
                )
            else:
                merged.append(event)
        line.events = merged


def _should_tie_free_repetition(previous: MusicalEvent, current: MusicalEvent) -> bool:
    total = previous.duration + current.duration
    if total > 4.0 + 1e-6:
        return False
    return previous.duration >= 1.0 - 1e-6 or current.duration >= 1.0 - 1e-6 or total >= 2.0 - 1e-6


def _default_measures(subject_duration: float, voices: int) -> int:
    return max(28, min(56, int(math.ceil((subject_duration * (voices + 7) + 48) / 4))))


def _ceil_to_measure(value: float) -> float:
    return math.ceil(value / 4.0) * 4.0


def _fit_duration_to_grid(duration: float, remaining: float) -> float:
    value = max(0.25, round(duration * 4) / 4)
    value = min(value, remaining)
    if 0.0 < remaining - value < 0.25 - 1e-6:
        value = remaining
    return round(value, 6)


def _round_to_grid(value: float, grid: float) -> float:
    return round(value / grid) * grid


def _parse_key_name(name: str) -> KeyContext:
    pieces = name.split()
    if len(pieces) == 1:
        return KeyContext(pieces[0], "minor")
    return KeyContext(pieces[0], pieces[1])


def _candidate_harmony_degrees(broad_degree: int, label: str) -> tuple[int, ...]:
    if "cadence" in label:
        candidates = [broad_degree, 2, 5, 1]
    elif "final" in label:
        candidates = [broad_degree, 1, 4, 5, 2, 6]
    else:
        candidates = [broad_degree, 1, 5, 4, 2, 6, 3]
    ordered: list[int] = []
    for degree in candidates:
        if degree not in ordered:
            ordered.append(degree)
    return tuple(ordered)


def _harmony_fit_cost(
    lines: list[VoiceLine],
    key_context: KeyContext,
    degree: int,
    start: float,
    end: float,
    broad_degree: int,
    label: str,
    previous_degree: int | None,
) -> float:
    chord = key_context.chord_pitches(degree, 24, 96, include_seventh=False)
    chord_pcs = {pitch % 12 for pitch in chord}
    root_pc = key_context.pitch_from_diatonic_index(degree - 1) % 12
    cost = 0.0
    checks = 0
    for offset in range(int(start), int(math.ceil(end))):
        active = [line.active_pitch(float(offset)) for line in lines]
        active = [pitch for pitch in active if pitch is not None]
        if not active:
            continue
        checks += 1
        chord_members = {pitch % 12 for pitch in active if pitch % 12 in chord_pcs}
        non_chord_count = sum(1 for pitch in active if pitch % 12 not in chord_pcs)
        cost += non_chord_count * 5.0
        if len(active) >= 2 and len(chord_members) < 2:
            cost += 8.0
        if len(active) >= 4 and len(chord_members) < 3:
            cost += 5.0
        bass = min(active)
        if bass % 12 not in chord_pcs:
            cost += 7.0
        elif bass % 12 == root_pc:
            cost -= 2.0
        elif bass % 12 == (root_pc + 7) % 12:
            cost -= 0.5
        else:
            cost += 1.5
    if checks:
        cost /= checks
    if degree == broad_degree:
        cost -= 4.0 if "cadence" in label else 1.2
    cost += _harmony_transition_cost(previous_degree, degree)
    return cost


def _harmony_transition_cost(previous_degree: int | None, degree: int) -> float:
    if previous_degree is None:
        return 0.0
    if previous_degree == degree:
        return 0.8
    if previous_degree == 5 and degree == 1:
        return -3.5
    if previous_degree in {2, 4} and degree == 5:
        return -2.0
    if previous_degree == 1 and degree in {4, 5, 6}:
        return -1.0
    if degree == 5:
        return -0.5
    if degree == 1:
        return -0.3
    return 0.4


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


def _is_strong_metric_time(value: float) -> bool:
    return abs(value - round(value)) < 1e-6


def _covered_grid_times(start: float, duration: float, grid: float) -> list[float]:
    first = math.ceil((start - 1e-6) / grid)
    last = math.floor((start + duration - 1e-6) / grid)
    return [round(i * grid, 6) for i in range(first, last + 1)]


def _cell_pitch_candidates(
    proposed: int,
    chord: list[int],
    key_context: KeyContext,
    low: int,
    high: int,
    previous: int | None = None,
    intended_interval: int = 0,
) -> set[int]:
    candidates = {proposed}
    for raw in range(proposed - 5, proposed + 6):
        candidates.add(key_context.snap_to_scale(raw))
    candidates.update(pitch for pitch in chord if abs(pitch - proposed) <= 5)
    candidates = {pitch for pitch in candidates if low <= pitch <= high}
    if previous is not None and intended_interval:
        directed = {
            pitch
            for pitch in candidates
            if (pitch - previous) * intended_interval > 0
        }
        if directed:
            candidates = directed
    if candidates:
        return candidates
    return {_fold_into_range(proposed, low, high)}


def _target_chord_context(harmony: list[HarmonySection], t: float) -> tuple[set[int], int]:
    section = _section_at(harmony, t)
    key_context = _parse_key_name(section.key_name)
    degree = section.progression[int(((t - section.start) // 4) % len(section.progression))]
    chord_pcs = {
        pitch % 12
        for pitch in key_context.chord_pitches(
            degree,
            24,
            96,
            include_seventh=False,
        )
    }
    root_pc = key_context.pitch_from_diatonic_index(degree - 1) % 12
    return chord_pcs, root_pc


def _polish_pitch_candidates(
    event: MusicalEvent,
    spec: VoiceSpec,
    harmony: list[HarmonySection],
) -> set[int]:
    chord_pcs, _ = _target_chord_context(harmony, event.offset)
    section = _section_at(harmony, event.offset)
    key_context = _parse_key_name(section.key_name)
    candidates = {event.pitch} if event.pitch is not None else set()
    if event.pitch is not None:
        for raw in range(event.pitch - 4, event.pitch + 5):
            snapped = key_context.snap_to_scale(raw)
            if spec.low <= snapped <= spec.high:
                candidates.add(snapped)
        for pitch in range(spec.low, spec.high + 1):
            if pitch % 12 in chord_pcs and abs(pitch - event.pitch) <= 5:
                candidates.add(pitch)
    return candidates


def _has_rapid_repeated_neighbor(events: list[MusicalEvent], event_index: int) -> bool:
    event = events[event_index]
    if event.pitch is None or event.label != "free counterpoint":
        return False
    neighbors: list[MusicalEvent] = []
    for previous in reversed(events[:event_index]):
        if previous.pitch is not None:
            neighbors.append(previous)
            break
    for following in events[event_index + 1 :]:
        if following.pitch is not None:
            neighbors.append(following)
            break
    for neighbor in neighbors:
        if neighbor.label != "free counterpoint" or neighbor.pitch != event.pitch:
            continue
        before, after = (neighbor, event) if neighbor.offset < event.offset else (event, neighbor)
        if abs(before.end - after.offset) < 1e-6 and min(before.duration, after.duration) <= 0.5 + 1e-6:
            return True
    return False


def _previous_event_pitch(events: list[MusicalEvent], event_index: int) -> int | None:
    for event in reversed(events[:event_index]):
        if event.pitch is not None:
            return event.pitch
    return None


def _next_event_pitch(events: list[MusicalEvent], event_index: int) -> int | None:
    for event in events[event_index + 1 :]:
        if event.pitch is not None:
            return event.pitch
    return None


def _target_harmony_pitch_cost(
    pitch: int,
    chord_pcs: set[int],
    root_pc: int,
    strong: bool,
    voice_index: int,
    voice_count: int,
) -> float:
    pitch_pc = pitch % 12
    if pitch_pc not in chord_pcs:
        return 8.0 if strong else 2.0
    if voice_index == voice_count - 1:
        if pitch_pc == root_pc:
            return 0.0
        if pitch_pc == (root_pc + 7) % 12:
            return 1.0
        return 3.0 if strong else 1.0
    return 0.0


def _harmonic_sonority_cost(
    active_by_voice: list[int | None],
    chord_pcs: set[int],
    root_pc: int,
    voice_index: int,
    strong: bool,
) -> float:
    active = [pitch for pitch in active_by_voice if pitch is not None]
    if len(active) < 2:
        return 0.0
    chord_members = {pitch % 12 for pitch in active if pitch % 12 in chord_pcs}
    non_chord_count = sum(1 for pitch in active if pitch % 12 not in chord_pcs)
    multiplier = 1.0 if strong else 0.25
    cost = non_chord_count * 4.0 * multiplier
    if len(active) >= 3 and len(chord_members) < 2:
        cost += 9.0 * multiplier
    if len(active) >= 4 and len(chord_members) < 3:
        cost += 8.0 * multiplier
    bass = min(active)
    if bass % 12 not in chord_pcs:
        cost += 9.0 * multiplier
    elif bass % 12 not in {root_pc, (root_pc + 7) % 12}:
        cost += 4.0 * multiplier

    proposed = active_by_voice[voice_index]
    if proposed is not None and proposed % 12 not in chord_pcs:
        cost += 2.0 * multiplier
    return cost


def _directional_scale_snap(key_context: KeyContext, previous: int, interval: int) -> int:
    raw = previous + interval
    snapped = key_context.snap_to_scale(raw)
    if interval == 0:
        return snapped
    if interval > 0 and snapped <= previous:
        return _nearest_scale_pitch(key_context, raw, lambda pitch: pitch > previous)
    if interval < 0 and snapped >= previous:
        return _nearest_scale_pitch(key_context, raw, lambda pitch: pitch < previous)
    return snapped


def _nearest_scale_pitch(key_context: KeyContext, raw: int, predicate) -> int:
    candidates = []
    for octave in range((raw // 12) - 2, (raw // 12) + 3):
        base = 12 * octave + key_context.pc
        candidates.extend(base + interval for interval in key_context.intervals)
    filtered = [pitch for pitch in candidates if predicate(pitch)]
    if not filtered:
        return key_context.snap_to_scale(raw)
    return min(filtered, key=lambda pitch: (abs(pitch - raw), abs(pitch)))


def _cell_contour_loss(previous: int, path: list[int], intended_intervals: list[int]) -> float:
    if not path:
        return 0.0
    actual_intervals = [path[0] - previous]
    actual_intervals.extend(current - before for before, current in zip(path, path[1:]))

    intended_cursor = 0
    actual_cursor = 0
    intended_points = [0]
    actual_points = [0]
    sign_mismatch = 0
    for intended, actual in zip(intended_intervals, actual_intervals):
        intended = max(-12, min(12, int(intended)))
        actual = max(-12, min(12, int(actual)))
        intended_cursor += intended
        actual_cursor += actual
        intended_points.append(intended_cursor)
        actual_points.append(actual_cursor)
        if intended and actual and (intended > 0) != (actual > 0):
            sign_mismatch += 1

    intended_range = max(intended_points) - min(intended_points)
    actual_range = max(actual_points) - min(actual_points)
    range_loss = max(0, intended_range - actual_range)
    endpoint_loss = abs(intended_points[-1] - actual_points[-1]) * 1.2
    return 9.0 * range_loss + 9.0 * sign_mismatch + endpoint_loss


def _melodic_continuity_cost(previous: int, current: int) -> float:
    leap = abs(current - previous)
    if leap == 0:
        return 5.5
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
    ascending = sorted(active)
    for index in range(0, len(ascending) - 2):
        if ascending[index + 2] - ascending[index] <= 4:
            cost += 800

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
