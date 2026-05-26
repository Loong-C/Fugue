from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


STYLE_PROFILE_VERSION = 8
RHYTHM_CLASSES = ("plain", "sixteenth", "dotted", "syncopated", "long")

DEFAULT_INTERVAL_WEIGHTS = {
    -7: 3,
    -5: 8,
    -4: 12,
    -3: 16,
    -2: 30,
    -1: 38,
    0: 18,
    1: 38,
    2: 30,
    3: 16,
    4: 12,
    5: 8,
    7: 3,
}

DEFAULT_DURATION_WEIGHTS = {
    0.25: 8,
    0.5: 42,
    0.75: 8,
    1.0: 40,
    1.5: 6,
    2.0: 10,
}


@dataclass(frozen=True)
class StyleCell:
    durations: tuple[float, ...]
    intervals: tuple[int, ...]
    phase: float
    weight: float
    rhythm_class: str = "plain"

    @property
    def span(self) -> float:
        return round(sum(self.durations), 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "durations": list(self.durations),
            "intervals": list(self.intervals),
            "phase": self.phase,
            "weight": self.weight,
            "rhythm_class": self.rhythm_class,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "StyleCell":
        durations = tuple(float(value) for value in data.get("durations", []))
        phase = float(data.get("phase", 0.0))
        rhythm_class = str(data.get("rhythm_class") or _classify_rhythm(durations, phase))
        return cls(
            durations,
            tuple(int(value) for value in data.get("intervals", [])),
            phase,
            float(data.get("weight", 1.0)),
            rhythm_class,
        )


@dataclass
class CorpusStyleModel:
    interval_weights: dict[int, float]
    duration_weights: dict[float, float]
    interval_transitions: dict[int, dict[int, float]] = field(default_factory=dict)
    duration_transitions: dict[float, dict[float, float]] = field(default_factory=dict)
    source: str = "built-in"
    duration_phase_weights: dict[float, dict[float, float]] = field(default_factory=dict)
    melodic_cells: list[StyleCell] = field(default_factory=list)
    rhythm_class_weights: dict[str, float] = field(default_factory=dict)
    melodic_cells_by_phase: dict[float, list[StyleCell]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.duration_phase_weights:
            self.duration_phase_weights = _default_phase_weights(self.duration_weights)
        if not self.melodic_cells:
            self.melodic_cells = _fallback_style_cells()
        if not self.rhythm_class_weights:
            self.rhythm_class_weights = _rhythm_class_weights(
                _style_cells_to_counter(self.melodic_cells),
                self.duration_weights,
            )
        self.melodic_cells_by_phase = {}
        for cell in self.melodic_cells:
            self.melodic_cells_by_phase.setdefault(cell.phase, []).append(cell)

    @classmethod
    def load(
        cls,
        root: Path,
        cache_path: Path | None = None,
        rebuild: bool = False,
    ) -> "CorpusStyleModel":
        cache = cache_path or (root / "data" / "processed" / "style_profile.json")
        if cache.exists() and not rebuild:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if (
                data.get("schema_version") != STYLE_PROFILE_VERSION
                or
                "interval_transitions" not in data
                or "duration_transitions" not in data
                or "duration_phase_weights" not in data
                or "melodic_cells" not in data
                or "rhythm_class_weights" not in data
            ):
                return cls.load(root, cache, rebuild=True)
            return cls(
                {int(k): float(v) for k, v in data["interval_weights"].items()},
                {float(k): float(v) for k, v in data["duration_weights"].items()},
                {
                    int(k): {int(next_k): float(weight) for next_k, weight in value.items()}
                    for k, value in data["interval_transitions"].items()
                },
                {
                    float(k): {float(next_k): float(weight) for next_k, weight in value.items()}
                    for k, value in data["duration_transitions"].items()
                },
                data.get("source", str(cache)),
                {
                    float(k): {float(next_k): float(weight) for next_k, weight in value.items()}
                    for k, value in data["duration_phase_weights"].items()
                },
                [StyleCell.from_dict(value) for value in data["melodic_cells"]],
                {str(k): float(v) for k, v in data["rhythm_class_weights"].items()},
            )

        jsb = root / "data" / "raw" / "jsb-chorales-dataset" / "Jsb16thSeparated.json"
        if not jsb.exists():
            return cls(
                dict(DEFAULT_INTERVAL_WEIGHTS),
                dict(DEFAULT_DURATION_WEIGHTS),
                source="built-in",
            )

        model = cls.from_jsb_json(jsb)
        fugue_dir = root / "data" / "raw" / "humdrum" / "bach-wtc-fugues" / "kern"
        if fugue_dir.exists():
            model.augment_from_humdrum_fugues(fugue_dir)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    "schema_version": STYLE_PROFILE_VERSION,
                    "interval_weights": model.interval_weights,
                    "duration_weights": model.duration_weights,
                    "interval_transitions": model.interval_transitions,
                    "duration_transitions": model.duration_transitions,
                    "duration_phase_weights": model.duration_phase_weights,
                    "melodic_cells": [cell.to_dict() for cell in model.melodic_cells],
                    "rhythm_class_weights": model.rhythm_class_weights,
                    "source": model.source,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return model

    @classmethod
    def from_jsb_json(cls, path: Path) -> "CorpusStyleModel":
        data = json.loads(path.read_text(encoding="utf-8"))
        interval_counter: Counter[int] = Counter(DEFAULT_INTERVAL_WEIGHTS)
        duration_counter: Counter[float] = Counter(DEFAULT_DURATION_WEIGHTS)
        interval_transition_counter: dict[int, Counter[int]] = {}
        duration_transition_counter: dict[float, Counter[float]] = {}
        duration_phase_counter: dict[float, Counter[float]] = {}
        cell_counter: Counter[tuple[tuple[float, ...], tuple[int, ...], float]] = Counter()
        for split in ("train", "valid", "test"):
            for chorale in data.get(split, []):
                if not chorale:
                    continue
                voice_count = len(chorale[0])
                for voice in range(voice_count):
                    note_runs = _runs_from_frames_with_offsets(
                        [frame[voice] for frame in chorale],
                        frame_quarters=0.25,
                    )
                    _update_melodic_counters(
                        note_runs,
                        interval_counter,
                        duration_counter,
                        interval_transition_counter,
                        duration_transition_counter,
                        duration_phase_counter,
                        cell_counter,
                    )
        return cls(
            dict(interval_counter),
            dict(duration_counter),
            _counter_map_to_dict(interval_transition_counter),
            _counter_map_to_dict(duration_transition_counter),
            f"JSB chorales Markov profile: {path}",
            _counter_map_to_dict(duration_phase_counter),
            _top_style_cells(cell_counter),
            _rhythm_class_weights(cell_counter, duration_counter),
        )

    def augment_from_humdrum_fugues(self, fugue_dir: Path, max_scores: int = 48) -> None:
        from music21 import converter, note

        interval_counter: Counter[int] = Counter(self.interval_weights)
        duration_counter: Counter[float] = Counter(self.duration_weights)
        interval_transition_counter = _dict_to_counter_map(self.interval_transitions)
        duration_transition_counter = _dict_to_counter_map(self.duration_transitions)
        duration_phase_counter = _dict_to_counter_map(self.duration_phase_weights)
        cell_counter = _style_cells_to_counter(self.melodic_cells)

        parsed = 0
        for path in sorted(fugue_dir.glob("*.krn"))[:max_scores]:
            try:
                score = converter.parse(path)
            except Exception:
                continue
            parsed += 1
            for part in score.parts:
                note_runs: list[tuple[float, int, float]] = []
                for element in part.flatten().notesAndRests:
                    if isinstance(element, note.Note):
                        note_runs.append(
                            (
                                _quantize(float(element.offset), 0.25),
                                int(element.pitch.midi),
                                max(0.25, round(float(element.quarterLength) * 4) / 4),
                            )
                        )
                _update_melodic_counters(
                    note_runs,
                    interval_counter,
                    duration_counter,
                    interval_transition_counter,
                    duration_transition_counter,
                    duration_phase_counter,
                    cell_counter,
                )

        if parsed:
            self.interval_weights = dict(interval_counter)
            self.duration_weights = dict(duration_counter)
            self.interval_transitions = _counter_map_to_dict(interval_transition_counter)
            self.duration_transitions = _counter_map_to_dict(duration_transition_counter)
            self.duration_phase_weights = _counter_map_to_dict(duration_phase_counter)
            self.melodic_cells = _top_style_cells(cell_counter)
            self.rhythm_class_weights = _rhythm_class_weights(cell_counter, duration_counter)
            self.source = f"{self.source}; WTC fugues: {fugue_dir} ({parsed} scores)"

    def sample_interval(
        self,
        rng: random.Random,
        temperature: float = 1.0,
        previous_interval: int | None = None,
    ) -> int:
        weights = self.interval_weights
        if previous_interval is not None:
            weights = self.interval_transitions.get(previous_interval, weights)
        return int(_weighted_sample(weights, rng, temperature))

    def sample_duration(
        self,
        rng: random.Random,
        remaining: float,
        temperature: float = 1.0,
        previous_duration: float | None = None,
        phase: float | None = None,
    ) -> float:
        weights = self.duration_weights
        if previous_duration is not None:
            weights = self.duration_transitions.get(previous_duration, weights)
        if phase is not None:
            phase_weights = self.duration_phase_weights.get(_phase_key(phase))
            if phase_weights:
                weights = _blend_weights(weights, phase_weights)
        allowed = {
            duration: weight
            for duration, weight in weights.items()
            if duration <= remaining + 1e-6
        }
        if not allowed:
            return max(0.25, remaining)
        value = float(_weighted_sample(allowed, rng, temperature))
        return min(value, remaining)

    def sample_cell(
        self,
        rng: random.Random,
        remaining: float,
        temperature: float = 1.0,
        phase: float | None = None,
        previous_interval: int | None = None,
        previous_duration: float | None = None,
    ) -> StyleCell:
        phase_key = None if phase is None else _phase_key(phase)
        source_cells = self.melodic_cells
        if phase_key is not None and phase_key in self.melodic_cells_by_phase:
            source_cells = self.melodic_cells_by_phase[phase_key]
        candidates = [
            cell
            for cell in source_cells
            if cell.span <= remaining + 1e-6
            and (remaining - cell.span < 1e-6 or remaining - cell.span >= 0.25 - 1e-6)
        ]
        if candidates:
            candidates_by_class: dict[str, list[StyleCell]] = {}
            for cell in candidates:
                candidates_by_class.setdefault(cell.rhythm_class, []).append(cell)
            class_weights = _candidate_class_weights(candidates_by_class, self.rhythm_class_weights)
            rhythm_class = str(_weighted_sample(class_weights, rng, max(1.0, temperature * 1.2)))
            class_candidates = candidates_by_class.get(rhythm_class, candidates)
            weights = {index: cell.weight for index, cell in enumerate(class_candidates)}
            return class_candidates[int(_weighted_sample(weights, rng, temperature))]

        duration = self.sample_duration(
            rng,
            remaining,
            temperature,
            previous_duration=previous_duration,
            phase=phase,
        )
        interval = self.sample_interval(rng, temperature, previous_interval=previous_interval)
        phase_value = 0.0 if phase_key is None else phase_key
        return StyleCell(
            (duration,),
            (interval,),
            phase_value,
            1.0,
            _classify_rhythm((duration,), phase_value),
        )

    def interval_penalty(self, interval: int, previous_interval: int | None = None) -> float:
        interval = max(-12, min(12, int(interval)))
        weights = (
            self.interval_transitions.get(previous_interval, self.interval_weights)
            if previous_interval is not None
            else self.interval_weights
        )
        return _negative_log_probability(interval, weights)

    def duration_penalty(
        self,
        duration: float,
        previous_duration: float | None = None,
        phase: float | None = None,
    ) -> float:
        duration = max(0.25, round(duration * 4) / 4)
        weights = (
            self.duration_transitions.get(previous_duration, self.duration_weights)
            if previous_duration is not None
            else self.duration_weights
        )
        if phase is not None:
            phase_weights = self.duration_phase_weights.get(_phase_key(phase))
            if phase_weights:
                weights = _blend_weights(weights, phase_weights)
        return _negative_log_probability(duration, weights)

    def style_penalty(self, pitches: list[int], durations: list[float]) -> float:
        """Negative log-likelihood style cost for a melodic fragment."""
        penalty = 0.0
        previous_interval = None
        for a, b in zip(pitches, pitches[1:]):
            interval = max(-12, min(12, int(b - a)))
            penalty += self.interval_penalty(interval, previous_interval)
            previous_interval = interval

        previous_duration = None
        for duration in durations:
            duration = max(0.25, round(duration * 4) / 4)
            penalty += self.duration_penalty(duration, previous_duration)
            previous_duration = duration
        return penalty


def _weighted_sample(
    weights: dict[object, float],
    rng: random.Random,
    temperature: float,
) -> object:
    temperature = max(0.1, temperature)
    items = list(weights.items())
    adjusted = [math.pow(max(weight, 0.0001), 1.0 / temperature) for _, weight in items]
    total = sum(adjusted)
    pick = rng.random() * total
    cursor = 0.0
    for (value, _), weight in zip(items, adjusted):
        cursor += weight
        if pick <= cursor:
            return value
    return items[-1][0]


def _runs_from_frames(frames: list[int], frame_quarters: float) -> list[tuple[int, float]]:
    return [(pitch, duration) for _, pitch, duration in _runs_from_frames_with_offsets(frames, frame_quarters)]


def _runs_from_frames_with_offsets(frames: list[int], frame_quarters: float) -> list[tuple[float, int, float]]:
    if not frames:
        return []
    runs: list[tuple[float, int, float]] = []
    previous = frames[0]
    run = 1
    start_index = 0
    for index, pitch in enumerate(frames[1:], start=1):
        if pitch == previous:
            run += 1
            continue
        if previous >= 0:
            runs.append(
                (
                    _quantize(start_index * frame_quarters, 0.25),
                    int(previous),
                    max(0.25, round(run * frame_quarters * 4) / 4),
                )
            )
        previous = pitch
        run = 1
        start_index = index
    if previous >= 0:
        runs.append(
            (
                _quantize(start_index * frame_quarters, 0.25),
                int(previous),
                max(0.25, round(run * frame_quarters * 4) / 4),
            )
        )
    return runs


def _update_melodic_counters(
    note_runs: list[tuple[float, int, float]],
    interval_counter: Counter[int],
    duration_counter: Counter[float],
    interval_transition_counter: dict[int, Counter[int]],
    duration_transition_counter: dict[float, Counter[float]],
    duration_phase_counter: dict[float, Counter[float]],
    cell_counter: Counter[tuple[tuple[float, ...], tuple[int, ...], float, str]],
) -> None:
    if not note_runs:
        return
    pitches = [pitch for _, pitch, _ in note_runs]
    durations = [duration for _, _, duration in note_runs]

    for offset, _, duration in note_runs:
        duration_counter[duration] += 1
        duration_phase_counter.setdefault(_phase_key(offset), Counter())[duration] += 1
    for previous_duration, duration in zip(durations, durations[1:]):
        duration_transition_counter.setdefault(previous_duration, Counter())[duration] += 1

    intervals = [
        max(-12, min(12, int(b - a)))
        for a, b in zip(pitches, pitches[1:])
        if abs(b - a) <= 12
    ]
    for interval in intervals:
        interval_counter[interval] += 1
    for previous_interval, interval in zip(intervals, intervals[1:]):
        interval_transition_counter.setdefault(previous_interval, Counter())[interval] += 1
    _update_cell_counter(note_runs, cell_counter)


def _counter_map_to_dict(counter_map: dict[int | float, Counter[int | float]]) -> dict:
    return {key: dict(counter) for key, counter in counter_map.items()}


def _dict_to_counter_map(values: dict[int | float, dict[int | float, float]]) -> dict:
    return {key: Counter(value) for key, value in values.items()}


def _update_cell_counter(
    note_runs: list[tuple[float, int, float]],
    cell_counter: Counter[tuple[tuple[float, ...], tuple[int, ...], float, str]],
) -> None:
    if len(note_runs) < 3:
        return
    for start in range(1, len(note_runs) - 1):
        phase = _phase_key(note_runs[start][0])
        durations: list[float] = []
        intervals: list[int] = []
        span = 0.0
        previous_pitch = note_runs[start - 1][1]
        for offset, pitch, duration in note_runs[start : start + 8]:
            if abs(offset - note_runs[start][0] - span) > 0.25 + 1e-6:
                break
            if duration < 0.25 - 1e-6:
                break
            quantized_duration = _quantize(duration, 0.25)
            if quantized_duration > 4.0 + 1e-6:
                break
            interval = max(-12, min(12, int(pitch - previous_pitch)))
            durations.append(quantized_duration)
            intervals.append(interval)
            span = round(span + quantized_duration, 6)
            previous_pitch = pitch
            if 1.0 - 1e-6 <= span <= 4.0 + 1e-6:
                duration_tuple = tuple(durations)
                rhythm_class = _classify_rhythm(duration_tuple, phase)
                if len(durations) >= _minimum_cell_notes(rhythm_class):
                    cell_counter[(duration_tuple, tuple(intervals), phase, rhythm_class)] += 1
            if span >= 4.0 - 1e-6:
                break


def _top_style_cells(
    cell_counter: Counter[tuple[tuple[float, ...], tuple[int, ...], float, str]],
    limit: int = 6144,
) -> list[StyleCell]:
    if not cell_counter:
        return _fallback_style_cells()

    by_class: dict[str, list[tuple[tuple[tuple[float, ...], tuple[int, ...], float, str], float]]] = {}
    for key, weight in cell_counter.items():
        by_class.setdefault(key[3], []).append((key, float(weight)))

    selected: dict[tuple[tuple[float, ...], tuple[int, ...], float, str], float] = {}
    class_limit = max(64, limit // max(1, len(by_class)))
    for rhythm_class in RHYTHM_CLASSES:
        items = by_class.get(rhythm_class, [])
        items.sort(key=lambda item: item[1], reverse=True)
        for key, weight in items[:class_limit]:
            selected[key] = weight

    if len(selected) < limit:
        for key, weight in cell_counter.most_common(limit):
            selected.setdefault(key, float(weight))
            if len(selected) >= limit:
                break

    ranked = sorted(selected.items(), key=lambda item: item[1], reverse=True)
    return [
        StyleCell(durations, intervals, phase, float(weight), rhythm_class)
        for (durations, intervals, phase, rhythm_class), weight in ranked[:limit]
    ]


def _style_cells_to_counter(cells: list[StyleCell]) -> Counter[tuple[tuple[float, ...], tuple[int, ...], float, str]]:
    counter: Counter[tuple[tuple[float, ...], tuple[int, ...], float, str]] = Counter()
    for cell in cells:
        rhythm_class = cell.rhythm_class or _classify_rhythm(cell.durations, cell.phase)
        counter[(cell.durations, cell.intervals, cell.phase, rhythm_class)] += cell.weight
    return counter


def _fallback_style_cells() -> list[StyleCell]:
    patterns = [
        ((0.5, 0.5, 1.0), (2, 1, -1)),
        ((0.5, 0.5, 1.0), (-2, -1, 1)),
        ((0.25, 0.25, 0.5, 1.0), (1, 2, -1, -2)),
        ((0.25, 0.25, 0.25, 0.25, 1.0), (-1, -2, 1, 2, 1)),
        ((0.75, 0.25, 0.5, 0.5), (2, -1, -2, 1)),
        ((0.5, 0.75, 0.25, 0.5), (-2, 1, 2, -1)),
        ((0.5, 0.5, 1.0), (1, 2, 2)),
        ((0.5, 0.5, 1.0), (-1, -2, -2)),
        ((1.0, 0.5, 0.5), (2, -1, -2)),
        ((1.0, 0.5, 0.5), (-2, 1, 2)),
        ((0.5, 1.0, 0.5), (2, -1, 1)),
        ((1.5, 0.5, 1.0), (-2, 1, 2)),
        ((2.0, 0.5, 0.5), (1, 2, -1)),
        ((1.0, 1.0, 1.0), (2, 1, -2)),
        ((1.0, 1.0, 1.0), (-2, -1, 2)),
        ((0.5, 0.5, 0.5, 0.5), (1, 2, -1, -2)),
        ((0.5, 0.5, 0.5, 0.5), (-1, -2, 1, 2)),
    ]
    cells: list[StyleCell] = []
    for phase in (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5):
        for index, (durations, intervals) in enumerate(patterns):
            cells.append(
                StyleCell(
                    durations,
                    intervals,
                    phase,
                    float(len(patterns) - index),
                    _classify_rhythm(durations, phase),
                )
            )
    return cells


def _default_phase_weights(duration_weights: dict[float, float]) -> dict[float, dict[float, float]]:
    normalized = {float(duration): float(weight) for duration, weight in duration_weights.items()}
    return {round(index * 0.25, 6): dict(normalized) for index in range(16)}


def _phase_key(offset: float) -> float:
    return _quantize(offset % 4.0, 0.25)


def _classify_rhythm(durations: tuple[float, ...], phase: float = 0.0) -> str:
    values = tuple(_quantize(float(duration), 0.25) for duration in durations)
    if any(duration < 0.5 - 1e-6 for duration in values):
        return "sixteenth"
    if any(_is_odd_quarter_duration(duration) for duration in values):
        return "dotted"

    cursor = _quantize(phase, 0.25)
    for duration in values:
        beat_phase = round(cursor % 1.0, 6)
        crosses_beat = math.floor(cursor + 1e-6) < math.floor(cursor + duration - 1e-6)
        if beat_phase in {0.25, 0.75} or (abs(beat_phase - 0.5) < 1e-6 and crosses_beat):
            return "syncopated"
        cursor = round(cursor + duration, 6)

    if any(duration >= 2.0 - 1e-6 for duration in values):
        return "long"
    return "plain"


def _is_odd_quarter_duration(duration: float) -> bool:
    units = round(duration / 0.25)
    return units % 2 == 1 and units > 1


def _minimum_cell_notes(rhythm_class: str) -> int:
    return 2 if rhythm_class in {"dotted", "long"} else 3


def _rhythm_class_weights(
    cell_counter: Counter[tuple[tuple[float, ...], tuple[int, ...], float, str]],
    duration_weights: dict[float, float] | Counter[float] | None = None,
) -> dict[str, float]:
    weights = {rhythm_class: 0.0 for rhythm_class in RHYTHM_CLASSES}
    for (durations, _, _, rhythm_class), weight in cell_counter.items():
        span = max(0.25, sum(durations))
        event_count = max(1, len(durations))
        density_corrected = (span * span) / math.pow(event_count, 2.0)
        weights[rhythm_class] = weights.get(rhythm_class, 0.0) + float(weight) * density_corrected
    if duration_weights:
        duration_priors = _duration_rhythm_class_weights(duration_weights)
        weights = {
            rhythm_class: math.sqrt(max(weight, 0.0001) * max(duration_priors.get(rhythm_class, 0.0001), 0.0001))
            for rhythm_class, weight in weights.items()
        }
    return {key: value for key, value in weights.items() if value > 0.0}


def _duration_rhythm_class_weights(duration_weights: dict[float, float] | Counter[float]) -> dict[str, float]:
    weights = {rhythm_class: 0.0 for rhythm_class in RHYTHM_CLASSES}
    plain_durations = {0.5, 1.0}
    for raw_duration, raw_weight in duration_weights.items():
        duration = _quantize(float(raw_duration), 0.25)
        weight = float(raw_weight)
        units = round(duration / 0.25)
        if duration < 0.5 - 1e-6:
            weights["sixteenth"] += weight * duration
        elif units % 2 == 1 and units > 1:
            weights["dotted"] += weight * duration
        elif duration >= 2.0 - 1e-6:
            weights["long"] += weight
        elif duration in plain_durations:
            weights["plain"] += weight * duration
    weights["syncopated"] = max(weights["syncopated"], weights["dotted"], weights["plain"] * 0.08)
    return weights


def _candidate_class_weights(
    candidates_by_class: dict[str, list[StyleCell]],
    rhythm_class_weights: dict[str, float],
) -> dict[object, float]:
    available = {
        rhythm_class: max(rhythm_class_weights.get(rhythm_class, 0.0), 0.0)
        for rhythm_class in candidates_by_class
    }
    if not available:
        return {}
    average = sum(available.values()) / max(1, len(available))
    floor = max(1.0, average * 0.18)
    return {rhythm_class: max(weight, floor) for rhythm_class, weight in available.items()}


def _quantize(value: float, grid: float) -> float:
    return round(math.floor((value / grid) + 0.5) * grid, 6)


def _blend_weights(
    primary: dict[int | float, float],
    secondary: dict[int | float, float],
) -> dict[int | float, float]:
    keys = set(primary) | set(secondary)
    return {
        key: math.sqrt(max(primary.get(key, 0.0001), 0.0001) * max(secondary.get(key, 0.0001), 0.0001))
        for key in keys
    }


def _negative_log_probability(value: int | float, weights: dict[int | float, float]) -> float:
    total = sum(max(weight, 0.0001) for weight in weights.values())
    probability = max(weights.get(value, 0.0001), 0.0001) / max(total, 0.0001)
    return -math.log(probability)
