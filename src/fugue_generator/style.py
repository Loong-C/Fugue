from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


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


@dataclass
class CorpusStyleModel:
    interval_weights: dict[int, float]
    duration_weights: dict[float, float]
    interval_transitions: dict[int, dict[int, float]] = field(default_factory=dict)
    duration_transitions: dict[float, dict[float, float]] = field(default_factory=dict)
    source: str = "built-in"

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
            if "interval_transitions" not in data or "duration_transitions" not in data:
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
                    "interval_weights": model.interval_weights,
                    "duration_weights": model.duration_weights,
                    "interval_transitions": model.interval_transitions,
                    "duration_transitions": model.duration_transitions,
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
        for split in ("train", "valid", "test"):
            for chorale in data.get(split, []):
                if not chorale:
                    continue
                voice_count = len(chorale[0])
                for voice in range(voice_count):
                    note_runs = _runs_from_frames([frame[voice] for frame in chorale], frame_quarters=0.25)
                    _update_melodic_counters(
                        note_runs,
                        interval_counter,
                        duration_counter,
                        interval_transition_counter,
                        duration_transition_counter,
                    )
        return cls(
            dict(interval_counter),
            dict(duration_counter),
            _counter_map_to_dict(interval_transition_counter),
            _counter_map_to_dict(duration_transition_counter),
            f"JSB chorales Markov profile: {path}",
        )

    def augment_from_humdrum_fugues(self, fugue_dir: Path, max_scores: int = 48) -> None:
        from music21 import converter, note

        interval_counter: Counter[int] = Counter(self.interval_weights)
        duration_counter: Counter[float] = Counter(self.duration_weights)
        interval_transition_counter = _dict_to_counter_map(self.interval_transitions)
        duration_transition_counter = _dict_to_counter_map(self.duration_transitions)

        parsed = 0
        for path in sorted(fugue_dir.glob("*.krn"))[:max_scores]:
            try:
                score = converter.parse(path)
            except Exception:
                continue
            parsed += 1
            for part in score.parts:
                note_runs: list[tuple[int, float]] = []
                for element in part.flatten().notesAndRests:
                    if isinstance(element, note.Note):
                        note_runs.append(
                            (
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
                )

        if parsed:
            self.interval_weights = dict(interval_counter)
            self.duration_weights = dict(duration_counter)
            self.interval_transitions = _counter_map_to_dict(interval_transition_counter)
            self.duration_transitions = _counter_map_to_dict(duration_transition_counter)
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
    ) -> float:
        weights = self.duration_weights
        if previous_duration is not None:
            weights = self.duration_transitions.get(previous_duration, weights)
        allowed = {
            duration: weight
            for duration, weight in weights.items()
            if duration <= remaining + 1e-6
        }
        if not allowed:
            return max(0.25, remaining)
        value = float(_weighted_sample(allowed, rng, temperature))
        return min(value, remaining)

    def style_penalty(self, pitches: list[int], durations: list[float]) -> float:
        """Negative log-likelihood style cost for a melodic fragment."""
        penalty = 0.0
        previous_interval = None
        for a, b in zip(pitches, pitches[1:]):
            interval = max(-12, min(12, int(b - a)))
            weights = (
                self.interval_transitions.get(previous_interval, self.interval_weights)
                if previous_interval is not None
                else self.interval_weights
            )
            penalty += _negative_log_probability(interval, weights)
            previous_interval = interval

        previous_duration = None
        for duration in durations:
            duration = max(0.25, round(duration * 4) / 4)
            weights = (
                self.duration_transitions.get(previous_duration, self.duration_weights)
                if previous_duration is not None
                else self.duration_weights
            )
            penalty += _negative_log_probability(duration, weights)
            previous_duration = duration
        return penalty


def _weighted_sample(
    weights: dict[int | float, float],
    rng: random.Random,
    temperature: float,
) -> int | float:
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
    if not frames:
        return []
    runs: list[tuple[int, float]] = []
    previous = frames[0]
    run = 1
    for pitch in frames[1:]:
        if pitch == previous:
            run += 1
            continue
        if previous >= 0:
            runs.append((int(previous), max(0.25, round(run * frame_quarters * 4) / 4)))
        previous = pitch
        run = 1
    if previous >= 0:
        runs.append((int(previous), max(0.25, round(run * frame_quarters * 4) / 4)))
    return runs


def _update_melodic_counters(
    note_runs: list[tuple[int, float]],
    interval_counter: Counter[int],
    duration_counter: Counter[float],
    interval_transition_counter: dict[int, Counter[int]],
    duration_transition_counter: dict[float, Counter[float]],
) -> None:
    if not note_runs:
        return
    pitches = [pitch for pitch, _ in note_runs]
    durations = [duration for _, duration in note_runs]

    for duration in durations:
        duration_counter[duration] += 1
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


def _counter_map_to_dict(counter_map: dict[int | float, Counter[int | float]]) -> dict:
    return {key: dict(counter) for key, counter in counter_map.items()}


def _dict_to_counter_map(values: dict[int | float, dict[int | float, float]]) -> dict:
    return {key: Counter(value) for key, value in values.items()}


def _negative_log_probability(value: int | float, weights: dict[int | float, float]) -> float:
    total = sum(max(weight, 0.0001) for weight in weights.values())
    probability = max(weights.get(value, 0.0001), 0.0001) / max(total, 0.0001)
    return -math.log(probability)
