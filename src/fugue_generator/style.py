from __future__ import annotations

import json
import math
import random
from collections import Counter
from dataclasses import dataclass
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
            return cls(
                {int(k): float(v) for k, v in data["interval_weights"].items()},
                {float(k): float(v) for k, v in data["duration_weights"].items()},
                data.get("source", str(cache)),
            )

        jsb = root / "data" / "raw" / "jsb-chorales-dataset" / "Jsb16thSeparated.json"
        if not jsb.exists():
            return cls(dict(DEFAULT_INTERVAL_WEIGHTS), dict(DEFAULT_DURATION_WEIGHTS), "built-in")

        model = cls.from_jsb_json(jsb)
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(
            json.dumps(
                {
                    "interval_weights": model.interval_weights,
                    "duration_weights": model.duration_weights,
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
        for split in ("train", "valid", "test"):
            for chorale in data.get(split, []):
                if not chorale:
                    continue
                voice_count = len(chorale[0])
                for voice in range(voice_count):
                    previous_pitch = None
                    previous_change_pitch = None
                    run = 0
                    for frame in chorale:
                        pitch = frame[voice]
                        if pitch == previous_pitch:
                            run += 1
                            continue
                        if previous_pitch is not None:
                            duration = max(0.25, min(4.0, round((run / 4.0) * 4) / 4))
                            duration_counter[duration] += 1
                            if previous_change_pitch is not None and previous_pitch >= 0:
                                interval = int(previous_pitch - previous_change_pitch)
                                if -12 <= interval <= 12:
                                    interval_counter[interval] += 1
                        if previous_pitch is not None and previous_pitch >= 0:
                            previous_change_pitch = previous_pitch
                        previous_pitch = pitch
                        run = 1
        return cls(dict(interval_counter), dict(duration_counter), f"JSB chorales: {path}")

    def sample_interval(self, rng: random.Random, temperature: float = 1.0) -> int:
        return int(_weighted_sample(self.interval_weights, rng, temperature))

    def sample_duration(self, rng: random.Random, remaining: float, temperature: float = 1.0) -> float:
        allowed = {
            duration: weight
            for duration, weight in self.duration_weights.items()
            if duration <= remaining + 1e-6
        }
        if not allowed:
            return max(0.25, remaining)
        value = float(_weighted_sample(allowed, rng, temperature))
        return min(value, remaining)


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
