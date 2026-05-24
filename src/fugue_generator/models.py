from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class MusicalEvent:
    """A note or rest in quarter-note units."""

    offset: float
    duration: float
    pitch: int | None
    label: str = ""

    @property
    def end(self) -> float:
        return self.offset + self.duration

    def shifted(self, offset: float = 0.0, pitch: int = 0, label: str | None = None) -> "MusicalEvent":
        return MusicalEvent(
            offset=self.offset + offset,
            duration=self.duration,
            pitch=None if self.pitch is None else self.pitch + pitch,
            label=self.label if label is None else label,
        )


@dataclass
class NoteSequence:
    events: list[MusicalEvent]
    name: str = "sequence"

    @property
    def duration(self) -> float:
        if not self.events:
            return 0.0
        return max(event.end for event in self.events)

    @property
    def pitches(self) -> list[int]:
        return [event.pitch for event in self.events if event.pitch is not None]

    def with_label(self, label: str) -> "NoteSequence":
        return NoteSequence(
            [
                MusicalEvent(event.offset, event.duration, event.pitch, label)
                for event in self.events
            ],
            name=self.name,
        )

    def quantized(self, grid: float = 0.25) -> "NoteSequence":
        return NoteSequence(
            [
                MusicalEvent(
                    round(event.offset / grid) * grid,
                    max(grid, round(event.duration / grid) * grid),
                    event.pitch,
                    event.label,
                )
                for event in self.events
            ],
            name=self.name,
        )


@dataclass(frozen=True)
class VoiceSpec:
    name: str
    low: int
    high: int
    center: int


@dataclass(frozen=True)
class EntryPlan:
    voice_index: int
    start: float
    kind: str
    key_name: str
    label: str


@dataclass(frozen=True)
class HarmonySection:
    start: float
    end: float
    key_name: str
    progression: tuple[int, ...]
    label: str


@dataclass
class VoiceLine:
    spec: VoiceSpec
    events: list[MusicalEvent] = field(default_factory=list)

    def add(self, event: MusicalEvent) -> None:
        self.events.append(event)
        self.events.sort(key=lambda item: (item.offset, item.duration))

    def add_many(self, events: Iterable[MusicalEvent]) -> None:
        self.events.extend(events)
        self.events.sort(key=lambda item: (item.offset, item.duration))

    @property
    def end(self) -> float:
        if not self.events:
            return 0.0
        return max(event.end for event in self.events)

    def active_pitch(self, t: float) -> int | None:
        for event in self.events:
            if event.pitch is not None and event.offset <= t < event.end - 1e-6:
                return event.pitch
        return None

    def previous_pitch(self, t: float) -> int | None:
        candidates = [event for event in self.events if event.pitch is not None and event.end <= t + 1e-6]
        if not candidates:
            return None
        return max(candidates, key=lambda event: event.end).pitch

    def next_pitch(self, t: float) -> int | None:
        candidates = [event for event in self.events if event.pitch is not None and event.offset >= t - 1e-6]
        if not candidates:
            return None
        return min(candidates, key=lambda event: event.offset).pitch

    def next_event_start(self, t: float) -> float | None:
        starts = [event.offset for event in self.events if event.offset >= t - 1e-6]
        return min(starts) if starts else None

    def overlaps(self, event: MusicalEvent) -> bool:
        return any(
            event.offset < other.end - 1e-6 and event.end > other.offset + 1e-6
            for other in self.events
        )

    def free_spans(self, total_end: float) -> list[tuple[float, float]]:
        spans: list[tuple[float, float]] = []
        cursor = 0.0
        for event in sorted(self.events, key=lambda item: item.offset):
            if event.offset > cursor + 1e-6:
                spans.append((cursor, event.offset))
            cursor = max(cursor, event.end)
        if cursor < total_end - 1e-6:
            spans.append((cursor, total_end))
        return spans


@dataclass
class FugueDiagnostics:
    score: float
    entries: list[EntryPlan]
    parallel_fifths: int
    parallel_octaves: int
    voice_crossings: int
    range_violations: int
    strong_dissonances: int
    monophonic_overlaps: int
    rhythmic_grid_violations: int
    short_note_count: int
    melody_issues: int
    vertical_clusters: int
    total_duration: float
    seed: int
    style_source: str
    output_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "score": self.score,
            "entries": [
                {
                    "voice": entry.voice_index,
                    "start": entry.start,
                    "kind": entry.kind,
                    "key": entry.key_name,
                    "label": entry.label,
                }
                for entry in self.entries
            ],
            "parallel_fifths": self.parallel_fifths,
            "parallel_octaves": self.parallel_octaves,
            "voice_crossings": self.voice_crossings,
            "range_violations": self.range_violations,
            "strong_dissonances": self.strong_dissonances,
            "monophonic_overlaps": self.monophonic_overlaps,
            "rhythmic_grid_violations": self.rhythmic_grid_violations,
            "short_note_count": self.short_note_count,
            "melody_issues": self.melody_issues,
            "vertical_clusters": self.vertical_clusters,
            "total_duration": self.total_duration,
            "seed": self.seed,
            "style_source": self.style_source,
            "output_path": None if self.output_path is None else str(self.output_path),
        }


@dataclass
class GeneratedFugue:
    voices: list[VoiceLine]
    entries: list[EntryPlan]
    harmony: list[HarmonySection]
    diagnostics: FugueDiagnostics
