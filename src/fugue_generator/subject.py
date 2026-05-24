from __future__ import annotations

from pathlib import Path

from music21 import converter, note, stream

from .models import MusicalEvent, NoteSequence


def load_subject(path: str | Path) -> NoteSequence:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Subject file not found: {source}")
    if source.suffix.lower() in {".txt", ".theme"}:
        return _load_text_subject(source)
    return _load_score_subject(source)


def _load_score_subject(path: Path) -> NoteSequence:
    score = converter.parse(path)
    source_stream: stream.Stream
    if score.parts:
        source_stream = max(score.parts, key=lambda part: len(part.recurse().notes))
    else:
        source_stream = score

    raw = []
    for element in source_stream.flatten().notesAndRests:
        if isinstance(element, note.Rest):
            pitch = None
        elif isinstance(element, note.Note):
            pitch = int(element.pitch.midi)
        else:
            pitches = getattr(element, "pitches", [])
            pitch = int(max(p.midi for p in pitches)) if pitches else None
        raw.append((float(element.offset), float(element.quarterLength), pitch))

    first_note_offset = next((offset for offset, _, pitch in raw if pitch is not None), None)
    if first_note_offset is None:
        raise ValueError(f"Subject file has no notes: {path}")

    events = [
        MusicalEvent(
            offset=max(0.0, offset - first_note_offset),
            duration=max(0.25, duration),
            pitch=pitch,
            label="subject",
        )
        for offset, duration, pitch in raw
        if offset >= first_note_offset
    ]
    events = _trim_trailing_rests(events)
    return NoteSequence(events, name=path.stem).quantized(0.25)


def _load_text_subject(path: Path) -> NoteSequence:
    """Load a compact text subject: C4:0.5 D4:0.5 Eb4:1 R:0.5 G4:1."""
    tokens = path.read_text(encoding="utf-8").replace("\n", " ").split()
    offset = 0.0
    events: list[MusicalEvent] = []
    for token in tokens:
        if ":" not in token:
            raise ValueError(f"Bad subject token {token!r}; expected NOTE:DURATION.")
        name, duration_text = token.split(":", 1)
        duration = float(duration_text)
        if name.upper() in {"R", "REST"}:
            pitch = None
        else:
            pitch = int(note.Note(name).pitch.midi)
        events.append(MusicalEvent(offset, duration, pitch, "subject"))
        offset += duration
    if not any(event.pitch is not None for event in events):
        raise ValueError(f"Subject file has no notes: {path}")
    return NoteSequence(_trim_trailing_rests(events), name=path.stem).quantized(0.25)


def _trim_trailing_rests(events: list[MusicalEvent]) -> list[MusicalEvent]:
    trimmed = list(events)
    while trimmed and trimmed[-1].pitch is None:
        trimmed.pop()
    return trimmed


def subject_summary(subject: NoteSequence) -> dict[str, object]:
    pitches = subject.pitches
    return {
        "name": subject.name,
        "duration": subject.duration,
        "notes": len(pitches),
        "low": min(pitches) if pitches else None,
        "high": max(pitches) if pitches else None,
        "ambitus": (max(pitches) - min(pitches)) if pitches else None,
    }

