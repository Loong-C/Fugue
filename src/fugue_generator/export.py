from __future__ import annotations

from pathlib import Path

from music21 import instrument, key, metadata, meter, note, stream, tempo

from .models import GeneratedFugue, MusicalEvent, VoiceLine
from .theory import KeyContext


def write_midi(fugue: GeneratedFugue, path: str | Path, key_context: KeyContext, bpm: int = 92) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    score = to_music21_score(fugue, key_context, bpm)
    score.write("midi", fp=str(target))
    return target


def write_musicxml(fugue: GeneratedFugue, path: str | Path, key_context: KeyContext, bpm: int = 92) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    score = to_music21_score(fugue, key_context, bpm)
    score.write("musicxml", fp=str(target))
    return target


def to_music21_score(fugue: GeneratedFugue, key_context: KeyContext, bpm: int = 92) -> stream.Score:
    score = stream.Score()
    score.metadata = metadata.Metadata()
    score.metadata.title = "Generated Fugue"
    score.insert(0, tempo.MetronomeMark(number=bpm))
    score.insert(0, meter.TimeSignature("4/4"))
    score.insert(0, _key_signature(key_context))

    for index, line in enumerate(fugue.voices):
        part = _line_to_part(line, index)
        score.insert(0, part)
    return score


def _key_signature(key_context: KeyContext) -> key.KeySignature:
    try:
        mode = "major" if key_context.mode.lower() in {"major", "ionian", "lydian", "mixolydian"} else "minor"
        return key.KeySignature(key.Key(key_context.tonic, mode).sharps)
    except Exception:
        return key.KeySignature(0)


def _line_to_part(line: VoiceLine, index: int) -> stream.Part:
    part = stream.Part(id=line.spec.name)
    piano = instrument.Piano()
    piano.midiChannel = index
    part.insert(0, piano)
    cursor = 0.0
    for event in sorted(line.events, key=lambda item: item.offset):
        if event.offset > cursor + 1e-6:
            part.append(note.Rest(quarterLength=event.offset - cursor))
            cursor = event.offset
        if event.end <= cursor + 1e-6:
            continue
        duration = event.duration
        part.append(_event_to_music21(event, duration))
        cursor = event.end
    return part


def _event_to_music21(event: MusicalEvent, duration: float):
    if event.pitch is None:
        return note.Rest(quarterLength=duration)
    item = note.Note(midi=event.pitch, quarterLength=duration)
    if event.label:
        item.lyric = event.label
    return item
