from __future__ import annotations

from dataclasses import dataclass

from .models import NoteSequence, MusicalEvent, VoiceSpec


TONIC_TO_PC = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

PC_TO_NAME_SHARP = {
    0: "C",
    1: "C#",
    2: "D",
    3: "Eb",
    4: "E",
    5: "F",
    6: "F#",
    7: "G",
    8: "Ab",
    9: "A",
    10: "Bb",
    11: "B",
}

MODE_INTERVALS = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "natural_minor": (0, 2, 3, 5, 7, 8, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
}

VOICE_RANGES_4 = (
    VoiceSpec("soprano", 60, 81, 72),
    VoiceSpec("alto", 55, 74, 64),
    VoiceSpec("tenor", 48, 67, 57),
    VoiceSpec("bass", 36, 60, 48),
)

VOICE_RANGES_3 = (
    VoiceSpec("upper", 60, 79, 70),
    VoiceSpec("middle", 52, 72, 62),
    VoiceSpec("lower", 40, 62, 50),
)


@dataclass(frozen=True)
class KeyContext:
    tonic: str
    mode: str = "minor"

    @property
    def pc(self) -> int:
        normalized = normalize_tonic(self.tonic)
        return TONIC_TO_PC[normalized]

    @property
    def intervals(self) -> tuple[int, ...]:
        return MODE_INTERVALS.get(self.mode.lower(), MODE_INTERVALS["minor"])

    @property
    def name(self) -> str:
        return f"{normalize_tonic(self.tonic)} {self.mode.lower()}"

    def pitch_from_diatonic_index(self, index: int, base_octave: int = 4) -> int:
        octave, degree = divmod(index, 7)
        return 12 * (base_octave + 1 + octave) + self.pc + self.intervals[degree]

    def diatonic_index_for_pitch(self, pitch: int) -> int:
        reference = 12 * 5 + self.pc
        rel = pitch - reference
        best_index = 0
        best_distance = 999
        for octave in range(-5, 6):
            for degree, interval in enumerate(self.intervals):
                candidate = 12 * octave + interval
                distance = abs(rel - candidate)
                if distance < best_distance:
                    best_distance = distance
                    best_index = octave * 7 + degree
        return best_index

    def snap_to_scale(self, pitch: int) -> int:
        candidates = []
        for octave in range((pitch // 12) - 1, (pitch // 12) + 2):
            base = 12 * octave + self.pc
            candidates.extend(base + interval for interval in self.intervals)
        return min(candidates, key=lambda candidate: (abs(candidate - pitch), candidate))

    def chord_pitches(self, degree: int, low: int, high: int, include_seventh: bool = False) -> list[int]:
        degree_index = degree - 1
        chord_degrees = [degree_index, degree_index + 2, degree_index + 4]
        if include_seventh:
            chord_degrees.append(degree_index + 6)
        pitches = []
        for octave in range(-3, 5):
            for idx in chord_degrees:
                octave_shift, scale_idx = divmod(idx, 7)
                pitch = 12 * (octave + octave_shift + 5) + self.pc + self.intervals[scale_idx]
                if low <= pitch <= high:
                    pitches.append(pitch)
        return sorted(set(pitches))

    def related(self, degree: int, mode: str | None = None) -> "KeyContext":
        pitch = self.pitch_from_diatonic_index(degree - 1)
        return KeyContext(PC_TO_NAME_SHARP[pitch % 12], mode or self.mode)


def normalize_tonic(value: str) -> str:
    value = value.strip()
    if not value:
        raise ValueError("Key tonic cannot be empty.")
    first = value[0].upper()
    rest = value[1:].replace("♯", "#").replace("♭", "b")
    tonic = first + rest
    if tonic not in TONIC_TO_PC:
        raise ValueError(f"Unsupported tonic {value!r}.")
    return tonic


def voice_specs(count: int) -> tuple[VoiceSpec, ...]:
    if count == 3:
        return VOICE_RANGES_3
    if count == 4:
        return VOICE_RANGES_4
    raise ValueError("Only three- and four-voice fugues are supported.")


def fit_sequence_to_voice(sequence: NoteSequence, spec: VoiceSpec) -> NoteSequence:
    if not sequence.pitches:
        return sequence
    best_shift = 0
    best_penalty = 10**9
    for shift in range(-36, 37, 12):
        shifted = [pitch + shift for pitch in sequence.pitches]
        outside = sum(max(0, spec.low - pitch, pitch - spec.high) for pitch in shifted)
        center_penalty = abs((sum(shifted) / len(shifted)) - spec.center)
        penalty = outside * 100 + center_penalty
        if penalty < best_penalty:
            best_penalty = penalty
            best_shift = shift
    return NoteSequence(
        [
            MusicalEvent(
                event.offset,
                event.duration,
                None if event.pitch is None else event.pitch + best_shift,
                event.label,
            )
            for event in sequence.events
        ],
        name=sequence.name,
    )


def transpose_sequence_to_key(sequence: NoteSequence, source: KeyContext, target: KeyContext) -> NoteSequence:
    events: list[MusicalEvent] = []
    for event in sequence.events:
        if event.pitch is None:
            events.append(event)
            continue
        diatonic_index = source.diatonic_index_for_pitch(event.pitch)
        pitch = target.pitch_from_diatonic_index(diatonic_index)
        while pitch - event.pitch > 6:
            pitch -= 12
        while event.pitch - pitch > 6:
            pitch += 12
        events.append(MusicalEvent(event.offset, event.duration, target.snap_to_scale(pitch), event.label))
    return NoteSequence(events, name=sequence.name)


def real_answer(sequence: NoteSequence, key: KeyContext) -> NoteSequence:
    return NoteSequence(
        [
            MusicalEvent(
                event.offset,
                event.duration,
                None if event.pitch is None else key.snap_to_scale(event.pitch + 7),
                "answer",
            )
            for event in sequence.events
        ],
        name=f"{sequence.name} real answer",
    )


def tonal_answer(sequence: NoteSequence, key: KeyContext) -> NoteSequence:
    events: list[MusicalEvent] = []
    for event in sequence.events:
        if event.pitch is None:
            events.append(MusicalEvent(event.offset, event.duration, None, "answer"))
            continue
        source_index = key.diatonic_index_for_pitch(event.pitch)
        answer_index = source_index + 4
        pitch = key.pitch_from_diatonic_index(answer_index)
        while pitch - event.pitch > 10:
            pitch -= 12
        while event.pitch - pitch > 2:
            pitch += 12
        events.append(MusicalEvent(event.offset, event.duration, key.snap_to_scale(pitch), "answer"))
    return NoteSequence(events, name=f"{sequence.name} tonal answer")


def dominant_degree_for_mode(mode: str) -> int:
    return 5


def relative_degree_for_mode(mode: str) -> int:
    return 3 if mode.lower() in {"minor", "natural_minor", "aeolian", "dorian", "phrygian"} else 6


def interval_class(a: int, b: int) -> int:
    return abs(a - b) % 12


def is_perfect_interval(a: int, b: int) -> bool:
    return interval_class(a, b) in {0, 7}
