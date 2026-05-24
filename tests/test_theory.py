from fugue_generator.models import MusicalEvent, NoteSequence
from fugue_generator.theory import KeyContext, tonal_answer, voice_specs


def test_tonal_answer_stays_in_requested_key() -> None:
    key = KeyContext("C", "minor")
    subject = NoteSequence(
        [
            MusicalEvent(0.0, 1.0, 60, "subject"),
            MusicalEvent(1.0, 1.0, 62, "subject"),
            MusicalEvent(2.0, 1.0, 67, "subject"),
        ],
        "subject",
    )

    answer = tonal_answer(subject, key)

    assert answer.pitches[0] % 12 == 7
    assert all(key.snap_to_scale(pitch) == pitch for pitch in answer.pitches)


def test_voice_specs_support_three_and_four_voices() -> None:
    assert len(voice_specs(3)) == 3
    assert len(voice_specs(4)) == 4

