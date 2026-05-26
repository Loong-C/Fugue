from pathlib import Path

from fugue_generator.export import write_midi
from fugue_generator.generator import (
    FugueGenerator,
    FugueRequest,
    _directional_scale_snap,
    _merge_sustained_free_repetitions,
)
from fugue_generator.models import MusicalEvent, VoiceLine, VoiceSpec
from fugue_generator.style import (
    DEFAULT_DURATION_WEIGHTS,
    DEFAULT_INTERVAL_WEIGHTS,
    CorpusStyleModel,
)
from fugue_generator.theory import KeyContext


def _theme(path: Path) -> Path:
    path.write_text(
        "C4:0.5 D4:0.5 Eb4:0.5 G4:0.5 F4:0.5 Eb4:0.5 D4:0.5 C4:1.0",
        encoding="utf-8",
    )
    return path


def _test_style() -> CorpusStyleModel:
    return CorpusStyleModel(
        dict(DEFAULT_INTERVAL_WEIGHTS),
        dict(DEFAULT_DURATION_WEIGHTS),
        source="test",
    )


def test_directional_scale_snap_preserves_melodic_direction() -> None:
    key = KeyContext("C", "minor")

    assert _directional_scale_snap(key, 60, 1) > 60
    assert _directional_scale_snap(key, 60, -1) < 60
    assert _directional_scale_snap(key, 60, 0) == 60


def test_sustained_free_repetitions_are_tied() -> None:
    line = VoiceLine(
        VoiceSpec("test", 48, 72, 60),
        [
            MusicalEvent(0.0, 1.0, 60, "free counterpoint"),
            MusicalEvent(1.0, 2.0, 60, "free counterpoint"),
            MusicalEvent(3.0, 0.5, 62, "free counterpoint"),
            MusicalEvent(3.5, 0.25, 62, "free counterpoint"),
        ],
    )

    _merge_sustained_free_repetitions([line])

    assert line.events[0] == MusicalEvent(0.0, 3.0, 60, "free counterpoint")
    assert line.events[1:] == [
        MusicalEvent(3.0, 0.5, 62, "free counterpoint"),
        MusicalEvent(3.5, 0.25, 62, "free counterpoint"),
    ]


def test_generate_three_voice_fugue_has_clean_entries(tmp_path: Path) -> None:
    subject = _theme(tmp_path / "subject.theme")
    generator = FugueGenerator(Path.cwd(), _test_style())

    fugue = generator.generate(
        FugueRequest(
            key="C",
            mode="minor",
            voices=3,
            subject_path=subject,
            seed=17,
            variants=3,
            temperature=0.9,
            measures=28,
        )
    )

    assert len(fugue.voices) == 3
    assert len(fugue.entries) >= 6
    assert fugue.diagnostics.parallel_fifths == 0
    assert fugue.diagnostics.parallel_octaves == 0
    assert fugue.diagnostics.range_violations == 0
    assert fugue.diagnostics.monophonic_overlaps == 0
    assert fugue.diagnostics.rhythmic_grid_violations == 0
    assert fugue.diagnostics.short_note_count == 0
    assert fugue.diagnostics.melody_issues == 0
    assert fugue.diagnostics.free_stagnation_issues == 0
    assert fugue.diagnostics.free_rhythm_issues == 0
    assert fugue.diagnostics.vertical_clusters == 0


def test_generate_four_voice_fugue_writes_midi(tmp_path: Path) -> None:
    subject = _theme(tmp_path / "subject.theme")
    generator = FugueGenerator(Path.cwd(), _test_style())

    fugue = generator.generate(
        FugueRequest(
            key="C",
            mode="minor",
            voices=4,
            subject_path=subject,
            seed=100,
            variants=4,
            temperature=0.8,
            measures=30,
        )
    )
    output = write_midi(fugue, tmp_path / "fugue.mid", KeyContext("C", "minor"))

    assert output.exists()
    assert output.stat().st_size > 0
    assert len(fugue.entries) >= 7
    assert fugue.diagnostics.parallel_fifths == 0
    assert fugue.diagnostics.parallel_octaves == 0
    assert fugue.diagnostics.range_violations == 0
    assert fugue.diagnostics.monophonic_overlaps == 0
    assert fugue.diagnostics.rhythmic_grid_violations == 0
    assert fugue.diagnostics.short_note_count == 0
    assert fugue.diagnostics.melody_issues == 0
    assert fugue.diagnostics.free_stagnation_issues == 0
    assert fugue.diagnostics.free_rhythm_issues == 0
    assert fugue.diagnostics.vertical_clusters == 0
