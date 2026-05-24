from pathlib import Path

from fugue_generator.export import write_midi
from fugue_generator.generator import FugueGenerator, FugueRequest
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
