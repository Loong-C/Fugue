from __future__ import annotations

import json
from pathlib import Path

import typer

from .evaluate import with_output_path
from .export import write_midi, write_musicxml
from .generator import FugueGenerator, FugueRequest
from .style import CorpusStyleModel
from .subject import load_subject, subject_summary
from .theory import KeyContext


app = typer.Typer(
    help="Generate short, school-style fugues with a hybrid symbolic/statistical engine.",
    no_args_is_help=True,
)


@app.callback()
def callback() -> None:
    """Fugue generator command group."""


@app.command()
def generate(
    key: str = typer.Option(..., help="Tonic, for example C, F#, or Bb."),
    mode: str = typer.Option("minor", help="Mode, for example major, minor, dorian."),
    voices: int = typer.Option(4, min=3, max=4, help="Number of voices."),
    subject: str = typer.Option(..., help="Path to a MIDI, MusicXML, or .theme subject file."),
    out: str = typer.Option("out/fugue.mid", help="Output MIDI path."),
    report: str | None = typer.Option(None, help="Optional JSON diagnostics path."),
    musicxml: str | None = typer.Option(None, help="Optional MusicXML output path."),
    write_variants_dir: str | None = typer.Option(
        None,
        help="Optional directory to write every generated candidate MIDI.",
    ),
    seed: int = typer.Option(1, help="Random seed. Change it for different versions."),
    temperature: float = typer.Option(1.0, min=0.2, max=3.0, help="Sampling temperature."),
    variants: int = typer.Option(5, min=1, max=32, help="Generate and score this many candidates."),
    measures: int | None = typer.Option(None, min=20, max=80, help="Optional total length in 4/4 measures."),
) -> None:
    """Generate a complete short fugue and write MIDI."""
    root = Path.cwd()
    request = FugueRequest(
        key=key,
        mode=mode,
        voices=voices,
        subject_path=Path(subject),
        seed=seed,
        temperature=temperature,
        variants=variants,
        measures=measures,
    )
    generator = FugueGenerator(root)
    candidates = generator.generate_candidates(request)
    fugue = max(candidates, key=lambda candidate: candidate.diagnostics.score)
    output = write_midi(fugue, out, KeyContext(key, mode))
    fugue.diagnostics = with_output_path(fugue.diagnostics, output)

    if write_variants_dir:
        variants_dir = Path(write_variants_dir)
        variants_dir.mkdir(parents=True, exist_ok=True)
        ranked_candidates = sorted(
            candidates,
            key=lambda item: item.diagnostics.score,
            reverse=True,
        )
        for index, candidate in enumerate(ranked_candidates, 1):
            candidate_path = variants_dir / f"variant_{index:02d}_score_{candidate.diagnostics.score:.0f}.mid"
            write_midi(candidate, candidate_path, KeyContext(key, mode))

    if musicxml:
        write_musicxml(fugue, musicxml, KeyContext(key, mode))
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        report_data = fugue.diagnostics.to_dict()
        ranked_candidates = sorted(candidates, key=lambda item: item.diagnostics.score, reverse=True)
        report_data["candidates"] = [candidate.diagnostics.to_dict() for candidate in ranked_candidates]
        Path(report).write_text(
            json.dumps(report_data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    typer.echo(f"Wrote {output}")
    typer.echo(
        "score={score:.1f} entries={entries} parallel5={p5} parallel8={p8} "
        "crossings={crossings} strong_dissonances={diss}".format(
            score=fugue.diagnostics.score,
            entries=len(fugue.entries),
            p5=fugue.diagnostics.parallel_fifths,
            p8=fugue.diagnostics.parallel_octaves,
            crossings=fugue.diagnostics.voice_crossings,
            diss=fugue.diagnostics.strong_dissonances,
        )
    )


@app.command("inspect-subject")
def inspect_subject(subject: str) -> None:
    """Print a compact summary of a subject file."""
    typer.echo(json.dumps(subject_summary(load_subject(subject)), indent=2))


@app.command("build-style-profile")
def build_style_profile() -> None:
    """Rebuild the corpus Markov profile used by the stochastic generator."""
    model = CorpusStyleModel.load(Path.cwd(), rebuild=True)
    typer.echo(f"Built style profile from {model.source}")
    typer.echo(
        "intervals={intervals} durations={durations} interval_transitions={it} "
        "duration_transitions={dt}".format(
            intervals=len(model.interval_weights),
            durations=len(model.duration_weights),
            it=len(model.interval_transitions),
            dt=len(model.duration_transitions),
        )
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
