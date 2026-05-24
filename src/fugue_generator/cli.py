from __future__ import annotations

import typer


app = typer.Typer(help="Research CLI for fugue generation experiments.")


@app.command()
def generate(
    key: str = typer.Option(..., help="Tonic, for example C, F#, or Bb."),
    mode: str = typer.Option("minor", help="Mode, for example major, minor, dorian."),
    voices: int = typer.Option(4, min=3, max=4, help="Number of voices."),
    subject: str = typer.Option(..., help="Path to a MIDI or MusicXML subject file."),
    out: str = typer.Option("out/fugue.mid", help="Output MIDI path."),
) -> None:
    """Placeholder for the future generator."""
    typer.echo(
        "Generator is not implemented yet. "
        f"Received key={key}, mode={mode}, voices={voices}, subject={subject}, out={out}."
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()

