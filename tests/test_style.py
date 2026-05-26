from collections import Counter
import random

from fugue_generator.style import CorpusStyleModel
from fugue_generator.style import StyleCell, _top_style_cells, _update_cell_counter


def test_style_penalty_prefers_seen_patterns() -> None:
    model = CorpusStyleModel(
        interval_weights={1: 10, 7: 1},
        duration_weights={0.5: 10, 2.0: 1},
        interval_transitions={1: {1: 10, 7: 1}},
        duration_transitions={0.5: {0.5: 10, 2.0: 1}},
        source="test",
    )

    familiar = model.style_penalty([60, 61, 62], [0.5, 0.5, 0.5])
    unfamiliar = model.style_penalty([60, 67, 74], [2.0, 2.0, 2.0])

    assert familiar < unfamiliar


def test_style_cells_preserve_quarter_grid_rhythm_classes() -> None:
    note_runs = [
        (0.0, 60, 0.5),
        (0.5, 62, 0.75),
        (1.25, 64, 0.5),
        (1.75, 65, 0.5),
        (2.25, 67, 0.25),
        (2.5, 69, 0.25),
        (2.75, 71, 0.5),
        (3.25, 72, 0.75),
        (4.0, 74, 2.0),
        (6.0, 76, 0.5),
    ]
    cell_counter = Counter()

    _update_cell_counter(note_runs, cell_counter)
    cells = _top_style_cells(cell_counter, limit=64)

    assert any(0.25 in cell.durations for cell in cells)
    assert any(0.75 in cell.durations for cell in cells)
    assert any(2.0 in cell.durations for cell in cells)
    assert {"sixteenth", "dotted", "long"} <= {cell.rhythm_class for cell in cells}


def test_sample_cell_can_return_learned_sixteenth_cell() -> None:
    model = CorpusStyleModel(
        interval_weights={1: 10},
        duration_weights={0.25: 10, 0.5: 8},
        melodic_cells=[
            StyleCell((0.25, 0.25, 0.5), (1, 1, -1), 0.0, 4.0, "sixteenth"),
        ],
        rhythm_class_weights={"sixteenth": 4.0},
        source="test",
    )

    cell = model.sample_cell(random.Random(7), remaining=1.0, phase=0.0)

    assert cell.durations == (0.25, 0.25, 0.5)
    assert cell.rhythm_class == "sixteenth"
