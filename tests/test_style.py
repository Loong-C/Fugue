from fugue_generator.style import CorpusStyleModel


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

