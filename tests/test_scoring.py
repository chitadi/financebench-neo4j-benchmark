from fbneo.scoring import heuristic_answer_correct


def test_heuristic_answer_correct_numeric() -> None:
    score, _reason = heuristic_answer_correct("$8.70", "The answer is $8.70 billion.")
    assert score == 1.0

