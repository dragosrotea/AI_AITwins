import pytest

from training.evaluate import binary_metrics


def test_binary_metrics_return_closed_class_precision_recall_and_f1() -> None:
    precision, recall, f1 = binary_metrics([0, 1, 1, 1], [0, 1, 0, 1])

    assert precision == pytest.approx(1.0)
    assert recall == pytest.approx(2 / 3)
    assert f1 == pytest.approx(0.8)


def test_binary_metrics_reject_mismatched_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        binary_metrics([1], [])
    with pytest.raises(ValueError, match="at least one label"):
        binary_metrics([], [])
