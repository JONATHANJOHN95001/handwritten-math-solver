"""The time limit must really stop the work: a solve left running burns the CPU until the app closes."""
import threading
import time

import pytest

from results import time_limited


def test_gives_back_the_value():
    assert time_limited(lambda: 5, 5) == 5


def test_gives_back_the_error():
    def work():
        raise ValueError("no")
    with pytest.raises(ValueError, match="no"):
        time_limited(work, 5)


def test_stops_the_work_after_the_limit():
    ticks = []
    started = threading.active_count()

    def work():
        while True:
            try:
                ticks.append(1)
                time.sleep(0.01)
            except Exception:  # SymPy swallows ordinary exceptions in places, so the stop must get past this
                pass

    with pytest.raises(TimeoutError):
        time_limited(work, 0.2)
    time.sleep(0.5)
    seen = len(ticks)
    time.sleep(0.3)
    assert len(ticks) == seen, "the work was still running after the time limit"
    assert threading.active_count() == started
