from so101_aimbot.perception.select import select_nearest_target
from so101_aimbot.types import CursorState, Target


def test_select_nearest_target() -> None:
    targets = [Target(0, 0), Target(10, 0), Target(100, 100)]
    cursor = CursorState(9, 1)
    assert select_nearest_target(targets, cursor) == Target(10, 0)


def test_select_empty() -> None:
    assert select_nearest_target([], CursorState(0, 0)) is None
