"""Kohya-style epoch-window average loss.

Matches sd-scripts ``LossRecorder``: epoch 0 grows the list, later epochs
overwrite the same intra-epoch index, and ``moving_average`` is the mean of
that window. Used both while training (TensorBoard) and to reconstruct
``Train/Avg_Loss`` from older ``Train/Loss`` series.
"""

from __future__ import annotations

from typing import Any, Optional


class LossRecorder:
    def __init__(self) -> None:
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch <= 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        n = len(self.loss_list)
        if n == 0:
            return 0.0
        return self.loss_total / n


def avg_loss_window(n_points: int, steps_per_epoch: Optional[int] = None) -> int:
    if steps_per_epoch is not None and steps_per_epoch > 0:
        return max(1, int(steps_per_epoch))
    if n_points <= 0:
        return 1
    return min(n_points, 100)


def synthesize_avg_loss(
    loss_points: list[dict[str, Any]],
    steps_per_epoch: Optional[int] = None,
) -> list[dict[str, Any]]:
    """Rebuild a Kohya-like Avg_Loss series from per-step Train/Loss points.

    When ``steps_per_epoch`` is unknown, uses a window of ``min(n, 100)``.
    After the window fills, overwrite-by-index is equivalent to the mean of
    the last ``W`` points.
    """
    if not loss_points:
        return []

    window = avg_loss_window(len(loss_points), steps_per_epoch)
    recorder = LossRecorder()
    out: list[dict[str, Any]] = []
    for i, point in enumerate(loss_points):
        value = float(point["value"])
        if i < window:
            recorder.add(epoch=0, step=i, loss=value)
        else:
            recorder.add(epoch=1, step=i % window, loss=value)
        item: dict[str, Any] = {
            "step": point["step"],
            "value": recorder.moving_average,
        }
        if "wall_time" in point:
            item["wall_time"] = point["wall_time"]
        out.append(item)
    return out
