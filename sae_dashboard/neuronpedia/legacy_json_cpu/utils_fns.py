from __future__ import annotations

from typing import Literal

import einops
import torch
from jaxtyping import Float
from torch import Tensor

from sae_dashboard.components import LogitsHistogramData


class RollingCorrCoef:
    def __init__(
        self,
        indices: list[int] | None = None,
        with_self: bool = False,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.n = 0
        self.X = None
        self.Y = None
        self.indices = indices
        self.with_self = with_self
        self.dtype = dtype
        self.device = device

    def update(self, x: Float[Tensor, "X N"], y: Float[Tensor, "Y N"]) -> None:
        assert x.ndim == 2 and y.ndim == 2, "Both x and y should be 2D"
        X, Nx = x.shape
        Y, Ny = y.shape
        assert Nx == Ny, "Error: x and y should have the same size in the last dimension"
        if self.with_self:
            assert X == Y, "If with_self is True, then x and y should be the same shape"
        if self.X is not None:
            assert X == self.X, "Error: updating a corrcoef object with different sized dataset."
        if self.Y is not None:
            assert Y == self.Y, "Error: updating a corrcoef object with different sized dataset."
        self.X = X
        self.Y = Y

        x = x.to(dtype=self.dtype, device=self.device)
        y = y.to(dtype=self.dtype, device=self.device)

        if self.n == 0:
            self.x_sum = torch.zeros(X, device=x.device, dtype=self.dtype)
            self.xy_sum = torch.zeros(X, Y, device=x.device, dtype=self.dtype)
            self.x2_sum = torch.zeros(X, device=x.device, dtype=self.dtype)
            if not self.with_self:
                self.y_sum = torch.zeros(Y, device=y.device, dtype=self.dtype)
                self.y2_sum = torch.zeros(Y, device=y.device, dtype=self.dtype)

        self.n += x.shape[-1]
        self.x_sum += einops.reduce(x, "X N -> X", "sum")
        self.xy_sum += einops.einsum(x, y, "X N, Y N -> X Y")
        self.x2_sum += einops.reduce(x**2, "X N -> X", "sum")
        if not self.with_self:
            self.y_sum += einops.reduce(y, "Y N -> Y", "sum")
            self.y2_sum += einops.reduce(y**2, "Y N -> Y", "sum")


def detached_legacy_tick_values(
    max_value: float,
    min_value: float,
    tickmode: Literal["ints", "5 ticks"],
) -> list[float]:
    assert tickmode in ["ints", "5 ticks"]
    if tickmode == "ints":
        top_tickval = int(max_value)
        return torch.arange(0, top_tickval + 1, 1).tolist()

    if max_value > -min_value:
        tickrange = 0.1 * int(1e-4 + max_value / (3 * 0.1)) + 1e-6
        num_positive_ticks = 3
        num_negative_ticks = int(-min_value / tickrange)
    else:
        tickrange = 0.1 * int(1e-4 + -min_value / (3 * 0.1)) + 1e-6
        num_negative_ticks = 3
        num_positive_ticks = int(max_value / tickrange)

    tick_vals = [round(-tickrange * i, 1) for i in range(num_negative_ticks, 0, -1)]
    tick_vals.append(0)
    tick_vals.extend(round(tickrange * i, 1) for i in range(1, 1 + num_positive_ticks))
    return tick_vals


def logits_histogram_from_data(
    *,
    data: Tensor,
    n_bins: int,
    tickmode: Literal["ints", "5 ticks"],
    title: str | None,
    compatibility: str,
) -> LogitsHistogramData:
    if compatibility != "detached_legacy":
        return LogitsHistogramData.from_data(
            data=data,
            n_bins=n_bins,
            tickmode=tickmode,
            title=title,
        )

    if data.numel() == 0:
        return LogitsHistogramData()

    max_value = data.max().item()
    min_value = data.min().item()
    bin_size = (max_value - min_value) / n_bins
    bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
    bar_heights = torch.histc(data, bins=n_bins).int().tolist()
    bar_values = [round(x, 5) for x in (bin_edges[:-1] + bin_size / 2).tolist()]
    tick_vals = detached_legacy_tick_values(max_value, min_value, tickmode)

    return LogitsHistogramData(
        bar_heights=bar_heights,
        bar_values=bar_values,
        tick_vals=tick_vals,
        title=title,
    )