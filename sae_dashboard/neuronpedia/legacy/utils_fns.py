from __future__ import annotations

from typing import Literal

import einops
import torch
from eindex import eindex
from jaxtyping import Float
from torch import Tensor

from sae_dashboard.components import LogitsHistogramData
from sae_dashboard.utils_fns import TopK


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
        self.device = torch.device(device)
        self._x_buf: Tensor | None = None
        self._y_buf: Tensor | None = None

    def _ensure_cpu_buffer(self, rows: int, cols: int, existing: Tensor | None) -> Tensor:
        if (
            existing is None
            or existing.shape[0] != rows
            or existing.shape[1] < cols
            or existing.dtype != self.dtype
            or existing.device != self.device
        ):
            return torch.empty(
                (rows, cols),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.device.type == "cpu" and torch.cuda.is_available(),
            )
        return existing

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

        # Benchmark-consistency path for the legacy dashboard's default CPU accumulator.
        # Keep the original synchronous DtoH semantics while reusing host buffers.
        # x = x.to(dtype=self.dtype, device=self.device)
        # y = y.to(dtype=self.dtype, device=self.device)
        if self.device.type == "cpu":
            x_src = x
            y_src = y
            self._x_buf = self._ensure_cpu_buffer(X, Nx, self._x_buf)
            x = self._x_buf[:, :Nx].copy_(x_src)
            if self.with_self:
                y = x
            else:
                self._y_buf = self._ensure_cpu_buffer(Y, Ny, self._y_buf)
                y = self._y_buf[:, :Ny].copy_(y_src)
        else:
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

    def corrcoef(
        self,
    ) -> tuple[Float[Tensor, "X Y"], Float[Tensor, "X Y"]]:
        if self.with_self:
            self.y_sum = self.x_sum
            self.y2_sum = self.x2_sum

        cossim_numer = self.xy_sum
        cossim_denom = torch.sqrt(torch.outer(self.x2_sum, self.y2_sum)) + 1e-6
        cossim = cossim_numer / cossim_denom

        pearson_numer = self.n * self.xy_sum - torch.outer(self.x_sum, self.y_sum)
        pearson_denom = (
            torch.sqrt(
                torch.outer(
                    self.n * self.x2_sum - self.x_sum**2,
                    self.n * self.y2_sum - self.y_sum**2,
                )
            )
            + 1e-6
        )
        pearson = pearson_numer / pearson_denom

        if self.with_self:
            d = cossim.shape[0]
            cossim[range(d), range(d)] = 0.0
            pearson[range(d), range(d)] = 0.0

        return pearson, cossim

    def topk_pearson(
        self,
        k: int,
        largest: bool = True,
    ) -> tuple[list[list[int]], list[list[float]], list[list[float]]]:
        pearson, cossim = self.corrcoef()
        pearson_topk = TopK(tensor=pearson, k=k, largest=largest)
        cossim_values = eindex(cossim, pearson_topk.indices, "X [X k]")

        indices = pearson_topk.indices.tolist()
        if self.indices is not None:
            indices = [[self.indices[i] for i in row] for row in indices]

        return indices, pearson_topk.values.tolist(), cossim_values.tolist()


def legacy_tick_values(
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
) -> LogitsHistogramData:
    if data.numel() == 0:
        return LogitsHistogramData()

    max_value = data.max().item()
    min_value = data.min().item()
    bin_size = (max_value - min_value) / n_bins
    bin_edges = torch.linspace(min_value, max_value, n_bins + 1)
    bar_heights = torch.histc(data, bins=n_bins).int().tolist()
    bar_values = [round(x, 5) for x in (bin_edges[:-1] + bin_size / 2).tolist()]
    tick_vals = legacy_tick_values(max_value, min_value, tickmode)

    return LogitsHistogramData(
        bar_heights=bar_heights,
        bar_values=bar_values,
        tick_vals=tick_vals,
        title=title,
    )