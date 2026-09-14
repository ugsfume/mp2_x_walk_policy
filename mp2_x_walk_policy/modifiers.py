"""Observation-path modifiers.

The robot's ~100 ms command-to-observation round trip is only partly modelled
on the action path (``DelayedDCMotor``, 50-90 ms). The remainder sits on the
sensing side and its split is not identifiable, so the observation share is
randomised per environment per episode.

``DigitalFilterCfg`` could express a fixed delay but zero-fills on reset -- on
``projected_gravity`` that feeds the policy a zero gravity vector for the first
N steps of every episode. This modifier wraps ``DelayBuffer`` instead, which
returns the freshest sample while the buffer is still filling.

Delay resolution is one POLICY step (20 ms on the 50 Hz tasks): modifiers run
once per ``observation_manager.compute()``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.utils.buffers import DelayBuffer
from isaaclab.utils.configclass import configclass
from isaaclab.utils.modifiers import ModifierBase, ModifierCfg


class RandomizedObservationDelay(ModifierBase):
    """Per-env observation delay resampled uniformly in [min, max] steps at reset.

    Uses the global torch RNG, so the lags are deterministic given the
    training seed. Same fill behaviour as above: freshest sample while
    filling after a reset, never zero-fill.
    """

    def __init__(self, cfg: ModifierCfg, data_dim: tuple[int, ...], device: str) -> None:
        super().__init__(cfg, data_dim, device)
        self._min = int(cfg.params["min_steps"])
        self._max = int(cfg.params["max_steps"])
        if not 0 <= self._min <= self._max:
            raise ValueError(f"bad delay range [{self._min}, {self._max}]")
        self._num_envs = data_dim[0]
        self._buffer = DelayBuffer(self._max, batch_size=self._num_envs, device=device)
        self._buffer.set_time_lag(self._sample(self._num_envs))

    def _sample(self, n: int) -> torch.Tensor:
        # DelayBuffer._time_lags is int32; randint defaults to int64 and the
        # per-env index-put refuses mixed dtypes.
        return torch.randint(
            self._min, self._max + 1, (n,), device=self._device, dtype=torch.int
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        self._buffer.reset(env_ids)
        if env_ids is None:
            self._buffer.set_time_lag(self._sample(self._num_envs))
        else:
            ids = torch.as_tensor(list(env_ids), dtype=torch.long, device=self._device) \
                if not torch.is_tensor(env_ids) else env_ids
            self._buffer.set_time_lag(self._sample(len(ids)), ids)

    def __call__(self, data: torch.Tensor, min_steps: int = 0, max_steps: int = 0) -> torch.Tensor:
        # params re-arrive as kwargs from the manager; consumed in __init__.
        return self._buffer.compute(data)


@configclass
class RandomizedObservationDelayCfg(ModifierCfg):
    func: type[RandomizedObservationDelay] = RandomizedObservationDelay
