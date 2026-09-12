from __future__ import annotations

from typing import Any, Optional

import torch

try:
    from family import FamilyModules, FamilySpec, UnsupportedFamilyError
except ImportError:
    from trainer.family import FamilyModules, FamilySpec, UnsupportedFamilyError

_MSG = "SD 3.5 LoRA training is not implemented yet"


class Sd35Family:
    def __init__(self, spec: FamilySpec):
        self.spec = spec
        self.trainable = False

    def _raise(self) -> None:
        raise UnsupportedFamilyError(_MSG)

    def load_pipeline(self, path: str, dtype: torch.dtype) -> Any:
        self._raise()

    def unpack(self, pipe: Any) -> FamilyModules:
        self._raise()

    def apply_lora(self, cfg: Any, modules: FamilyModules) -> FamilyModules:
        self._raise()

    def load_lora(self, cfg: Any, modules: FamilyModules) -> dict[str, Any]:
        if not str(getattr(cfg, "resume_lora_path", "") or "").strip():
            return {}
        self._raise()

    def build_noise_scheduler(self, pipe: Any, cfg: Any) -> Any:
        self._raise()

    def extra_cond(
        self,
        *,
        src_wh: tuple[int, int],
        bucket_wh: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]:
        self._raise()

    def encode_prompts(
        self,
        prompts: list[str],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any:
        self._raise()

    def denoise_loss(
        self,
        *,
        latents: torch.Tensor,
        encoded: Any,
        extra: dict[str, torch.Tensor],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self._raise()

    def compute_loss(
        self,
        *,
        prompts: list[str],
        latents: torch.Tensor,
        extra: dict[str, torch.Tensor],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        self._raise()

    def prediction_type(self, cfg: Any) -> str:
        self._raise()

    def save_lora(
        self,
        accelerator: Any,
        modules: FamilyModules,
        cfg: Any,
        global_step: int,
        epoch: Optional[int] = None,
        final: bool = False,
    ) -> None:
        self._raise()

    def generate_sample(
        self,
        *,
        accelerator: Any,
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
        global_step: int,
        output_dir_base: Any,
        swap_ctx: Any = None,
    ) -> None:
        self._raise()
