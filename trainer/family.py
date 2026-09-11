from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    import torch


class ModelSpecError(ValueError):
    """`[model_spec]` is missing from the catalog or internally inconsistent."""


class UnsupportedFamilyError(ValueError):
    """Catalog family exists but this trainer build cannot train it."""


_SD35_NOT_IMPLEMENTED = "SD 3.5 LoRA training is not implemented yet"


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    base_model_version: str
    architecture: str
    implementation: str
    sai_model_spec: str
    trainable: bool


CATALOG: dict[str, FamilySpec] = {
    "sdxl_base_v1-0": FamilySpec(
        family_id="sdxl",
        base_model_version="sdxl_base_v1-0",
        architecture="stable-diffusion-xl-v1-base/lora",
        implementation="https://github.com/Stability-AI/generative-models",
        sai_model_spec="1.0.0",
        trainable=True,
    ),
    "sd3.5-large": FamilySpec(
        family_id="sd3.5",
        base_model_version="sd3.5-large",
        architecture="stable-diffusion-v3-5-large/lora",
        implementation="https://github.com/Stability-AI/sd3.5",
        sai_model_spec="1.0.0",
        trainable=False,
    ),
}


@dataclass
class FamilyModules:
    pipe: Any
    vae: Any
    denoise: Any
    tokenizers: list
    text_encoders: list
    noise_scheduler: Any = None


@runtime_checkable
class ModelFamily(Protocol):
    spec: FamilySpec
    trainable: bool

    def load_pipeline(self, path: str, dtype: torch.dtype) -> Any: ...

    def unpack(self, pipe: Any) -> FamilyModules: ...

    def apply_lora(self, cfg: Any, modules: FamilyModules) -> FamilyModules: ...

    def load_lora(self, cfg: Any, modules: FamilyModules) -> dict[str, Any]: ...

    def build_noise_scheduler(self, pipe: Any, cfg: Any) -> Any: ...

    def extra_cond(
        self,
        *,
        src_wh: tuple[int, int],
        bucket_wh: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> dict[str, torch.Tensor]: ...

    def encode_prompts(
        self,
        prompts: list[str],
        modules: FamilyModules,
        cfg: Any,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Any: ...

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
    ) -> torch.Tensor: ...

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
    ) -> torch.Tensor: ...

    def prediction_type(self, cfg: Any) -> str: ...

    def save_lora(
        self,
        accelerator: Any,
        modules: FamilyModules,
        cfg: Any,
        global_step: int,
        epoch: Optional[int] = None,
        final: bool = False,
    ) -> None: ...

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
    ) -> None: ...


def catalog_versions() -> tuple[str, ...]:
    return tuple(CATALOG.keys())


def lookup_spec(base_model_version: str) -> FamilySpec:
    spec = CATALOG.get(base_model_version)
    if spec is None:
        allowed = ", ".join(catalog_versions())
        raise ModelSpecError(
            f"unknown base_model_version {base_model_version!r}; expected one of: {allowed}"
        )
    return spec


def require_matching_spec(
    base_model_version: str,
    architecture: str,
    implementation: str,
    sai_model_spec: str,
) -> FamilySpec:
    spec = lookup_spec(base_model_version)
    mismatches: list[str] = []
    if architecture != spec.architecture:
        mismatches.append(
            f"modelspec_architecture={architecture!r} (expected {spec.architecture!r})"
        )
    if implementation != spec.implementation:
        mismatches.append(
            f"modelspec_implementation={implementation!r} (expected {spec.implementation!r})"
        )
    if sai_model_spec != spec.sai_model_spec:
        mismatches.append(
            f"modelspec_sai_model_spec={sai_model_spec!r} (expected {spec.sai_model_spec!r})"
        )
    if mismatches:
        raise ModelSpecError(
            f"[model_spec] does not match catalog row for {base_model_version!r}: "
            + "; ".join(mismatches)
        )
    return spec


def require_trainable(family: ModelFamily) -> None:
    if family.trainable:
        return
    if family.spec.family_id == "sd3.5":
        raise UnsupportedFamilyError(_SD35_NOT_IMPLEMENTED)
    raise UnsupportedFamilyError(
        f"{family.spec.base_model_version} LoRA training is not implemented yet"
    )


def resolve_family(cfg: Any) -> ModelFamily:
    spec = require_matching_spec(
        cfg.base_model_version,
        cfg.modelspec_architecture,
        cfg.modelspec_implementation,
        cfg.modelspec_sai_model_spec,
    )
    if spec.family_id == "sdxl":
        try:
            from family_sdxl import SdxlFamily
        except ImportError:
            from trainer.family_sdxl import SdxlFamily
        return SdxlFamily(spec)
    if spec.family_id == "sd3.5":
        try:
            from family_sd35 import Sd35Family
        except ImportError:
            from trainer.family_sd35 import Sd35Family
        return Sd35Family(spec)
    raise UnsupportedFamilyError(
        f"{spec.base_model_version} LoRA training is not implemented yet"
    )
