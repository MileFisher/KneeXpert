"""
Model compute profiler for KneeXpert.

Calculates and caches per-model compute cost for all X-ray ensemble models and
the MRI pipeline (MACS-Net + DeiT-S). Results are exposed via the health
endpoint and prediction responses.

Uses fvcore for counting when available, falls back to published architecture
figures otherwise.

Units — read this before quoting any number
-------------------------------------------
The source-of-truth constants below are **GMACs** (giga multiply-accumulate
operations). That is what `fvcore.nn.FlopCountAnalysis` actually returns and
what the torchvision / timm / DeiT papers report, despite all three calling it
"FLOPs". One MAC is one multiply plus one add, so:

    GFLOPs = 2 x GMACs

Every entry emitted by this module carries both `gmacs` and `gflops` so the
consumer never has to guess which convention a number follows.
"""

from __future__ import annotations

from typing import Any

import torch

from mri.config import MAX_SAMPLES_PER_STUDY

# One multiply-accumulate = one multiply + one add.
FLOPS_PER_MAC = 2

# ── X-ray model compute ────────────────────────────────────────────────────────

# GMACs for standard torchvision models at 224x224 input (1x3x224x224).
# Source: torchvision model zoo docs + fvcore measurement.
XRAY_MODEL_GMACS: dict[str, float] = {
    "densenet201": 4.31,
    "resnet101": 7.85,
    "resnet50": 4.12,
    "vgg19": 19.67,
    "vgg19_bn": 19.67,
}

# Parameter counts (millions) for standard torchvision models.
#
# These are the stock 1000-class ImageNet figures. This project swaps in a
# 5-class head, so they run ~2M high (DenseNet201: 1920x1000 = 1.92M becomes
# 1920x5 = 9.6k). Only a fallback — get_xray_flops() measures the real model
# when fvcore is available, which is the normal path.
XRAY_MODEL_PARAMS_M: dict[str, float] = {
    "densenet201": 20.01,
    "resnet101": 44.55,
    "resnet50": 25.56,
    "vgg19": 143.67,
    "vgg19_bn": 143.67,
}

# Custom MLP head, as a rough fallback only. Measurement shows the real heads
# are ~1.1M (DenseNet/ResNet50) to ~2.2M (VGG), so this overshoots — but it is
# only reached when fvcore is unavailable. The GMACs delta is genuinely
# negligible either way.
CUSTOM_HEAD_EXTRA_PARAMS_M = 5.0
CUSTOM_HEAD_EXTRA_GMACS = 0.01

# ── MRI model compute ──────────────────────────────────────────────────────────
# Note: "MACS" in MACS-Net is the model's name; "GMACs" is the unit. Unrelated.

# MACS-Net: SwinUNETR(in_channels=1, out_channels=1, feature_size=24,
# spatial_dims=2) at 128x128 — see mri/macs_net.py.
#
# Measured with fvcore on that exact architecture: conv 0.835 + linear 0.305 +
# matmul 0.050 + norms 0.018 = 1.21 GMACs, 6.30M params.
#
# These previously read 5.73 GMACs / 62.19M params, which are the *3D*
# Swin-UNETR feature_size=48 figures — a different network. The 10x parameter
# gap is what exposed it: the sanity check rejected a correct 1.21 GMACs
# reading for falling below a baseline that was itself wrong.
MACS_NET_GMACS = 1.21
MACS_NET_PARAMS_M = 6.30

# DeiT-Small (deit_small_patch16_224): 224x224 input, 16x16 patches.
# From timm / DeiT paper: ~4.6 GMACs, ~22M params.
DEIT_SMALL_GMACS = 4.61
DEIT_SMALL_PARAMS_M = 22.05

# The 2.5D pipeline stacks (z-1, z, z+1) and cleans EACH of the three planes
# with MACS-Net before a single DeiT forward pass — see
# mri/preprocess.py::build_25d_tensor. So MACS-Net runs 3x per sampled slice.
MACS_CALLS_PER_SAMPLED_SLICE = 3
DEIT_CALLS_PER_SAMPLED_SLICE = 1

# ── Measurement sanity check ───────────────────────────────────────────────────

# fvcore contributes 0 for any op it has no handler for, and it has none for
# Swin-style attention — the exact shape MACS-Net is built from. The failure is
# silent: you get a smaller number, not an error. So a measurement that lands
# below this fraction of the published architecture estimate is treated as a
# counting failure and discarded rather than trusted.
#
# One-sided on purpose: a measurement ABOVE the estimate is plausibly a genuine
# correction (our constants are published figures for the stock architecture,
# not this checkpoint), so it is accepted.
MEASUREMENT_PLAUSIBILITY_FLOOR = 0.5

# Ops fvcore lists as "unsupported" that contribute no multiply-accumulates
# anyway. Pooling, elementwise arithmetic, normalisation and activations are
# conventionally excluded from MAC counts, so their absence is correct rather
# than an undercount — they must not trigger a lower-bound warning.
#
# Deliberately NOT here: aten::scaled_dot_product_attention, aten::einsum,
# aten::bmm, aten::matmul. Those are real MACs and their absence is a genuine
# undercount.
NEGLIGIBLE_OPS: frozenset[str] = frozenset({
    # pooling
    "aten::adaptive_avg_pool2d",
    "aten::avg_pool2d",
    "aten::max_pool2d",
    # elementwise arithmetic
    "aten::abs",
    "aten::add",
    "aten::add_",
    "aten::div",
    "aten::div_",
    "aten::exp",
    "aten::mul",
    "aten::mul_",
    "aten::ne",
    "aten::neg",
    "aten::rsub",
    "aten::sqrt",
    "aten::sub",
    "aten::sub_",
    # activations
    "aten::gelu",
    "aten::leaky_relu",
    "aten::leaky_relu_",
    "aten::relu",
    "aten::relu_",
    "aten::sigmoid",
    "aten::silu",
    "aten::softmax",
    "aten::tanh",
    # normalisation
    "aten::batch_norm",
    "aten::group_norm",
    "aten::instance_norm",
    "aten::layer_norm",
    # shape / memory movement (aten::roll is Swin's shifted-window op)
    "aten::cat",
    "aten::clone",
    "aten::dropout",
    "aten::fill_",
    "aten::flatten",
    "aten::pad",
    "aten::permute",
    "aten::reshape",
    "aten::roll",
    "aten::transpose",
    "aten::view",
})


# ── Cached results ─────────────────────────────────────────────────────────────

_xray_flops_cache: dict[str, dict[str, Any]] | None = None
_mri_flops_cache: dict[str, dict[str, Any]] | None = None

UNIT_NOTE = (
    "gmacs = multiply-accumulate operations (the fvcore / torchvision / timm "
    "convention, often mislabelled 'FLOPs'); gflops = 2 x gmacs."
)


def _cost(gmacs: float) -> dict[str, float]:
    """Express a MAC count in both conventions."""
    return {
        "gmacs": round(gmacs, 2),
        "gflops": round(gmacs * FLOPS_PER_MAC, 2),
    }


def _estimate_xray_flops(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Estimate compute for an X-ray model from its architecture config."""
    family = config.get("family", model_name.split("_")[0])
    is_custom = config.get("is_custom", False)

    base_gmacs = XRAY_MODEL_GMACS.get(family, 4.0)
    base_params_m = XRAY_MODEL_PARAMS_M.get(family, 25.0)

    if is_custom:
        base_gmacs += CUSTOM_HEAD_EXTRA_GMACS
        base_params_m += CUSTOM_HEAD_EXTRA_PARAMS_M

    return {
        "model_id": model_name,
        "family": family,
        **_cost(base_gmacs),
        "params_m": round(base_params_m, 2),
        "input_shape": [1, 3, 224, 224],
        "per": "image",
        "method": "architecture_estimate",
    }


def get_xray_flops() -> dict[str, dict[str, Any]]:
    """
    Compute cost for all configured X-ray models.

    Seeds every entry with the published architecture figure, then overwrites
    with a live fvcore measurement where one is credible. Costs a few seconds
    on the first call (all eight architectures are built and traced); cached
    thereafter.
    """
    global _xray_flops_cache
    if _xray_flops_cache is not None:
        return _xray_flops_cache

    from xray.loader import MODELS_CONFIG

    result: dict[str, dict[str, Any]] = {}
    for name, cfg in MODELS_CONFIG.items():
        result[name] = _estimate_xray_flops(name, cfg)

    _measure_xray_with_fvcore(result)

    _xray_flops_cache = result
    return result


def get_mri_flops() -> dict[str, dict[str, Any]]:
    """
    Compute cost for MRI pipeline models (MACS-Net + DeiT-S).

    Figures are **per forward pass**, not per study. Use `mri_study_cost()` for
    the study-level total, which accounts for the 3 MACS-Net calls per sampled
    slice and the number of slices sampled.
    """
    global _mri_flops_cache
    if _mri_flops_cache is not None:
        return _mri_flops_cache

    result: dict[str, dict[str, Any]] = {
        "macs_net": {
            "model_id": "macs_net",
            "display_name": "MACS-Net (Swin-UNETR)",
            **_cost(MACS_NET_GMACS),
            "params_m": MACS_NET_PARAMS_M,
            "input_shape": [1, 1, 128, 128],
            "per": "forward_pass",
            "calls_per_sampled_slice": MACS_CALLS_PER_SAMPLED_SLICE,
            "method": "architecture_estimate",
        },
        "deit_small": {
            "model_id": "deit_small",
            "display_name": "DeiT-Small (2.5D multi-label)",
            **_cost(DEIT_SMALL_GMACS),
            "params_m": DEIT_SMALL_PARAMS_M,
            "input_shape": [1, 3, 224, 224],
            "per": "forward_pass",
            "calls_per_sampled_slice": DEIT_CALLS_PER_SAMPLED_SLICE,
            "method": "architecture_estimate",
        },
    }

    _measure_with_fvcore(result)

    _mri_flops_cache = result
    return result


def is_plausible_measurement(measured_gmacs: float, estimated_gmacs: float) -> bool:
    """
    Whether a measured MAC count is credible against the architecture estimate.

    Rejects silent undercounts (see MEASUREMENT_PLAUSIBILITY_FLOOR). A zero or
    negative measurement is always rejected; a measurement above the estimate
    is always accepted.
    """
    if measured_gmacs <= 0:
        return False
    if estimated_gmacs <= 0:
        return True
    return measured_gmacs >= estimated_gmacs * MEASUREMENT_PLAUSIBILITY_FLOOR


def _apply_measurement(
    entry: dict[str, Any],
    measured_gmacs: float,
    params_m: float,
    unhandled_ops: list[str],
) -> None:
    """
    Adopt or reject one fvcore measurement, in place.

    Shared by the X-ray and MRI paths so both get identical semantics. Three
    ways a measurement can lose to the architecture estimate:

    1. It falls below the plausibility floor — treated as a counting failure.
    2. Significant ops were skipped AND the result is below the estimate, so
       the measurement is a known lower bound on a figure the estimate already
       accounts for (DeiT's attention is the live example).
    3. Anything raised while measuring (handled by the callers).

    The parameter count is adopted unconditionally: numel() is exact and owes
    nothing to op coverage.
    """
    estimated_gmacs = entry["gmacs"]
    entry["params_m"] = round(params_m, 2)
    significant = [op for op in unhandled_ops if op not in NEGLIGIBLE_OPS]

    if not is_plausible_measurement(measured_gmacs, estimated_gmacs):
        entry["rejected_fvcore_gmacs"] = round(measured_gmacs, 2)
        detail = (
            f" Ops with no fvcore handler: {', '.join(significant)}."
            if significant
            else (
                " No significant ops were skipped, so the architecture estimate "
                "itself is the more likely error — worth re-baselining."
            )
        )
        entry["measurement_warning"] = (
            f"fvcore reported {measured_gmacs:.2f} GMACs, under "
            f"{MEASUREMENT_PLAUSIBILITY_FLOOR:.0%} of the {estimated_gmacs:.2f} GMACs "
            f"architecture estimate — discarded as an undercount, keeping the estimate."
            + detail
        )
        return

    if significant and measured_gmacs < estimated_gmacs:
        entry["rejected_fvcore_gmacs"] = round(measured_gmacs, 2)
        entry["measurement_warning"] = (
            f"fvcore reported {measured_gmacs:.2f} GMACs but has no handler for "
            f"{', '.join(significant)}, making it a lower bound below the "
            f"{estimated_gmacs:.2f} GMACs estimate — keeping the estimate, which "
            f"accounts for those ops."
        )
        return

    entry.update(_cost(measured_gmacs))
    entry["method"] = "fvcore"
    if significant:
        entry["measurement_warning"] = (
            f"fvcore has no handler for {', '.join(significant)}; the measured count "
            f"exceeds the estimate anyway, so it is adopted as a lower bound."
        )


def _unhandled_ops(analysis: Any, limit: int = 4) -> list[str]:
    """Op names fvcore had no handler for — each contributed 0 to the total."""
    try:
        skipped = analysis.unsupported_ops()
    except Exception:
        return []
    return sorted(skipped)[:limit]


def _measure_with_fvcore(result: dict[str, dict[str, Any]]) -> None:
    """
    Attempt precise measurement with fvcore (if installed and weights present).

    Leaves the architecture estimate in place on any failure — including a
    measurement that fails the plausibility check — so a silently-undercounted
    Swin attention stack can never quietly replace a good published figure.
    `method` always says which figure the caller ended up with, and
    `measurement_warning` explains anything that went wrong.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    targets = (
        ("macs_net", "mri.macs_net", "get_macs", (1, 1, 128, 128)),
        ("deit_small", "mri.deit_classifier", "get_deit", (1, 3, 224, 224)),
    )

    for key, module_path, getter_name, shape in targets:
        try:
            module = __import__(module_path, fromlist=[getter_name])
            wrapper = getattr(module, getter_name)()
            wrapper.load()

            dummy = torch.randn(*shape, device=device)
            analysis = FlopCountAnalysis(wrapper.model, dummy)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)

            _apply_measurement(
                result[key],
                measured_gmacs=analysis.total() / 1e9,
                params_m=sum(p.numel() for p in wrapper.model.parameters()) / 1e6,
                unhandled_ops=_unhandled_ops(analysis),
            )
        except Exception:
            continue


def _measure_xray_with_fvcore(result: dict[str, dict[str, Any]]) -> None:
    """
    Measure every configured X-ray model with fvcore.

    Needs no checkpoints. FLOPs and parameter counts depend only on the
    architecture, and ModelLoader builds the architecture before it touches the
    .pth file (see loader._load_model), so this produces real figures even on a
    clone with no weights on disk — unlike the MRI path, which must `.load()`.

    Models are built one at a time and released, so peak memory is one model
    (VGG19 at ~570 MB) rather than all eight at once.
    """
    try:
        from fvcore.nn import FlopCountAnalysis
    except ImportError:
        return

    # Instantiated only for its architecture builders; no weights are read.
    # loader.py is off-limits for edits, so we call what it already exposes.
    from xray.loader import MODELS_CONFIG, ModelLoader

    loader = ModelLoader()
    dummy = torch.randn(1, 3, 224, 224)

    for name, cfg in MODELS_CONFIG.items():
        entry = result.get(name)
        if entry is None:
            continue
        model = None
        try:
            model = (
                loader._create_custom_model(cfg["custom_arch"])
                if cfg.get("is_custom")
                else loader._create_standard_model(cfg["type"])
            )
            if model is None:
                continue
            model.eval()

            analysis = FlopCountAnalysis(model, dummy)
            analysis.unsupported_ops_warnings(False)
            analysis.uncalled_modules_warnings(False)

            _apply_measurement(
                entry,
                measured_gmacs=analysis.total() / 1e9,
                params_m=sum(p.numel() for p in model.parameters()) / 1e6,
                unhandled_ops=_unhandled_ops(analysis),
            )
        except Exception:
            continue
        finally:
            del model


def mri_study_cost(sampled_slices: int | None = None) -> dict[str, Any]:
    """
    Study-level MRI pipeline cost.

    A single sampled slice costs `3 x MACS-Net + 1 x DeiT` because the 2.5D
    tensor stacks three MACS-cleaned planes. A study runs that for every
    sampled slice.

    Args:
        sampled_slices: Slices actually processed. Defaults to the configured
            ceiling (`MAX_SAMPLES_PER_STUDY`) for the worst-case estimate.
    """
    mri = get_mri_flops()
    slices = MAX_SAMPLES_PER_STUDY if sampled_slices is None else max(0, int(sampled_slices))

    per_slice_gmacs = (
        mri["macs_net"]["gmacs"] * MACS_CALLS_PER_SAMPLED_SLICE
        + mri["deit_small"]["gmacs"] * DEIT_CALLS_PER_SAMPLED_SLICE
    )

    return {
        "per_sampled_slice": {
            **_cost(per_slice_gmacs),
            "macs_net_calls": MACS_CALLS_PER_SAMPLED_SLICE,
            "deit_calls": DEIT_CALLS_PER_SAMPLED_SLICE,
        },
        "per_study": {
            **_cost(per_slice_gmacs * slices),
            "sampled_slices": slices,
            "is_upper_bound": sampled_slices is None,
        },
    }


def get_all_flops() -> dict[str, Any]:
    """Compute-cost summary for the health endpoint."""
    xray = get_xray_flops()
    mri = get_mri_flops()
    study = mri_study_cost()

    xray_total_gmacs = sum(v["gmacs"] for v in xray.values())
    model_count = len(xray)

    return {
        "unit_note": UNIT_NOTE,
        "xray_models": xray,
        "mri_models": mri,
        "totals": {
            "xray_mean_per_model": _cost(xray_total_gmacs / model_count) if model_count else _cost(0),
            "xray_ensemble_all": {**_cost(xray_total_gmacs), "model_count": model_count},
            "mri_per_sampled_slice": study["per_sampled_slice"],
            "mri_per_study_max": study["per_study"],
        },
    }


def get_model_flops_for_response(
    model_names: list[str],
    sampled_slices: int | None = None,
) -> dict[str, Any]:
    """
    Compute cost for the specific models used in a prediction response.

    Args:
        model_names: Model ids reported in the prediction result.
        sampled_slices: For MRI, the number of slices actually processed
            (`slices_processed`). Required for an accurate study-level total —
            without it the total falls back to the configured maximum.
    """
    xray = get_xray_flops()
    mri = get_mri_flops()

    result: dict[str, Any] = {"unit_note": UNIT_NOTE}
    for name in model_names:
        source = xray.get(name) or mri.get(name)
        if source is None:
            continue
        entry: dict[str, Any] = {
            "gmacs": source["gmacs"],
            "gflops": source["gflops"],
            "params_m": source["params_m"],
            "per": source["per"],
            "method": source["method"],
        }
        for optional in ("calls_per_sampled_slice", "measurement_warning"):
            if optional in source:
                entry[optional] = source[optional]
        result[name] = entry

    # MRI pipeline: report both the per-slice unit cost and the study total.
    if "macs_net" in model_names and "deit_small" in model_names:
        study = mri_study_cost(sampled_slices)
        result["pipeline_per_sampled_slice"] = study["per_sampled_slice"]
        result["pipeline_total"] = {
            **study["per_study"],
            "params_m": round(
                mri["macs_net"]["params_m"] + mri["deit_small"]["params_m"], 2
            ),
        }

    return result
