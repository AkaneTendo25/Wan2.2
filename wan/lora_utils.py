# Copyright 2024-2025 LoRA Utils for Wan2.2
"""
LoRA utilities for Wan2.2 models.
Supports both single model and MoE (high/low noise experts) architectures.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch

try:
    from safetensors.torch import load_file as _st_load
except ImportError:
    _st_load = None


def _lora_pair(sd: dict) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Extracts LoRA weight pairs (A, B) or (down, up) from state dict.

    Args:
        sd: State dict containing LoRA weights

    Returns:
        Dictionary mapping base layer name to (A/down, B/up) tensor pairs
    """
    buckets = {}
    for k, v in sd.items():
        if not k.endswith(".weight"):
            continue
        if k.endswith(".lora_down.weight"):
            base = k[:-len(".lora_down.weight")]
            buckets.setdefault(base, {})["down"] = v
        elif k.endswith(".lora_up.weight"):
            base = k[:-len(".lora_up.weight")]
            buckets.setdefault(base, {})["up"] = v
        elif k.endswith(".lora_A.weight"):
            base = k[:-len(".lora_A.weight")]
            buckets.setdefault(base, {})["A"] = v
        elif k.endswith(".lora_B.weight"):
            base = k[:-len(".lora_B.weight")]
            buckets.setdefault(base, {})["B"] = v

    pairs = {}
    for base, d in buckets.items():
        if "down" in d and "up" in d:
            pairs[base] = (d["down"], d["up"])
        elif "A" in d and "B" in d:
            pairs[base] = (d["A"], d["B"])
    return pairs


def _strip_prefixes(name: str) -> str:
    """
    Remove common prefixes from parameter names.

    Args:
        name: Parameter name

    Returns:
        Name with common prefixes stripped
    """
    prefixes = [
        "diffusion_model.",
        "model.diffusion_model.",
        "module.diffusion_model.",
        "transformer.",
        "pipe.dit.",
        "dit.",
    ]
    for prefix in prefixes:
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _normalize_to_wan(base_no_weight: str) -> str:
    """
    Normalize LoRA parameter names to match Wan2.2 naming conventions.

    Handles different LoRA formats from various frameworks:
    - Diffusers: lora_unet_blocks_0_attn1_to_q
    - ComfyUI: transformer.blocks.0.attn1.to_q
    - A1111: model_transformer_blocks_0_attn1_to_q

    Args:
        base_no_weight: Base parameter name without .weight suffix

    Returns:
        Normalized parameter name matching Wan2.2 conventions
    """
    import re

    name = base_no_weight

    # Handle underscore-separated format (e.g., lora_unet_blocks_0_self_attn_q)
    # Convert to dot notation: blocks.0.self_attn.q
    if "lora_unet" in name or "lora_transformer" in name:
        # Remove lora_unet_ or lora_transformer_ prefix
        name = re.sub(r"^lora_(unet|transformer)_", "", name)
        # Replace _ with . for structure
        # But preserve multi-word identifiers like self_attn
        parts = name.split("_")
        result = []
        i = 0
        while i < len(parts):
            part = parts[i]
            # Check if this is part of a multi-word identifier
            if i + 1 < len(parts):
                next_part = parts[i + 1]
                if part in ["self", "cross", "feed", "layer"] or next_part in ["attn", "norm", "forward"]:
                    result.append(f"{part}_{next_part}")
                    i += 2
                    continue
            result.append(part)
            i += 1
        name = ".".join(result)

    # Normalize attention layer names
    # .attn1. -> .self_attn.
    # .attn2. -> .cross_attn.
    name = name.replace(".attn1.", ".self_attn.")
    name = name.replace(".attn2.", ".cross_attn.")

    # Normalize projection names
    # .to_q -> .q
    # .to_k -> .k
    # .to_v -> .v
    # .to_out.0 -> .o
    name = name.replace(".to_q", ".q")
    name = name.replace(".to_k", ".k")
    name = name.replace(".to_v", ".v")
    name = name.replace(".to_out.0", ".o")
    name = name.replace(".to_out", ".o")

    # Normalize FFN names
    # .mlp.fc1 -> .ffn.0
    # .mlp.fc2 -> .ffn.2
    # .feed_forward.net.0.proj -> .ffn.0
    name = name.replace(".mlp.fc1", ".ffn.0")
    name = name.replace(".mlp.fc2", ".ffn.2")
    name = re.sub(r"\.feed_forward\.net\.0\.proj", ".ffn.0", name)
    name = re.sub(r"\.feed_forward\.net\.2\.proj", ".ffn.2", name)

    return name


def _detect_alpha(lsd: dict) -> Optional[float]:
    """
    Detect global alpha value from LoRA state dict.

    Args:
        lsd: LoRA state dict

    Returns:
        Global alpha value if found, None otherwise
    """
    for k, v in lsd.items():
        if k.endswith("alpha") and v.numel() == 1:
            try:
                return float(v.item())
            except Exception:
                pass
    return None


def compute_lora_deltas(
    lora_paths: List[str],
    lora_scales: Optional[List[float]] = None,
    normalize_names: bool = True,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, torch.Tensor]:
    """
    Compute accumulated LoRA weight deltas from multiple LoRA files.

    Formula: Δ = (B @ A) * (alpha / r) * scale

    Args:
        lora_paths: List of paths to .safetensors LoRA files
        lora_scales: Optional scaling factors for each LoRA (default: 1.0 for all)
        normalize_names: Whether to normalize parameter names to Wan2.2 conventions
        logger: Optional logger for info/debug messages

    Returns:
        Dictionary mapping target parameter names to accumulated delta tensors (CPU, float32)

    Raises:
        SystemExit: If safetensors is not installed
    """
    if _st_load is None:
        raise SystemExit(
            "LoRA requested but `safetensors` is not installed. "
            "Run: pip install safetensors"
        )

    if logger is None:
        logger = logging.getLogger("lora")

    # Extend scales to match number of paths
    if lora_scales is None or len(lora_scales) < len(lora_paths):
        lora_scales = list(lora_scales or []) + [1.0] * (
            len(lora_paths) - len(lora_scales or [])
        )

    deltas = {}

    for path, scale in zip(lora_paths, lora_scales):
        logger.info(f"Loading LoRA from: {path} (scale={scale})")
        lsd = _st_load(path, device="cpu")

        # Try to detect global alpha
        global_alpha = _detect_alpha(lsd)
        if global_alpha is not None:
            logger.info(f"  Detected global alpha = {global_alpha}")

        # Process LoRA pairs
        pairs = _lora_pair(lsd)
        logger.info(f"  Found {len(pairs)} LoRA layer pairs")

        for base, (A, B) in pairs.items():
            # Compute delta: Δ = (B @ A) * (alpha / r) * scale
            r = int(A.shape[0])

            # Check for per-layer alpha
            alpha_key = base + ".alpha"
            if alpha_key in lsd:
                alpha = float(lsd[alpha_key].item())
            elif global_alpha is not None:
                alpha = global_alpha
            else:
                alpha = float(r)

            delta = (B.float() @ A.float()) * (alpha / r) * float(scale)

            # Normalize target key
            base_stripped = _strip_prefixes(base)
            if normalize_names:
                base_wan = _normalize_to_wan(base_stripped)
                tgt_key = base_wan + ".weight"
            else:
                tgt_key = base_stripped + ".weight"

            # Accumulate deltas
            deltas[tgt_key] = deltas.get(tgt_key, 0) + delta

    logger.info(f"Prepared {len(deltas)} LoRA target matrices")
    return deltas


def _resolve_param_key(param_names: List[str], target: str) -> Optional[str]:
    """
    Resolve target key to actual parameter name in model.

    Args:
        param_names: List of all parameter names in model
        target: Target parameter name from LoRA

    Returns:
        Resolved parameter name if found, None otherwise
    """
    # Exact match
    if target in param_names:
        return target

    # Suffix match (find shortest match)
    candidates = [name for name in param_names if name.endswith(target)]
    if not candidates:
        return None

    return sorted(candidates, key=len)[0]


def apply_lora_to_model(
    model: torch.nn.Module,
    deltas: Dict[str, torch.Tensor],
    logger: Optional[logging.Logger] = None,
    verbose: bool = False,
) -> Tuple[int, List[str], List[str]]:
    """
    Apply LoRA deltas to model parameters in-place.

    Args:
        model: PyTorch model to apply LoRA to
        deltas: Dictionary of parameter name to delta tensor
        logger: Optional logger
        verbose: Whether to log detailed match information

    Returns:
        Tuple of (matched_count, not_found_keys, mismatched_keys)
    """
    if logger is None:
        logger = logging.getLogger("lora")

    # Get model parameters
    if hasattr(model, "module"):
        param_map = dict(model.module.named_parameters())
    else:
        param_map = dict(model.named_parameters())

    param_names = list(param_map.keys())

    matched = 0
    not_found = []
    mismatched = []

    for tgt_key, delta in deltas.items():
        # Resolve target key to actual parameter name
        resolved = _resolve_param_key(param_names, tgt_key)

        if resolved is None:
            not_found.append(tgt_key)
            continue

        p = param_map[resolved]

        # Check shape compatibility
        if tuple(p.shape) != tuple(delta.shape):
            mismatch_info = (
                f"{tgt_key} ~ {resolved}: "
                f"base={tuple(p.shape)} vs delta={tuple(delta.shape)}"
            )
            mismatched.append(mismatch_info)
            continue

        # Apply delta in-place
        p.data.add_(delta.to(dtype=p.dtype, device=p.device))
        matched += 1

    # Log summary
    logger.info(
        f"[LoRA] Applied to model: "
        f"matched={matched}, not_found={len(not_found)}, "
        f"shape_mismatch={len(mismatched)}"
    )

    if verbose:
        for key in not_found[:30]:
            logger.debug(f"[LoRA] Not found: {key}")
        for info in mismatched[:30]:
            logger.debug(f"[LoRA] Shape mismatch: {info}")

    return matched, not_found, mismatched


def load_and_apply_lora(
    model: torch.nn.Module,
    lora_paths: List[str],
    lora_scales: Optional[List[float]] = None,
    normalize_names: bool = True,
    verbose: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[int, List[str], List[str]]:
    """
    Convenience function to load LoRA weights and apply to model.

    Args:
        model: PyTorch model
        lora_paths: List of paths to LoRA .safetensors files
        lora_scales: Optional scaling factors
        normalize_names: Whether to normalize parameter names
        verbose: Whether to log detailed information
        logger: Optional logger

    Returns:
        Tuple of (matched_count, not_found_keys, mismatched_keys)
    """
    if logger is None:
        logger = logging.getLogger("lora")
        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

    logger.info(f"Loading and applying LoRA: {', '.join(lora_paths)}")

    # Compute deltas
    deltas = compute_lora_deltas(
        lora_paths=lora_paths,
        lora_scales=lora_scales,
        normalize_names=normalize_names,
        logger=logger,
    )

    # Apply to model
    return apply_lora_to_model(
        model=model,
        deltas=deltas,
        logger=logger,
        verbose=verbose,
    )
