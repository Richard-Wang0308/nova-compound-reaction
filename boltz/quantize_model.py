"""
Quantization utilities for Boltz-2 models.
Supports FP16 and INT8 (weight-only) quantization for faster inference.
"""
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Literal
import bittensor as bt


def quantize_model_fp16(model: nn.Module) -> nn.Module:
    """
    FP16 quantization is handled by PyTorch Lightning trainer with precision="16-mixed".
    This function is kept for compatibility but doesn't modify the model.
    
    NOTE: Do NOT use .half() here - it causes dtype mismatches (Float inputs vs Half weights).
    PyTorch Lightning AMP handles dtype conversions automatically.
    
    Args:
        model: PyTorch model (unchanged)
    
    Returns:
        Model (unchanged - FP16 handled by trainer AMP)
    """
    bt.logging.info("FP16 quantization will be handled by PyTorch Lightning AMP")
    # Don't convert to .half() - let PyTorch Lightning handle it with AMP
    return model


def quantize_model_int8_weight_only(model: nn.Module) -> nn.Module:
    """
    Apply INT8 weight-only quantization (experimental).
    This quantizes weights to INT8 but keeps activations in FP16/FP32.
    
    Args:
        model: PyTorch model to quantize
    
    Returns:
        INT8 weight-only quantized model
    """
    try:
        from torch.ao.quantization import quantize_dynamic
        bt.logging.info("Quantizing model to INT8 (weight-only)...")
        
        # Quantize linear and embedding layers
        model = quantize_dynamic(
            model,
            {nn.Linear, nn.Embedding},
            dtype=torch.qint8
        )
        bt.logging.info("Model quantized to INT8 (weight-only)")
        return model
    except ImportError:
        bt.logging.warning("torch.ao.quantization not available, skipping INT8 quantization")
        return model
    except Exception as e:
        bt.logging.warning(f"INT8 quantization failed: {e}, using original model")
        return model


def quantize_model(
    model: nn.Module,
    quantization: Literal["none", "fp16", "int8"] = "fp16"
) -> nn.Module:
    """
    Quantize a model based on the specified method.
    
    Args:
        model: PyTorch model to quantize
        quantization: Quantization method ("none", "fp16", or "int8")
    
    Returns:
        Quantized model
    """
    if quantization == "none":
        return model
    elif quantization == "fp16":
        return quantize_model_fp16(model)
    elif quantization == "int8":
        return quantize_model_int8_weight_only(model)
    else:
        bt.logging.warning(f"Unknown quantization method: {quantization}, using original model")
        return model


def save_quantized_model(
    model: nn.Module,
    output_path: Path,
    quantization: Literal["none", "fp16", "int8"] = "fp16"
) -> None:
    """
    Save a quantized model to disk.
    
    Args:
        model: Quantized model
        output_path: Path to save the model
        quantization: Quantization method used
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if quantization == "none":
        torch.save(model.state_dict(), output_path)
    else:
        # Save quantized model
        torch.save({
            "state_dict": model.state_dict(),
            "quantization": quantization,
        }, output_path)
    
    bt.logging.info(f"Quantized model saved to {output_path}")

