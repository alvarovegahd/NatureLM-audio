"""Module for training utilities.

This module contains utility functions for training models. For example, saving model checkpoints.
"""

import logging
import os
import tempfile
from typing import Any, Union

import torch
import torch.nn as nn

from NatureLM.storage_handler import StorageHandler

# from NatureLM.storage_utils import GSPath, is_gcs_path

logger = logging.getLogger(__name__)


def maybe_unwrap_dist_model(model: nn.Module, use_distributed: bool) -> nn.Module:
    return model.module if use_distributed else model


def get_state_dict(model, drop_untrained_params: bool = True) -> dict[str, Any]:
    """Get model state dict. Optionally drop untrained parameters to keep only those that require gradient.

    Args:
        model: Model to get state dict from
        drop_untrained_params: Whether to drop untrained parameters

    Returns:
        dict: Model state dict
    """
    if not drop_untrained_params:
        return model.state_dict()

    param_grad_dict = {k: v.requires_grad for (k, v) in model.named_parameters()}
    state_dict = model.state_dict()

    for k in list(state_dict.keys()):
        if k in param_grad_dict.keys() and not param_grad_dict[k]:
            # delete parameters that do not require gradient
            del state_dict[k]

    return state_dict



def torch_save_to_bucket(save_obj: Any, save_path: Union[str, os.PathLike], compress: bool = True) -> None:
    """Save an object to a storage backend (local or GCS)."""
    handler = StorageHandler()  # or inject it

    resolved_path = handler.resolve(save_path)

    # If GCS: resolved_path will be a GSPath and has `.upload_from`
    # If local: resolved_path is a Path and can use torch.save directly
    if hasattr(resolved_path, "upload_from"):
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(save_obj, tmp.name, _use_new_zipfile_serialization=compress)
            resolved_path.upload_from(tmp.name)
    else:
        torch.save(save_obj, resolved_path, _use_new_zipfile_serialization=compress)


def save_model_checkpoint(
    model: nn.Module,
    save_path: Union[str, os.PathLike],
    use_distributed: bool = False,
    drop_untrained_params: bool = False,
    **objects_to_save,
) -> None:
    """Save model checkpoint.

    Args:
        model (nn.Module): Model to save
        output_dir (str): Output directory to save checkpoint
        use_distributed (bool): Whether the model is distributed, if so, unwrap it. Default: False.
        is_best (bool): Whether the model is the best in the training run. Default: False.
        drop_untrained_params (bool): Whether to drop untrained parameters to save. Default: True.
        prefix (str): Prefix to add to the checkpoint file name. Default: "".
        extention (str): Extension to use for the checkpoint file. Default: "pth".
        **objects_to_save: Additional objects to save, e.g. optimizer state dict, etc.
    """
   
    handler = StorageHandler()
    resolved_path = handler.resolve(save_path)

    model_no_ddp = maybe_unwrap_dist_model(model, use_distributed)
    state_dict = get_state_dict(model_no_ddp, drop_untrained_params)
    save_obj = {
        "model": state_dict,
        **objects_to_save,
    }

    logger.info(f"Saving checkpoint to {resolved_path}")

    if hasattr(resolved_path, "upload_from"):
        # It's a GSPath
        with tempfile.NamedTemporaryFile() as tmp:
            torch.save(save_obj, tmp.name)
            resolved_path.upload_from(tmp.name)
    else:
        # It's a local Path
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_obj, resolved_path)