"""

This module provides a utility function to determine the appropriate device for 
PyTorch computations based on user preference and hardware availability.

"""
import torch

def get_device(preference: str = "auto") -> torch.device:
    if preference != "auto":
        return torch.device(preference)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")