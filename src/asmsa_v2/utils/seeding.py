import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch.

    Parameters
    ----------
    seed : int
    deterministic : bool
        Also pin cuDNN to deterministic kernels. This makes runs bit-wise
        reproducible at some cost in speed, which is worth it when comparing
        two objectives or reproducing a published result, but is off by
        default because most of the pipeline does not need it.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
