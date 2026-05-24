"""Abstract backend interface."""

from abc import ABC, abstractmethod
from typing import List, Union

import numpy as np


class BaseBackend(ABC):
    """Backend interface for running MERaLiON-3-3B-ASR inference."""

    @abstractmethod
    def transcribe_chunks(self, chunks: List[np.ndarray]) -> List[str]:
        """Transcribe a list of mono 16 kHz float32 chunks. Returns a list of strings."""
