"""Mini Intelligence: an English query router for a fixed pool of 20 language models."""
from .router import E06Router, MiniIntelligence

__version__ = "1.0.0"
__all__ = ["MiniIntelligence", "E06Router"]
