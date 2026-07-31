"""The assembly layer: natural language to :class:`AssemblySpec`.

This is the only part of the package that touches the network, and the only
part whose output cannot be trusted. Its job is narrow by design: choose which
catalog modules to use and which ports to join. Geometry, mass and inertia are
never asked for, and URDF is never generated here.

The model runs locally behind an OpenAI-compatible HTTP endpoint (Ollama,
LM Studio, vLLM, llama.cpp server), configured entirely through environment
variables.
"""

from urdfgen.llm.config import LLMConfig, LLMConfigError, load_config
from urdfgen.llm.repair import AssemblyOutcome, assemble_with_repair

__all__ = [
    "AssemblyOutcome",
    "LLMConfig",
    "LLMConfigError",
    "assemble_with_repair",
    "load_config",
]
