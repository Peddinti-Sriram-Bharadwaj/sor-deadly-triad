"""Over- and under-relaxed (SOR) Q-learning on classic deadly-triad counterexamples."""

from .problems import baird, tsitsiklis_van_roy, PROBLEMS
from .learners import expected_run, neural_run, sor_target

__all__ = ["baird", "tsitsiklis_van_roy", "PROBLEMS", "expected_run", "neural_run", "sor_target"]
