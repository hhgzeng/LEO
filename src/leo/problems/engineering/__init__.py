"""Engineering optimization problems from Section 3.4 & Appendix A."""

from leo.problems.engineering.heat_transfer import HeatTransferProblem
from leo.problems.engineering.nosecone import NoseConeProblem
from leo.problems.engineering.nozzle import NozzleShapeProblem
from leo.problems.engineering.windfarm import WindFarmProblem

__all__ = [
    "HeatTransferProblem",
    "NoseConeProblem",
    "NozzleShapeProblem",
    "WindFarmProblem",
]
