"""Data schemas for HPC-DS1 state representation and outputs."""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Literal
import json
from datetime import datetime


@dataclass
class Hardware:
    """Hardware specifications."""
    cpu_cores: int
    cpu_freq_ghz: float
    l3_cache_mb: int
    gpu_model: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    ram_gb: int = 24
    interconnect: Literal["ethernet", "infiniband", "none"] = "ethernet"


@dataclass
class Workload:
    """Workload specification."""
    name: str  # "nbody", "lammps", "gromacs"
    problem_size: int
    algorithm_type: Literal["embarrassingly_parallel", "comm_heavy", "memory_bound"]
    flops_estimate: float = 0.0


@dataclass
class Configuration:
    """HPC configuration parameters."""
    mpi_ranks: int
    omp_threads: int
    gpu_enabled: bool = False
    cuda_block_size: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass
class Telemetry:
    """Performance telemetry from an experiment."""
    runtime_sec: float
    cpu_util_pct: float
    gpu_util_pct: Optional[float] = None
    memory_util_pct: float = 0.0
    memory_gb_used: float = 0.0
    mpi_comm_overhead_pct: float = 0.0
    l3_cache_miss_rate: Optional[float] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


@dataclass
class ExperimentRecord:
    """A single completed experiment in the history."""
    step: int
    config: Configuration
    telemetry: Telemetry
    runtime_sec: float
    speedup_vs_baseline: float
    cost_sec: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "config": self.config.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "runtime_sec": self.runtime_sec,
            "speedup_vs_baseline": self.speedup_vs_baseline,
            "cost_sec": self.cost_sec,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            step=d["step"],
            config=Configuration.from_dict(d["config"]),
            telemetry=Telemetry.from_dict(d["telemetry"]),
            runtime_sec=d["runtime_sec"],
            speedup_vs_baseline=d["speedup_vs_baseline"],
            cost_sec=d.get("cost_sec", 0.0),
            timestamp=d.get("timestamp", ""),
        )


@dataclass
class HPCState:
    """Complete HPC tuning state for decision making."""
    hardware: Hardware
    workload: Workload
    current_config: Configuration
    telemetry: Telemetry
    experiment_history: List[ExperimentRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "hardware": asdict(self.hardware),
            "workload": asdict(self.workload),
            "current_config": self.current_config.to_dict(),
            "telemetry": self.telemetry.to_dict(),
            "experiment_history": [e.to_dict() for e in self.experiment_history],
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            hardware=Hardware(**d["hardware"]),
            workload=Workload(**d["workload"]),
            current_config=Configuration.from_dict(d["current_config"]),
            telemetry=Telemetry.from_dict(d["telemetry"]),
            experiment_history=[ExperimentRecord.from_dict(e) for e in d.get("experiment_history", [])],
        )

    @classmethod
    def from_json(cls, json_str: str):
        return cls.from_dict(json.loads(json_str))

    def add_experiment(self, record: ExperimentRecord):
        """Add an experiment to history (keep last 10)."""
        self.experiment_history.append(record)
        self.experiment_history = self.experiment_history[-10:]


# Candidate Actions for HPC Tuning
HPC_ACTIONS = [
    "INCREASE_MPI",          # Double MPI ranks
    "DECREASE_MPI",          # Halve MPI ranks
    "INCREASE_OMP",          # Double OpenMP threads
    "DECREASE_OMP",          # Halve OpenMP threads
    "ENABLE_GPU",            # Turn on GPU
    "DISABLE_GPU",           # Turn off GPU
    "INCREASE_BLOCK_SIZE",   # Increase CUDA block size
    "DECREASE_BLOCK_SIZE",   # Decrease CUDA block size
]

HPCAction = Literal[
    "INCREASE_MPI",
    "DECREASE_MPI",
    "INCREASE_OMP",
    "DECREASE_OMP",
    "ENABLE_GPU",
    "DISABLE_GPU",
    "INCREASE_BLOCK_SIZE",
    "DECREASE_BLOCK_SIZE",
]


@dataclass
class HPC_DS1_Output:
    """Output from HPC-DS1 decision model."""
    action_values: List[float]  # 8 values, one per action
    action_confidence: List[float]  # 8 confidence scores
    action_ranking: List[str]  # Sorted by value, highest first
    top_action: str  # Highest-value action
    top_value: float
    top_confidence: float
    stop_prob: float  # Probability model recommends stopping
    bottleneck_scores: Dict[str, float]  # CPU/GPU/Memory/Comm bottleneck probabilities

    def to_dict(self) -> dict:
        return {
            "action_values": self.action_values,
            "action_confidence": self.action_confidence,
            "action_ranking": self.action_ranking,
            "top_action": self.top_action,
            "top_value": self.top_value,
            "top_confidence": self.top_confidence,
            "stop_prob": self.stop_prob,
            "bottleneck_scores": self.bottleneck_scores,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)
