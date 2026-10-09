"""Experiment database for logging and querying."""

import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime


class ExperimentDB:
    """Simple file-based experiment database."""

    def __init__(self, db_file: Path = None):
        """
        Initialize experiment database.

        Args:
            db_file: Path to JSONL file (one JSON per line)
        """
        if db_file is None:
            db_file = Path("experiments.jsonl")

        self.db_file = Path(db_file)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing experiments
        self.experiments = []
        if self.db_file.exists():
            with open(self.db_file, "r") as f:
                for line in f:
                    if line.strip():
                        self.experiments.append(json.loads(line))

    def log_experiment(self, experiment: Dict[str, Any]) -> None:
        """Log a single experiment."""
        experiment["timestamp"] = experiment.get("timestamp", datetime.now().isoformat())
        self.experiments.append(experiment)

        # Append to file
        with open(self.db_file, "a") as f:
            f.write(json.dumps(experiment) + "\n")

    def get_all_experiments(self) -> List[Dict]:
        """Get all experiments."""
        return self.experiments

    def get_experiment_count(self) -> int:
        """Get number of experiments."""
        return len(self.experiments)

    def filter_by_config(self, mpi: int = None, omp: int = None, gpu: bool = None) -> List[Dict]:
        """Filter experiments by configuration."""
        results = []
        for exp in self.experiments:
            config = exp.get("config", {})
            if (mpi is None or config.get("mpi_ranks") == mpi) and \
               (omp is None or config.get("omp_threads") == omp) and \
               (gpu is None or config.get("gpu_enabled") == gpu):
                results.append(exp)
        return results

    def export_csv(self, output_file: Path) -> None:
        """Export experiments to CSV."""
        import csv

        if not self.experiments:
            print(f"No experiments to export")
            return

        # Flatten nested dicts
        rows = []
        for exp in self.experiments:
            row = {
                "timestamp": exp.get("timestamp", ""),
                "mpi": exp.get("config", {}).get("mpi_ranks", ""),
                "omp": exp.get("config", {}).get("omp_threads", ""),
                "gpu": exp.get("config", {}).get("gpu_enabled", ""),
                "block_size": exp.get("config", {}).get("cuda_block_size", ""),
                "problem_size": exp.get("config", {}).get("problem_size", ""),
                "runtime": exp.get("runtime_sec", ""),
                "speedup": exp.get("speedup_vs_baseline", ""),
                "cpu_util": exp.get("telemetry", {}).get("cpu_util_pct", ""),
                "gpu_util": exp.get("telemetry", {}).get("gpu_util_pct", ""),
            }
            rows.append(row)

        with open(output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Exported {len(rows)} experiments to {output_file}")
