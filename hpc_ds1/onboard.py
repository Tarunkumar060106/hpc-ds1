"""`hpc-ds1 setup`: teach HPC-DS1 the machine it is installed on.

    measure   run the kernels in every configuration on this machine (resumable)
    learn     label the measurements, train, calibrate the confidence
    guard     fit the familiarity guard on this machine's measurements
    save      ~/.hpc_ds1/model/hpc_ds1.pt, picked up by hpc_ds1.load() from then on

Everything stays on the machine: measurements in ~/.hpc_ds1/data, a report in
~/.hpc_ds1/setup_report.json. Set HPC_DS1_HOME to keep them somewhere else.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from .api import DEFAULT_ESCALATE_BELOW, DEFAULT_THRESHOLD, decision_mode, home_dir
from .collect import collect
from .dataset import build_states, build_tables, load_hardware, load_measurements
from .training import evaluate, fit_guard, fit_temperature, reliability, save_checkpoint, split, tensors, train_one


def setup(home=None, quick=False, workloads=None, size="medium", collect_data=True, device=None, log=print):
    home = Path(home or home_dir())
    data, ckpt = home / "data", home / "model" / "hpc_ds1.pt"
    t0 = time.time()

    if collect_data:
        from .runner import Engine
        log(f"[1/3] measuring this machine ({'quick' if quick else 'full'} run) -> {data}")
        engine = Engine()
        hw = engine.hardware
        log(f"      {hw['cpu_model']} | {hw['cpu_cores']} cores / {hw['cpu_threads']} threads | "
            f"{hw['ram_gb']} GB | GPU: {hw['gpu_model'] or 'none'}")
        try:
            collect(engine, data, workloads, quick=quick, log=lambda m: log("      " + m))
        finally:
            engine.close()
    if not (data / "measurements.jsonl").exists():
        raise FileNotFoundError(f"no measurements in {data}; run setup without --skip-measure first")

    log("[2/3] learning from the measurements")
    hardware = load_hardware(data / "hardware.json")
    tables, failed = build_tables(load_measurements(data / "measurements.jsonl"))
    states = build_states(tables, hardware)
    tr_rows, va_rows, te_rows = split(states)
    if not tr_rows or not va_rows or not te_rows:
        raise ValueError("not enough measurements to train and test; measure more sizes")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tr, va, te = tensors(tr_rows, device), tensors(va_rows, device), tensors(te_rows, device)
    log(f"      {sum(len(t) for t in tables.values())} correct runs, {sum(len(f) for f in failed.values())} "
        f"failed or wrong -> {len(states)} states (train {len(tr_rows)}, val {len(va_rows)}, "
        f"test {len(te_rows)} on held-out sizes); training on {device}")
    net, info = train_one(size, tr, va, device)

    _, _, vout = evaluate(net, va)
    temp = fit_temperature(vout["logits"], va["good"])
    net.temperature.fill_(temp)
    _, acc, tout = evaluate(net, te)
    conf, pred = F.softmax(tout["logits"] / temp, 1).max(1)
    correct = te["good"].gather(1, pred.unsqueeze(1)).squeeze(1)
    ece, _ = reliability(conf.cpu(), correct.cpu())

    log("[3/3] fitting the familiarity guard and saving")
    guard = fit_guard(tr_rows, va_rows, hardware)
    checks = guard.check([r["state"] for r in te_rows])
    modes = [decision_mode(c, k["familiar"], DEFAULT_THRESHOLD, DEFAULT_ESCALATE_BELOW)
             for c, k in zip(conf.tolist(), checks)]
    auto = torch.tensor([m == "auto" for m in modes], device=correct.device)
    report = {
        "machine": hardware.get("name"), "home": str(home), "quick": quick,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "measurements": sum(len(t) for t in tables.values()) + sum(len(f) for f in failed.values()),
        "states": len(states), "model_size": size, "parameters": info["parameters"],
        "test_accuracy": round(acc, 4), "ece": ece, "temperature": round(temp, 4),
        "modes_on_test": {m: round(modes.count(m) / len(modes), 4) for m in ("auto", "confirm", "escalate")},
        "auto_accuracy": round(correct[auto].float().mean().item(), 4) if auto.any() else None,
        "minutes": round((time.time() - t0) / 60, 1),
    }
    meta = {"size": size, "trained_at": report["trained_at"], "trained_on": hardware.get("name"),
            "setup": True, "quick": quick,
            "test_metrics": {"accuracy": report["test_accuracy"], "ece": ece, "temperature": report["temperature"]}}
    save_checkpoint(ckpt, net, meta, guard)
    (home / "setup_report.json").write_text(json.dumps(report, indent=2))
    log(f"      picks a best move {acc:.1%} on held-out sizes, calibration error {ece:.3f}; "
        f"auto-runs {report['modes_on_test']['auto']:.0%} of decisions"
        + (f" at {report['auto_accuracy']:.0%} accuracy" if report["auto_accuracy"] is not None else ""))
    log(f"saved {ckpt} ({report['minutes']} min). hpc_ds1.load() now uses this machine's model.")
    return report
