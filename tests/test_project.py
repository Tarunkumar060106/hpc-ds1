"""Fast checks: config space, real kernel decomposition, labelling, the API.

    python -m pytest tests -q
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hpc_ds1.dataset import build_states, label
from hpc_ds1.kernels import KERNELS
from hpc_ds1.runner import Engine
from hpc_ds1.space import ACTIONS, STOP, all_configs, apply_action, config_key, in_space, make_config, valid_actions

ROOT = Path(__file__).resolve().parent.parent
SMALL = {"gemm": 256, "fft": 32, "stencil": 32, "spmv": 4000, "nbody": 512, "montecarlo": 200_000,
         "mltrain": 256, "server": 64, "llm_train": 4, "cnn_train": 8}


def test_space_is_closed_under_valid_actions():
    for c in all_configs():
        for a in valid_actions(c):
            n = apply_action(c, a)
            assert in_space(n), (c, a, n)
    assert len(all_configs()) == 44
    assert "USE_GPU" not in valid_actions(make_config(), has_gpu=False)


def test_apply_action_examples():
    c = make_config(ranks=2, threads=4)
    assert apply_action(c, "ADD_RANKS") == make_config(4, 4)
    assert apply_action(c, "USE_GPU") == make_config(1, 4, True)
    assert apply_action(make_config(1, 1), "REMOVE_RANKS") is None
    assert apply_action(c, STOP) == c


@pytest.mark.parametrize("workload", list(KERNELS))
def test_rank_decomposition_matches_single_rank(workload):
    k = KERNELS[workload]
    size = SMALL[workload]

    def run(nshards):
        parts = []
        for s in range(nshards):
            inp = k.setup(size, s, nshards, "cpu", torch.float32)
            parts.append(k.run(inp))
        return k.combine(parts, size).numpy()

    one, four = run(1), run(4)
    err = np.linalg.norm(four - one) / np.linalg.norm(one)
    assert err <= k.tolerance, err


def test_engine_measures_real_runs():
    eng = Engine()
    try:
        r1 = eng.measure("gemm", 256, {"ranks": 1, "threads": 2}, min_total=0.0)
        r2 = eng.measure("gemm", 256, {"ranks": 2, "threads": 1}, min_total=0.0)
    finally:
        eng.close()
    assert r1["ok"] and r2["ok"]
    assert r1["telemetry"]["runtime_sec"] > 0
    assert np.allclose(r1["result"], r2["result"], rtol=1e-4, atol=1e-3)


def test_label_uses_measured_runtimes_and_noise_floor():
    tel = lambda rt: {"runtime_sec": rt, "noise_pct": 1.0}  # noqa: E731
    base = make_config(1, 1)
    table = {config_key(base): tel(1.0), config_key(make_config(2, 1)): tel(0.5),
             config_key(make_config(1, 2)): tel(0.6)}
    action, gain, outcomes = label(base, table)
    assert action == "ADD_RANKS" and abs(gain - 0.5) < 1e-9
    assert outcomes["USE_BF16"] is None  # never measured -> unknown, not invented
    table[config_key(make_config(2, 1))] = tel(0.97)
    table[config_key(make_config(1, 2))] = tel(0.98)
    assert label(base, table)[0] == STOP  # 3% faster is inside the 5% noise floor


@pytest.mark.skipif(not (ROOT / "hpc_ds1" / "models" / "hpc_ds1.pt").exists(), reason="train the model first")
def test_api_decides_on_real_states():
    import hpc_ds1
    from hpc_ds1.dataset import build_tables, load_hardware, load_measurements
    model = hpc_ds1.load(device="cpu")
    tables, _ = build_tables(load_measurements())
    states = build_states(dict(list(tables.items())[:1]), load_hardware())
    d = model.decide(states[0]["state"])
    assert d.choice in ACTIONS and 0 <= d.confidence <= 1
    assert abs(sum(d.probabilities.values()) - 1) < 1e-3
    d2 = model.decide(states[0]["state"], ["STOP_TUNING", "ADD_THREADS"])
    assert set(d2.probabilities) <= {"STOP_TUNING", "ADD_THREADS"}
    t = model.trace(states[0]["state"])
    assert set(t["layers"]) >= {c["id"] for c in model.layer_catalog()}
    json.dumps(t)


def _real_states(workloads=None):
    from hpc_ds1.dataset import build_tables, load_hardware, load_measurements
    hw = load_hardware()
    tables, _ = build_tables(load_measurements())
    if workloads:
        tables = {k: v for k, v in tables.items() if k[0] in workloads}
    return build_states(tables, hw), hw


def test_decision_tiers():
    from hpc_ds1.api import decision_mode
    assert decision_mode(0.9, True, 0.85, 0.5) == "auto"
    assert decision_mode(0.7, True, 0.85, 0.5) == "confirm"
    assert decision_mode(0.3, True, 0.85, 0.5) == "escalate"
    assert decision_mode(0.99, False, 0.85, 0.5) == "escalate"  # unfamiliar is never automatic


def test_guard_flags_unseen_kernel_size_and_machine():
    from hpc_ds1.guard import FamiliarityGuard
    states, hw = _real_states()
    rest = [s["state"] for s in states if s["workload"] != "fft"]
    guard = FamiliarityGuard.fit(rest[::2], rest[1::2], hw)
    assert sum(c["familiar"] for c in guard.check(rest[1::2])) >= 0.95 * len(rest[1::2])
    fft = [s["state"] for s in states if s["workload"] == "fft"][:20]
    assert not any(c["familiar"] for c in guard.check(fft))
    big = dict(rest[0], workload={"name": rest[0]["workload"]["name"], "size": 10 ** 9})
    assert any("outside the measured range" in r for r in guard.check([big])[0]["reasons"])
    other = dict(rest[0], hardware={**rest[0]["hardware"], "cpu_threads": 128, "cpu_cores": 64})
    assert any("different machine" in r for r in guard.check([other])[0]["reasons"])
    assert FamiliarityGuard.from_dict(json.loads(json.dumps(guard.to_dict()))).check(fft)[0]["familiar"] is False


def test_tuner_asks_human_when_policy_says_so():
    from hpc_ds1.tuner import TableRunner, run_tuning
    states, hw = _real_states(["stencil"])
    inst = ("stencil", states[0]["size"])
    from hpc_ds1.dataset import build_tables, load_measurements
    table = build_tables(load_measurements())[0][inst]
    asked = []
    policy = lambda state, exclude=(): ("ADD_THREADS", 0.99, 0, True)  # noqa: E731  confident but needs a human
    r = run_tuning(policy, *inst, make_config(1, 1), TableRunner(table, hw), threshold=0.85,
                   on_escalate=lambda a, c, s: asked.append(a) or False)
    assert asked == ["ADD_THREADS"] and r.stopped_reason == "escalation_declined"


def test_setup_trains_this_machines_model(tmp_path, monkeypatch):
    from hpc_ds1.dataset import data_dir
    from hpc_ds1.onboard import setup
    src = data_dir()
    data = tmp_path / "data"
    data.mkdir()
    (data / "hardware.json").write_text((src / "hardware.json").read_text())
    rows = [line for line in (src / "measurements.jsonl").read_text().splitlines()
            if json.loads(line)["workload"] in ("stencil", "gemm")]
    (data / "measurements.jsonl").write_text("\n".join(rows) + "\n")
    report = setup(tmp_path, collect_data=False, size="tiny", device="cpu", log=lambda m: None)
    assert (tmp_path / "model" / "hpc_ds1.pt").exists() and report["test_accuracy"] > 0.5
    fft = [s["state"] for s in _real_states(["fft"])[0]][:1]
    monkeypatch.setenv("HPC_DS1_HOME", str(tmp_path))  # load() now prefers this machine's own model
    import hpc_ds1
    model = hpc_ds1.load(device="cpu")
    assert model.guard is not None and set(model.guard.workloads) == {"stencil", "gemm"}
    d = model.decide(fft[0])
    assert d.mode == "escalate" and not d.familiar


def test_cli_parses():
    import subprocess
    import sys
    out = subprocess.run([sys.executable, "-m", "hpc_ds1", "--help"], capture_output=True, text=True, cwd=ROOT)
    assert out.returncode == 0 and "setup" in out.stdout


def test_fleet_places_on_earliest_finish_and_audits(tmp_path):
    from hpc_ds1.facility import AuditLog
    from hpc_ds1.fleet import Fleet, estimate_runtime
    audit = AuditLog(tmp_path / "audit.jsonl")
    fleet = Fleet(audit=audit)
    fast = fleet.register("gpu-box", {"name": "a"}, {"gemm": {"2048": 0.01}}, {})
    slow = fleet.register("old-laptop", {"name": "b"}, {"gemm": {"2048": 0.5}}, {})
    fleet.register("new-box", {"name": "c"}, {}, {})  # nothing measured: no estimate
    assert abs(estimate_runtime(fast["profile"], "gemm", 4096) - 0.08) < 1e-9  # work scales as n^3
    t = fleet.submit("gemm", 2048)
    assert t["device"] == fast["id"]
    assert any(c["note"] for c in t["placement"]["candidates"])
    for _ in range(60):  # fill the fast device's queue until the slow one finishes earlier
        if fleet.submit("gemm", 2048)["device"] == slow["id"]:
            break
    else:
        raise AssertionError("queue time never moved work to the other device")
    got = fleet.poll(fast["id"])
    state = {"workload": {"name": "gemm", "size": 2048}, "config": got["config"], "telemetry": {"runtime_sec": 0.2}}
    dec = {"choice": "USE_GPU", "confidence": 0.7, "mode": "confirm", "reasons": []}
    fleet.update(got["id"], fast["id"], {"event": {"decision": dec, "state": state}})
    fleet.update(got["id"], fast["id"], {"pending": {"action": "USE_GPU", "confidence": 0.7, "mode": "confirm"}})
    assert fleet.queue()[0]["task"] == got["id"]
    fleet.approve(got["id"], True, "alice")
    assert fleet.update(got["id"], fast["id"], {})["answer"]["approve"] is True
    row = audit.recent()[0]
    assert row["choice"] == "USE_GPU" and row["approved_by"] == "alice"
    assert (tmp_path / "audit.jsonl").exists()


def test_server_workload_has_no_gpu_or_bf16_version():
    k = KERNELS["server"]
    with pytest.raises(RuntimeError):
        k.setup(64, 0, 1, "cuda", torch.float32)
    with pytest.raises(RuntimeError):
        k.setup(64, 0, 1, "cpu", torch.bfloat16)


def test_model_inputs_follow_its_own_workload_list():
    from hpc_ds1.features import featurize
    from hpc_ds1.net import HPCDS1Net
    state = {"hardware": {"cpu_threads": 16}, "workload": {"name": "brand_new_task", "size": 1000},
             "config": {"ranks": 1, "threads": 1, "gpu": False, "precision": "fp32"},
             "telemetry": {"runtime_sec": 0.1}}
    net = HPCDS1Net(dim=32, encoder_blocks=1, fusion_layers=1, heads=4, ffn_dim=64, workloads=["gemm", "fft"])
    f = featurize([state], workloads=net.workloads)
    assert f["workload"].shape[1] == 2 + 2 and f["workload"][0, 2:].sum() == 0  # unknown task: no identity
    out = net(f)
    assert out["logits"].shape[1] == len(ACTIONS)


def test_plugin_task_is_discovered_from_home_folder(tmp_path):
    import shutil
    import subprocess
    import sys
    (tmp_path / "plugins").mkdir()
    shutil.copy(ROOT / "examples" / "plugins" / "etl_batch.py", tmp_path / "plugins")
    code = ("from hpc_ds1.kernels import KERNELS, WORKLOAD_NAMES; import numpy as np, torch; k = KERNELS['etl']; "
            "one = k.combine([k.run(k.setup(5000, 0, 1, 'cpu', torch.float32))], 5000); "
            "four = k.combine([k.run(k.setup(5000, s, 4, 'cpu', torch.float32)) for s in range(4)], 5000); "
            "print(WORKLOAD_NAMES[-1], float((one - four).abs().max()))")
    env = {**__import__("os").environ, "HPC_DS1_HOME": str(tmp_path)}
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=ROOT, env=env)
    assert out.returncode == 0, out.stderr
    name, err = out.stdout.split()
    assert name == "etl" and float(err) < 1e-3


def test_system1_predicts_a_valid_config_without_running(tmp_path):
    from hpc_ds1.dataset import load_hardware, load_measurements
    from hpc_ds1.space import in_space
    from hpc_ds1.system1 import Calibrator, System1, rows_from_measurements, size_ranges, train_member
    from hpc_ds1.features import _hw
    rows = [r for r in load_measurements() if r["workload"] in ("stencil", "gemm")]
    hw = load_hardware()
    x, y, f, g = rows_from_measurements(rows, hw)
    members = [train_member(x, y, f, g, seed, "cpu", {"epochs": 60}) for seed in (0, 1)]
    s1 = System1(members, __import__("hpc_ds1").WORKLOAD_NAMES, Calibrator(), size_ranges(rows), _hw(hw), hw["name"])
    torch.save(s1.to_checkpoint(), tmp_path / "s1.pt")
    s1 = System1.load(tmp_path / "s1.pt")
    p = s1.predict(hw, "stencil", 160)
    assert in_space(p.config, True) and 0 <= p.confidence <= 1 and p.expected_runtime_sec > 0 and p.familiar
    far = s1.predict(hw, "stencil", 10_000)
    assert far.mode == "escalate" and any("outside the measured range" in r for r in far.reasons)
    other = s1.predict({**hw, "cpu_threads": 128, "cpu_cores": 64}, "gemm", 2048)
    assert any("different machine" in r for r in other.reasons)


def test_dual_process_keeps_a_confirmed_system1_pick():
    from hpc_ds1.system1 import gated
    calls = []

    def system2(state, exclude=()):
        calls.append(state)
        return "ADD_THREADS", 0.9, 0, False
    as_predicted = {"telemetry": {"runtime_sec": 0.01}}
    assert gated(system2, lambda s: True)(as_predicted)[0] == "STOP_TUNING" and not calls
    assert gated(system2, lambda s: False)(as_predicted)[0] == "ADD_THREADS"  # pick didn't hold up: System 2
    assert gated(system2, None)(as_predicted)[0] == "ADD_THREADS"              # System 1 unsure: System 2


def test_live_runs_are_logged_and_learned(tmp_path, monkeypatch):
    """Level 5: a real run made through the tuner becomes training data for System 1."""
    from hpc_ds1.learn import learn, load_live_runs
    from hpc_ds1.tuner import LiveRunner
    monkeypatch.setenv("HPC_DS1_HOME", str(tmp_path))
    monkeypatch.setenv("HPC_DS1_LOG_LIVE", "1")
    eng = Engine()
    try:
        LiveRunner("stencil", 96, eng)({"ranks": 1, "threads": 2, "gpu": False, "precision": "fp32"})
    finally:
        eng.close()
    live = load_live_runs(eng.hardware["name"])
    assert len(live) == 1 and live[0]["source"] == "live" and live[0]["correct"]

    class Prev:
        hp = {"members": 2, "epochs": 30}
        from hpc_ds1.system1 import Calibrator
        calibrator = Calibrator()
    report = learn(Prev(), device="cpu", log=lambda m: None)
    assert report["live_runs"] == 1 and (tmp_path / "model" / "system1.pt").exists()


def test_encoder_questions_are_answered_from_measurements():
    """Every generated question's answer must agree with the measured runtimes."""
    from hpc_ds1.dataset import build_tables, load_hardware, load_measurements
    from hpc_ds1.encoder import NOUL, build_examples, describe
    hw = load_hardware()
    tables, failed = build_tables(load_measurements())
    tables = {k: v for k, v in tables.items() if k[0] == "stencil"}
    failed = {k: v for k, v in failed.items() if k[0] == "stencil"}
    states = build_states(tables, hw)
    exs = build_examples(states, tables, failed, hw, per_state=7, seed=0)
    assert exs and all(0 <= e.answer < len(e.options) for e in exs)
    assert any(e.held_out_phrasing for e in exs) and any(not e.held_out_phrasing for e in exs)
    for e in exs:
        if e.family == "device":  # stencil's best measured config is on the GPU
            assert e.options[e.answer] == "the GPU"
        if e.family in ("action_helps", "near_best", "fails"):
            assert e.options == NOUL
    assert "Measured: runtime" in describe(states[0]["state"])


def test_change_questions_are_grounded_in_system1_estimates():
    from hpc_ds1.encoder import detect_change, grounded_answer
    assert detect_change("Would the graphics card make this faster?") == "USE_GPU"
    assert detect_change("Will doubling the processes speed this server batch up?") == "ADD_RANKS"
    assert detect_change("Would more threads per process help?") == "ADD_THREADS"
    assert detect_change("Should this job use fp32 or bf16?") is None          # not about one change's effect
    assert detect_change("Is this already about as fast as it gets?") is None
    hints = {"USE_GPU": 0.09, "ADD_RANKS": 1.4, "ADD_THREADS": 1.01}
    assert grounded_answer("Would the graphics card make this faster?", hints)[0] == "yes"
    assert grounded_answer("Will adding more workers make it quicker?", hints)[0] == "no"
    assert grounded_answer("Would more threads per process help?", hints)[0] == "unclear"
    assert grounded_answer("Would moving to the GPU make it slower?", hints)[0] == "no"   # inverted question
    assert grounded_answer("Would the graphics card help?", None) is None              # no estimate: encoder answers


def _device(fleet, name, gpu, mem, profile, peaks=None):
    return fleet.register(name, {"name": name, "gpu_model": gpu, "gpu_memory_gb": mem}, profile, {},
                          {"gpu_peak_gb": peaks or {}, "failed": {}})


def test_fleet_sends_a_job_that_does_not_fit_to_a_bigger_gpu():
    from hpc_ds1.fleet import Fleet
    f = Fleet()
    small = _device(f, "laptop", "RTX 4050 Laptop GPU", 6.0, {"llm_train": {"16": 0.1, "96": 1.2}},
                    {"llm_train": {"16": 1.3, "96": 7.4}})
    big = _device(f, "lab", "RTX 5090", 31.8, {})
    fits = f.submit("llm_train", 16)
    assert fits["device"] == small["id"]  # fits and is the only measured device: stays on the laptop
    t = f.submit("llm_train", 96)
    assert t["device"] == big["id"]
    no = next(c for c in t["placement"]["candidates"] if c["device"] == small["id"])
    assert not no["can_run"] and "7.4 GB" in no["why_not"] and "6 GB" in no["why_not"]
    assert "laptop can't" in t["placement"]["reason"]
    assert f.submit("llm_train", 128)["device"] == big["id"]  # unmeasured size: need fitted from the measured ones


def test_fleet_requeues_a_task_that_failed_on_a_device():
    from hpc_ds1.fleet import Fleet
    f = Fleet()
    a = _device(f, "a", "GPU A", 8.0, {"gemm": {"2048": 0.001}})
    b = _device(f, "b", "GPU B", 8.0, {"gemm": {"2048": 0.002}})
    t = f.submit("gemm", 2048)
    assert t["device"] == a["id"]
    f.poll(a["id"])
    f.update(t["id"], a["id"], {"status": "failed", "error": "CUDA out of memory"})
    assert t["status"] == "queued" and t["device"] == b["id"]
    assert t["redirects"][0]["from"] == "a" and "out of memory" in t["redirects"][0]["reason"]
    assert f.submit("gemm", 2048)["device"] == b["id"]  # a is remembered as unable to run it
    f.poll(b["id"])
    f.update(t["id"], b["id"], {"status": "failed", "error": "crash"})
    assert t["status"] == "failed"  # nowhere left to go


def test_collection_skips_slow_cpu_settings_from_a_real_probe():
    from hpc_ds1.collect import _cpu_skip

    class E:
        hardware = {"cpu_threads": 16}
    k = KERNELS["llm_train"]
    cpu = {"ranks": 1, "threads": 2, "gpu": False, "precision": "bf16"}
    assert _cpu_skip(E, k, 4, cpu, {}) is None  # nothing measured yet: measure it
    assert "budget" in _cpu_skip(E, k, 4, cpu, {"bf16": (4, 70.0)})
    assert _cpu_skip(E, k, 4, {**cpu, "gpu": True}, {"bf16": (4, 70.0)}) is None  # GPU settings are always measured
    assert _cpu_skip(E, KERNELS["gemm"], 1024, cpu, {"bf16": (1024, 70.0)}) is None  # only GPU-class workloads


def test_usable_gpu_memory_is_where_measured_runs_start_spilling():
    from hpc_ds1.fleet import can_run, usable_gpu_memory
    k = KERNELS["llm_train"]

    def run(size, sec, peak):
        return {"workload": "llm_train", "size": size, "ok": True, "correct": True,
                "config": {"ranks": 1, "threads": 1, "gpu": True, "precision": "bf16"},
                "telemetry": {"runtime_sec": sec, "gpu_peak_gb": peak}}
    rows = [run(16, 0.085, 1.33), run(32, 0.167, 2.54), run(48, 0.718, 3.74), run(64, 1.94, 4.94)]
    assert usable_gpu_memory(rows) == 3.74  # 48 is >2x slower per sequence than 16 and 32
    assert usable_gpu_memory(rows[:2]) is None  # never slowed down: no limit learned
    dev = {"name": "laptop", "hardware": {"gpu_model": "RTX 4050", "gpu_memory_gb": 6.0},
           "limits": {"gpu_usable_gb": 3.74}}
    assert can_run(dev, "llm_train", 32, 2.54)[0]
    ok, why = can_run(dev, "llm_train", 48, 3.74)
    assert not ok and "slow down sharply from 3.7 GB" in why
    assert k.gpu_class


@pytest.mark.parametrize("page", ["console.html", "index.html"])
def test_ui_pages_have_valid_javascript(page, tmp_path):
    import re
    import shutil
    import subprocess
    from hpc_ds1.paths import UI_DIR
    if shutil.which("node") is None:
        pytest.skip("node is not installed")
    js = "\n".join(re.findall(r"<script>(.*?)</script>", (UI_DIR / page).read_text(encoding="utf-8"), re.S))
    f = tmp_path / "page.js"
    f.write_text(js, encoding="utf-8")
    out = subprocess.run(["node", "--check", str(f)], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr[:400]
