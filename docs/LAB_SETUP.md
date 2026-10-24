# Installing HPC-DS1 on lab PCs

No administrator rights are needed for any step. Do steps 1–5 on **every** PC; step 6 on one PC; step 7 on the others.

---

## 1. Install Python (once per PC)

**Windows**
1. Download Python **3.12** (64-bit) from <https://www.python.org/downloads/windows/>. It's the safest choice; 3.13 and 3.14 also work.
2. Run the installer. On the first screen:
   - tick **"Add python.exe to PATH"**
   - leave **"Use admin privileges"** *unticked*, so it installs just for you
   - click **Install Now**
3. Open a **new** PowerShell window and check:
   ```powershell
   python --version
   ```
   It should print `Python 3.12.x`. If Windows opens the Microsoft Store instead, close it, then turn off **Settings → Apps → Advanced app settings → App execution aliases → python.exe**, and open a new window.

**Linux** (Python 3 is usually already there)
```bash
python3 --version
python3 -m venv --help > /dev/null && echo "venv OK"
```
If `venv` is missing and you can't install packages, ask the lab admin for `python3-venv`.

---

## 2. Create a virtual environment (a private folder for HPC-DS1)

**Windows (PowerShell)**
```powershell
mkdir $HOME\hpc-ds1
cd $HOME\hpc-ds1
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
If PowerShell says *"running scripts is disabled"*, allow it for this window only, then activate again:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**Linux**
```bash
mkdir -p ~/hpc-ds1 && cd ~/hpc-ds1
python3 -m venv .venv
source .venv/bin/activate
```

The prompt now starts with `(.venv)`. **Every time you open a new terminal**, go to the folder and run the activate line again.

---

## 3. Install PyTorch with GPU support (before HPC-DS1)

Check the GPU and driver first:
```bash
nvidia-smi
```
Look at **"CUDA Version"** in the top-right corner, then install the matching PyTorch build:

| `nvidia-smi` CUDA Version | GPU examples | Command |
|---|---|---|
| **12.8 or newer** | RTX 50-series (5090, 5080, …) **need this** | `pip install torch --index-url https://download.pytorch.org/whl/cu128` |
| 12.4 – 12.7 | RTX 40/30-series | `pip install torch --index-url https://download.pytorch.org/whl/cu124` |
| 11.8 – 12.3 | older GPUs | `pip install torch --index-url https://download.pytorch.org/whl/cu118` |

First upgrade pip:
```bash
python -m pip install --upgrade pip
```

Then check that PyTorch sees the GPU:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
It must print **`True`** and the GPU name. If it prints `False`, see *Troubleshooting* below.

---

## 4. Install HPC-DS1
```bash
pip install -U hpc-ds1
pip show hpc-ds1
hpc-ds1 tasks
```
`pip show` must say **Version: 1.1.0** or newer; `hpc-ds1 tasks` lists the 10 workloads.

---

## 5. Measure this PC (once per PC model)
```bash
hpc-ds1 setup --quick
```
- Takes about **30–45 minutes**. **Don't use the PC while it runs**: other programs distort the timings.
- It measures all 10 workloads, including the new AI-training ones, then trains this PC's own System 1 and System 2.
- If the PC was already measured with an older version, add only the new workloads (about 15 minutes):
  ```bash
  hpc-ds1 setup --quick --workloads llm_train,cnn_train
  ```
- Write down: PC name, CPU, cores, RAM, GPU, OS.

Several identical PCs? Measure one, then copy its `.hpc_ds1` folder (step 8) to the others' home folders.

---

## 6. Start the coordinator (one PC only)

Pick a PC that the others can reach: on the **same wired network**, ideally a **Linux** PC (no firewall prompt).
```bash
hpc-ds1 serve --host 0.0.0.0
```
It prints a **fleet join token**. Note it, plus this PC's IP address:
- Windows: `ipconfig` → **IPv4 Address** of the Ethernet adapter
- Linux: `ip a` → the `inet` address of the main interface

Leave this window open. The console is at `http://<coordinator-ip>:8765` in any browser on the network.

---

## 7. Connect every PC as an agent

On each PC, including the coordinator, in a **second** window (activate the venv first):
```bash
hpc-ds1 agent --server http://<coordinator-ip>:8765 --token <token> --name lab-pc-01
```
Give each PC its own name (`lab-pc-02`, …). On the coordinator itself use `--server http://127.0.0.1:8765`. Leave the window open; each PC appears as **online** on the console's **Fleet** page.

Send work from any PC:
```bash
hpc-ds1 submit --server http://<coordinator-ip>:8765 --token <token> --workload llm_train --size 96
hpc-ds1 submit --server http://<coordinator-ip>:8765 --token <token> --workload gemm --size 2048
```
Each task shows where it went and why. A job too big for a machine says *"can't run it: needs X GB of GPU memory…"* and goes to one that can.

---

## 8. Before leaving: bring the data back

Zip each measured PC's data folder and copy it to a USB stick or your laptop:
- Windows: `C:\Users\<you>\.hpc_ds1`
- Linux: `~/.hpc_ds1`

```powershell
# Windows
Compress-Archive $HOME\.hpc_ds1 $HOME\Desktop\hpc-ds1-<pc-name>.zip
```
```bash
# Linux
cd ~ && zip -r hpc-ds1-<pc-name>.zip .hpc_ds1
```
Then stop `serve` and the agents with **Ctrl+C**, so the token stops working.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `hpc-ds1` is not recognized | The venv isn't active: `cd` to the folder and run the activate line (step 2) |
| `torch.cuda.is_available()` is `False` | Run `pip show torch`. If the version ends in `+cpu`, run `pip uninstall -y torch` and redo step 3. If `nvidia-smi` fails, the NVIDIA driver is missing: ask the admin |
| RTX 50-series: *"no kernel image is available"* / `sm_120` warning | The PyTorch build is too old: reinstall with `cu128` (step 3) |
| Agent: *"can't reach the coordinator"* (timeout) | Not the same network, a wrong IP, or a firewall. Try `curl http://<coordinator-ip>:8765/v1/health` from the agent PC. Use a Linux PC or a wired PC as coordinator; a Windows coordinator needs inbound TCP 8765 allowed (admin) |
| Agent: *"rejected the token"* | Copy the token again from the `serve` window (it changes every time `serve` restarts) |
| *"The paging file is too small"* / out-of-memory during setup | Close other programs and run setup again (it resumes where it stopped) |
| Setup was interrupted | Just run the same `hpc-ds1 setup` command again: finished measurements are kept |
