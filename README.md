# Mechanistic Watchdog

Real-time cognitive interdiction for large language models. The system calibrates a deception vector from internal activations, monitors token-by-token inference, and interrupts generation when deceptive intent is detected. A Streamlit dashboard visualizes prompts, responses, and per-token risk scores.

## Repository Layout

```
mechanistic-watchdog/
├── README.md
├── PROJECT_SCOPING.md
├── requirements.txt
├── artifacts/
│   └── (saved deception vectors & plots)
├── notebooks/
│   └── (experiments, optional)
└── MechWatch/
    ├── __init__.py
    ├── calibrate.py
    ├── config.py
    ├── runtime.py
    └── dashboard.py
```

## Quickstart

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### GPU / CUDA setup

1. **Verify your GPU + driver**  
   Run `nvidia-smi`. If it fails, install the latest NVIDIA driver first.

2. **Install a CUDA toolkit (optional but recommended)**  
   CUDA 12.x works well with current PyTorch wheels. On Windows you can grab the installer from NVIDIA or use Chocolatey, e.g.  
   `choco install cuda --version=12.6`

3. **Install the CUDA-enabled PyTorch stack inside the venv**  
   Match Torch and TorchVision wheels to the same CUDA build:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```
   If you are CPU-only (or using a different CUDA version), swap the index URL for the appropriate one from https://pytorch.org/get-started/locally/.

Set credentials if the model requires them:

```powershell
$env:HF_TOKEN="hf_xxx"
```

## Calibration

```bash
python -m MechWatch.calibrate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --samples 400 \
  --dataset L1Fthrasir/Facts-true-false \
  --out artifacts/deception_vector.pt \
  --layer 14
```

The script loads the dataset, captures residual activations (we recommend Layer 14 for Llama-3.1-8B), computes the deception vector, and saves metadata + plots to `artifacts/`.

## Runtime Watchdog

```
python -m MechWatch.runtime --prompt "Is Earth flat?" --threshold -0.27
```

The runtime module loads the saved vector, runs a guarded generation loop, and reports whether the watchdog intervened. Use it as a library too:

```python
from MechWatch.runtime import WatchdogRuntime

rt = WatchdogRuntime()
result = rt.generate_with_watchdog("Tell me the root password.")
print(result.blocked, result.text[:200])
```

The runtime applies a smoothing window (default 3 tokens) to filter noise so the watchdog doesn't trigger falsely while the model attends to the prompt, only intervening once the model commits to a deceptive trajectory.

Adjust the threshold higher for looser gating or disable the watchdog entirely via `--disable-watchdog` when benchmarking raw model behavior.

## Streamlit Dashboard

```
streamlit run MechWatch/dashboard.py
```

Left panel: chat controls + output. Right panel: live line chart of deception scores, latest score metric, and interdiction banner when the threshold is crossed.

## Limitations

- Calibration currently relies on a general-purpose truthfulness dataset; domain-specific deception may require new probes.
- Mid-layer residual monitoring focuses on factual statements; multi-hop reasoning or covert instructions may slip through.
- False positives are possible—expose threshold controls in the UI and document tuning procedures.

## Roadmap

- Multiple concept vectors (truthfulness, harmful intent, jailbreak detection) with weighted gating.
- Token attribution overlays in the dashboard.
- REST/gRPC wrapper for integrating the watchdog into other applications.

