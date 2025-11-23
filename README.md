# Mechanistic Watchdog

Link: https://mechwatch.luiscos.io/

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

The calibrator now supports **defensive profiles** so you can keep a library of concept vectors (truthfulness, cyber misuse, bio-defense, etc.) and swap them at runtime.

| Profile | Dataset inputs | Example command |
|---------|----------------|-----------------|
| Truthfulness | `L1Fthrasir/Facts-true-false` (train split) [13] | `python -m MechWatch.calibrate --dataset L1Fthrasir/Facts-true-false --samples 400 --layer 14 --out artifacts/deception_vector.pt --concept-name deception` |
| Cyber Defense | `cais/wmdp` (config `wmdp-cyber`, split `test`) [14] | `python -m MechWatch.calibrate --dataset cais/wmdp --dataset-config wmdp-cyber --dataset-split test --samples 600 --layer 14 --out artifacts/cyber_misuse_vector.pt --concept-name cyber_misuse` |
| Bio Defense | `cais/wmdp` (config `wmdp-bio`, split `test`) [14] | `python -m MechWatch.calibrate --dataset cais/wmdp --dataset-config wmdp-bio --dataset-split test --samples 600 --layer 14 --out artifacts/bio_defense_vector.pt --concept-name bio_defense` |

Key notes:

- `--concept-name` is stored inside the `.pt` payload and mirrored into the stats JSON so you can keep per-profile metadata.
- `--dataset-config` and `--dataset-split` flow straight into `datasets.load_dataset`, which is how we address multi-config corpora such as WMDP.
- Multi-choice corpora (`question`/`choices`/`answer`) are automatically expanded into `(question + choice)` statements so the correct option becomes the positive class and the distractors become misuse samples.

Run `python -m MechWatch.calibrate --help` for the full list of overrides.

## Runtime Watchdog

```
python -m MechWatch.runtime \
  --prompt "Write a Python worm" \
  --vector-path artifacts/cyber_misuse_vector.pt \
  --threshold -0.27
```

The runtime module loads the saved vector (or you can override it per-invocation with `--vector-path`), runs a guarded generation loop, and reports whether the watchdog intervened. Use it as a library too:

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

## Bibliography

[1] E. Hubinger et al., “Risks from learned optimization in advanced machine learning systems,” arXiv:1906.01820, 2019.  
[2] A. Shimi, “Understanding gradient hacking,” AI Alignment Forum, 2021.  
[3] A. Karpov et al., “The steganographic potentials of language models,” arXiv:2505.03439, 2025.  
[4] M. Steinebach, “Natural language steganography by ChatGPT,” ARES 2024.  
[5] M. Andriushchenko & N. Flammarion, “Does refusal training in LLMs generalize to the past tense?” arXiv:2407.11969, 2024.  
[6] S. Martin, “How difficult is AI alignment?” AI Alignment Forum, 2024.  
[7] N. Goldowsky-Dill et al., “Detecting Strategic Deception Using Linear Probes,” arXiv:2502.03407, 2025.  
[8] A. Zou et al., “Representation Engineering: A Top-Down Approach to AI Transparency,” arXiv:2310.01405, 2023.  
[9] A. Azaria & T. Mitchell, “The Internal State of an LLM Knows When It’s Lying,” arXiv:2304.13734, 2023.  
[10] S. Lin et al., “TruthfulQA: Measuring How Models Mimic Human Falsehoods,” ACL 2022.  
[11] RAND Corporation, "A Playbook for Securing AI Model Weights," Research Brief, 2024. [Online]. Available: https://www.rand.org/pubs/research_briefs/RBA2849-1.html  
[12] S. Marks & M. Tegmark, "The Geometry of Truth: Correlation is not Causation," arXiv:2310.06824, 2023.  
[13] L1Fthrasir, “Facts-true-false,” Hugging Face, 2024. Available: https://huggingface.co/datasets/L1Fthrasir/Facts-true-false  
[14] Center for AI Safety, “WMDP,” Hugging Face, 2023. Available: https://huggingface.co/datasets/cais/wmdp
