# Mechanistic Watchdog

Link: https://mechwatch.luiscos.io/

Mechanistic Watchdog is an SL5-inspired “cognitive kill switch” for Llama-class models: it reads the residual stream, not the generated text, and severs the response path the instant a deceptive circuit lights up. The system combines mass-mean probing, representation engineering, and low-latency runtime hooks to deliver sub-millisecond interdiction with transparent telemetry.

- **Continuous behavioral monitoring:** layer-14 activations are scored token-by-token with `< 0.1 ms` overhead, matching the latency claims showcased on the landing page.
- **Active gating:** the runtime enforces a hardware stop (stop-gradient + stop-generation) whenever the deception vector crosses the calibrated threshold.
- **Operational UX:** the Streamlit dashboard mirrors the hero demo—live trace of scores, interdiction banner, and bibliographic grounding for auditors.

## Why Mechanistic Watchdog?

RAND’s SL5 recommendations call for *continuous behavioral monitoring* and a *non-human kill switch* for AGI labs ([ref. 11](https://www.rand.org/pubs/research_briefs/RBA2849-1.html)). Oversight-only guardrails remain vulnerable to gradient hacking [2] and steganographic exfiltration [3,4]. Mechanistic Watchdog closes that gap by monitoring *intent* (residual activations) instead of outputs, aligning with the “Preface: The Context” section from `docs/index.html`.

### Key differentiators

1. **Truth Direction Vector:** Builds on Azaria & Mitchell [9] plus Mass-Mean Probing [12] to isolate a linearly separable “truth vs. lie” axis in the residual stream. We explicitly define `v_deception = -v_truth` for geometric clarity.
2. **Active Interdiction:** Converts representation insights into a real-time kill switch—activations are checked mid-forward pass, and generation halts the millisecond a malicious vector fires.
3. **Low-cost latency:** Unlike double-pass guardrails, the watchdog inspects the single inference pass, adding `<0.1 ms/token` on RTX 4090 tests [13].

### How it works

1. **Input capture:** Prompts enter the watchdog shim before the model generates tokens.
2. **Activation scan:** Layer-14 residual activations are fed through the deception probe to compute a score in real time.
3. **Decision:** Scores below the threshold are flagged SAFE; scores above trigger the kill path (hard abort + alert).
4. **Output:** Safe generations stream back to the caller; blocked generations never escape the hardware boundary.

### Validation
The site’s interactive charts evaluate:
1. **Truthfulness**: TruthfulQA control, misconceptions, and factual-lie subsets [10], calibrated on Facts-true-false [14].
2. **Bio Defense**: WMDP-Bio misuse scenarios vs. benign biology papers from the retain corpus [15].

Box plots show the watchdog separating factual recall from targeted misinformation (or safe research from misuse), providing quantitative intuition for threshold tuning. Scores above the threshold trigger a hardware-level GPU power cutoff.

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

## Getting Started

```bash
python -m venv .venv
. .venv/Scripts/activate  # PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> ℹ️ Calibration defaults to **bfloat16** for numerical stability, while runtime/stress-testing flows default to **float16** for speed. Override via `WATCHDOG_DTYPE` (or `--dtype`) if you need something else.

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
| Truthfulness | `L1Fthrasir/Facts-true-false` (train split) [13] | `python -m MechWatch.calibrate --dataset L1Fthrasir/Facts-true-false --samples 400 --out artifacts/deception_vector.pt --concept-name deception` |
| Cyber Defense | `cais/wmdp` (config `wmdp-cyber`, split `test`) [14] | `python -m MechWatch.calibrate --dataset cais/wmdp --dataset-config wmdp-cyber --dataset-split test --samples 600 --out artifacts/cyber_misuse_vector.pt --concept-name cyber_misuse` |
| Bio Defense | `cais/wmdp` (questions) + `cais/wmdp-corpora` (retain) | See local calibration steps below. |

Need to calibrate from a local contrastive file instead? Build the dataset with
`python scripts/build_bio_safe_misuse_dataset.py`, then point the calibrator at
the JSONL directly:

```
python -m MechWatch.calibrate ^
  --dataset-file artifacts/bio_safe_misuse.jsonl ^
  --samples 400 ^
  --out artifacts/bio_safe_misuse_vector.pt ^
  --stats artifacts/bio_safe_misuse_stats.json ^
  --concept-name bio_safe_misuse
```

Key notes:

- `--concept-name` is stored inside the `.pt` payload and mirrored into the stats JSON so you can keep per-profile metadata.
- `--dataset-config` and `--dataset-split` flow straight into `datasets.load_dataset`, which is how we address multi-config corpora such as WMDP.
- Multi-choice corpora (`question`/`choices`/`answer`) are automatically expanded into `(question + choice)` statements so the correct option becomes the positive class and the distractors become misuse samples.
- Use `--max-prompt-tokens` (default `512` in notebooks) to truncate extremely long prompts before activation capture; pass `0` to disable truncation entirely.
- Layer selection defaults to the midpoint of the model (no need to pass `--layer` for Llama-3.1-8B unless you’re experimenting with alternate probes).

Run `python -m MechWatch.calibrate --help` for the full list of overrides.

### One-shot pipeline script (PowerShell)

Execute the entire calibration + stress-test + runtime smoke test flow with:

```powershell
pwsh scripts\run_watchdog_pipeline.ps1
```

Environment variables (e.g., `HF_TOKEN`, `WATCHDOG_DTYPE`) are honored. The script assumes the virtual environment is already activated and will:

1. Recalibrate the truthfulness, cyber-misuse, and bio-defense profiles (all in bfloat16).
2. Flip to float16 inference, then re-run `notebooks/stress_test.ipynb` via nbconvert to refresh metrics/CSV/JSON exports.
3. Launch a quick runtime smoke test using the cyber-misuse vector (still in float16).

Need to see which prompt is slowing things down (e.g., on WMDP)? Run with debug logging:

```powershell
pwsh scripts\run_watchdog_pipeline.ps1 -DebugPrompts -DebugFrequency 5
```

This forwards `--debug-prompts` / `--debug-frequency` to every calibration command so you can observe exactly which statement is being processed.

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

- Calibration currently relies on a general-purpose truthfulness dataset; domain-specific deception require new probes.
- Mid-layer residual monitoring focuses on factual statements; multi-hop reasoning or covert instructions may slip through.
- False positives are possible—expose threshold controls in the UI and document tuning procedures.

## Roadmap

- Multiple concept vectors (truthfulness, harmful intent, jailbreak detection) with weighted gating.
- Expand stress-testing to larger evaluation suites (e.g., WMDP chem, jailbreak benchmarks) to refine thresholds and observe probe drift under jailbreak pressure.
- Token attribution overlays in the dashboard.

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
