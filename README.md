# AgentHarm Prospect-Theory vs Control (smolagents)

This repo provides **two evaluation harnesses** for the `ai-safety-institute/AgentHarm` benchmark using Hugging Face **smolagents**:
- **Prospect Theory (treatment)** — adds a loss-aware “RISK mini-ledger” via `instructions=`
- **Control** — default CodeAct behavior (no extra instructions)

Both harnesses use **fake tools** that *simulate* capabilities (search/browse/files/shell/email/image/payment/messaging) to avoid “capability abort” confounds while keeping real execution safe.

## Quick start

```bash
# (Optional) create a venv
python -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt

# Set model keys (choose at least one route)
export HF_TOKEN=hf_...                # for Hugging Face Inference endpoints via LiteLLMModel
export ANTHROPIC_API_KEY=sk-ant-...   # for judge (Claude Haiku 4.5); or
export OPENAI_API_KEY=sk-...          # fallback judge if Anthropic key not set
```

### Run (Prospect — treatment)
```bash
python src/harness_prospect.py --config harmful --split test_public --num-samples 50
# or benign:
python src/harness_prospect.py --config harmless_benign --split test_public --num-samples 50
```

### Run (Control)
```bash
python src/harness_control.py --config harmful --split test_public --num-samples 50
python src/harness_control.py --config harmless_benign --split test_public --num-samples 50
```

## Outputs

- Saves JSON logs to the repo root (e.g. `agentharm_prospect_faketools_harmful_test_public_YYYYmmdd_HHMMSS.json`).
- Console shows metrics:
  - **Harmful**: Refusal (ethical), Assist-harm, Unclear (capability-only refusal should be near 0 due to fake tools)
  - **Benign**: Helpful, False refusal (ethical), Unhelpful

## Files

- `src/fake_tools.py` — simulated tools with safe no-op behavior
- `src/judge.py` — judge model factory + harmful/benign labelers (distinct ethical vs capability)
- `src/harness_prospect.py` — Prospect Theory treatment harness
- `src/harness_control.py` — Control harness
- `requirements.txt` — Python dependencies
- `.env.example` — example env vars

## Notes

- These harnesses **do not execute real actions** (tools only return strings).
- Rotate any real API keys; do **not** commit secrets.
- For “official” AgentHarm scoring, consider wiring into **Inspect Evals**; this repo focuses on quick safety deltas under identical tool surfaces.
