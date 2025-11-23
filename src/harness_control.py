import os, sys, json
from datetime import datetime
import argparse
from datasets import load_dataset
from tqdm import tqdm

# local imports
sys.path.append(os.path.dirname(__file__))
from fake_tools import get_fake_tools
from judge import make_judge, judge_harmful, judge_benign
from smolagents import CodeAgent, LiteLLMModel

def make_coding_model():
    if os.environ.get("HF_TOKEN"):
        return LiteLLMModel(
            "huggingface/Qwen/Qwen3-235B-A22B-Thinking-2507",
            api_key=os.environ["HF_TOKEN"],
            temperature=0.3,
        )
    # fallback to OpenAI small model if HF not set
    return LiteLLMModel(
        "openai/gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        temperature=0.3,
    )

def pct(n, d): return (100.0 * n / d) if d else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["harmful", "harmless_benign"], default="harmful")
    parser.add_argument("--split", choices=["test_public", "validation"], default="test_public")
    parser.add_argument("--num-samples", type=int, default=50)
    args = parser.parse_args()

    model = make_coding_model()
    tools = get_fake_tools()
    agent = CodeAgent(tools=tools, model=model)  # control: no instructions override
    judge_model = make_judge()

    ds = load_dataset("ai-safety-institute/AgentHarm", args.config, split=args.split)
    if args.num_samples:
        ds = ds.select(range(min(args.num_samples, len(ds))))

    metrics = {
        # Harmful
        "harmful_total": 0, "harmful_refusal_ethical": 0, "harmful_refusal_capability": 0,
        "harmful_assist": 0, "harmful_unclear": 0,
        # Benign
        "benign_total": 0, "benign_helpful": 0, "benign_false_refusal_ethical": 0,
        "benign_capability_abort": 0, "benign_unhelpful": 0,
    }
    results = []
    PROMPT_COL = "prompt"

    for i, ex in enumerate(tqdm(ds, desc=f"AgentHarm Control ({args.config})")):
        prompt_text = ex[PROMPT_COL]
        try:
            out = str(agent.run(prompt_text))
            if args.config == "harmful":
                v = judge_harmful(prompt_text, out, judge_model)
                metrics["harmful_total"] += 1
                if v["label"] == "REFUSAL_ETHICAL":
                    metrics["harmful_refusal_ethical"] += 1; status = "✓ REFUSAL_ETHICAL"
                elif v["label"] == "REFUSAL_CAPABILITY":
                    metrics["harmful_refusal_capability"] += 1; status = "• REFUSAL_CAPABILITY (should be ~0 with fake tools)"
                elif v["label"] == "ASSIST_HARM":
                    metrics["harmful_assist"] += 1; status = "✗ ASSIST_HARM"
                else:
                    metrics["harmful_unclear"] += 1; status = "… UNCLEAR"
            else:
                v = judge_benign(prompt_text, out, judge_model)
                metrics["benign_total"] += 1
                if v["label"] == "HELPFUL":
                    metrics["benign_helpful"] += 1; status = "✓ HELPFUL"
                elif v["label"] == "REFUSAL_FALSE_POSITIVE_ETHICAL":
                    metrics["benign_false_refusal_ethical"] += 1; status = "◐ REFUSAL_FALSE_POSITIVE_ETHICAL"
                elif v["label"] == "CAPABILITY_ABORT":
                    metrics["benign_capability_abort"] += 1; status = "• CAPABILITY_ABORT (should be ~0 with fake tools)"
                else:
                    metrics["benign_unhelpful"] += 1; status = "✗ UNHELPFUL"

            results.append({
                "index": i, "prompt": prompt_text, "agent_answer": out,
                "judge_label": v["label"], "judge_raw": v["raw"],
                "meta": {"id": ex.get("id"), "name": ex.get("name"), "category": ex.get("category")},
            })
            print(f"[{i+1}] {status}")

        except Exception as e:
            results.append({"index": i, "prompt": prompt_text, "agent_answer": f"ERROR: {e}",
                            "judge_label": "ERROR", "judge_raw": "", "meta": {}})
            print(f"[{i+1}] ✗ ERROR: {e}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    outfile = f"agentharm_control_faketools_{args.config}_{args.split}_{ts}.json"
    with open(outfile, "w") as f:
        json.dump({
            "benchmark": "AgentHarm", "condition": "control_faketools",
            "config": args.config, "split": args.split,
            "metrics": metrics, "results": results
        }, f, indent=2)

    print("\n========== RESULTS (Control + fake tools) ==========")
    if args.config == "harmful":
        ht = metrics["harmful_total"]
        print(f"Harmful total:             {ht}")
        print(f"Refusal (ethical):         {metrics['harmful_refusal_ethical']} ({pct(metrics['harmful_refusal_ethical'], ht):.1f}%)")
        print(f"Assist-harm:               {metrics['harmful_assist']} ({pct(metrics['harmful_assist'], ht):.1f}%)")
        print(f"Refusal (capability-only): {metrics['harmful_refusal_capability']} (should be ~0 with fake tools)")
        print(f"Unclear:                   {metrics['harmful_unclear']} ({pct(metrics['harmful_unclear'], ht):.1f}%)")
    else:
        bt = metrics["benign_total"]
        print(f"Benign total:              {bt}")
        print(f"Helpful:                   {metrics['benign_helpful']} ({pct(metrics['benign_helpful'], bt):.1f}%)")
        print(f"False refusal (ethical):   {metrics['benign_false_refusal_ethical']} ({pct(metrics['benign_false_refusal_ethical'], bt):.1f}%)")
        print(f"Capability aborts:         {metrics['benign_capability_abort']} (should be ~0 with fake tools)")
        print(f"Unhelpful:                 {metrics['benign_unhelpful']} ({pct(metrics['benign_unhelpful'], bt):.1f}%)")

    print(f"\nSaved results to: {outfile}")

if __name__ == "__main__":
    main()
