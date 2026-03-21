#!/usr/bin/env python3
"""AutoResearch agent for CFT.

Runs queued experiments using the cft library, interprets results via Claude,
and suggests follow-up experiments. Designed to run in GitHub Actions on a
schedule, but works locally too.

Usage:
    python research/agent.py                      # run up to 3 experiments + interpret
    python research/agent.py --max-experiments 5  # run more
    python research/agent.py --run-only           # skip Claude interpretation
    python research/agent.py --interpret-only     # skip running, just interpret recent
"""

import argparse
import json
import os
import re
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

# Add project root to path so we can import cft
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cft import (
    SocialSimulator,
    HypothesisTester,
    CFT, GFT, QST, ICT, TST, DCT,
)

RESEARCH_DIR = Path(__file__).resolve().parent
RESULTS_DIR = RESEARCH_DIR / "results"
QUEUE_FILE = RESEARCH_DIR / "queue.yaml"
LOG_FILE = RESEARCH_DIR / "log.md"

THEORY_MAP = {
    "CFT": CFT, "GFT": GFT, "QST": QST,
    "ICT": ICT, "TST": TST, "DCT": DCT,
}

PROJECT_CONTEXT = """You are analyzing results from CFT (Consensus-Fracture Theory), an open
Python library that implements six competing theories of how social groups form and fracture.

The six theories:
- CFT: Threshold cliques - groups form when all pairwise affinities exceed threshold theta.
- GFT: Gradient field - agents move in behavioral space following affinity gradients.
- QST: Quantum-inspired superposition - agents exist in superposition of group memberships.
- ICT: Information cascade - groups form through bandwidth-limited communication rounds.
- TST: Thermodynamic (Potts model) - group assignments minimize social free energy.
- DCT: Dual-context - each agent has proximity (fast, who you're near) and alignment
  (slow, what you believe). Groups need both. Novel theory unique to this project.

Scenarios for synthetic data:
- random: uniform MBTI types, Gaussian opinions
- clustered: k distinct MBTI+opinion clusters (beta controls clustering strength)
- polarized: two opposing camps
- hierarchical: influencers + followers

Key metrics:
- NMI (Normalized Mutual Information): partition similarity, 0=random, 1=perfect match
- PAS (Prediction Accuracy Score): composite of group count + partition + size accuracy
- CTAI (Cross-Theory Agreement Index): mean pairwise NMI across all theory pairs
- Wilcoxon p-value: statistical significance of ranking differences (requires n_runs >= 5)
"""


# ---------------------------------------------------------------------------
# Queue management
# ---------------------------------------------------------------------------

def load_queue():
    with open(QUEUE_FILE) as f:
        return yaml.safe_load(f)


def save_queue(queue):
    with open(QUEUE_FILE, "w") as f:
        yaml.dump(queue, f, default_flow_style=False, sort_keys=False)


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_experiment(exp):
    """Execute a single experiment config and return results dict."""
    config = exp["config"]
    exp_type = config["type"]
    sim_kwargs = dict(config.get("simulator", {}))

    seed = sim_kwargs.pop("seed", 42)
    sim = SocialSimulator(seed=seed, **sim_kwargs)

    # Build theory configs if specified
    theory_configs = None
    if "theories" in config:
        theory_configs = {}
        for name, params in config["theories"].items():
            tc = dict(params)
            tc["class"] = THEORY_MAP[name]
            theory_configs[name] = tc

    ht_kwargs = {}
    if theory_configs:
        ht_kwargs["theory_configs"] = theory_configs
    ht = HypothesisTester(simulator=sim, **ht_kwargs)

    if exp_type == "compare_theories":
        result = ht.compare_theories(
            n_runs=config.get("n_runs", 1),
            use_temporal_split=config.get("use_temporal_split", False),
        )
        return serialize(result)

    elif exp_type == "parameter_sweep":
        results = ht.parameter_sweep(config["param"], config["values"])
        return serialize(results)

    elif exp_type == "temporal_prediction":
        result = ht.temporal_prediction(config["t_freeze"], config["t_predict"])
        return serialize(result)

    elif exp_type == "test_claim":
        claim = config["claim"]
        claim_kwargs = {
            k: v for k, v in config.items()
            if k not in ("type", "claim", "simulator", "theories")
        }
        result = ht.test_claim(claim, **claim_kwargs)
        return serialize(result)

    else:
        raise ValueError(f"Unknown experiment type: {exp_type}")


def save_result(exp, result):
    """Save experiment result to JSON file, return path."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = re.sub(r"[^a-z0-9]+", "-", exp["id"].lower()).strip("-")
    filename = f"{ts}_{slug}.json"
    path = RESULTS_DIR / filename

    output = {
        "experiment_id": exp["id"],
        "question": exp["question"],
        "config": exp["config"],
        "result": result,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    return path.relative_to(RESEARCH_DIR.parent)


def serialize(obj):
    """Convert numpy types to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, "__dict__") and not isinstance(obj, type):
        return serialize(vars(obj))
    else:
        return obj


# ---------------------------------------------------------------------------
# Claude interpretation
# ---------------------------------------------------------------------------

def interpret_results(completed_experiments):
    """Call Claude to interpret a batch of completed experiments."""
    try:
        import anthropic
    except ImportError:
        print("anthropic package not installed, skipping interpretation")
        return None

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ANTHROPIC_API_KEY not set, skipping interpretation")
        return None

    client = anthropic.Anthropic(api_key=api_key)

    # Build the prompt
    experiments_text = ""
    for exp, result in completed_experiments:
        experiments_text += f"\n### Experiment: {exp['id']}\n"
        experiments_text += f"**Question:** {exp['question']}\n"
        experiments_text += f"**Config:** {json.dumps(exp['config'], indent=2, default=str)}\n"
        experiments_text += f"**Result:**\n```json\n{json.dumps(result, indent=2, default=str)[:3000]}\n```\n"

    # Load recently completed experiments for broader context
    recent_context = load_recent_results(limit=5)
    context_text = ""
    if recent_context:
        context_text = "\n## Recent prior results (for context)\n"
        for rc in recent_context:
            context_text += f"- {rc['experiment_id']}: {rc['question']}\n"

    prompt = f"""{PROJECT_CONTEXT}

## Experiments just completed
{experiments_text}
{context_text}
## Your task

Analyze these experiment results. Structure your response as:

### Key Findings
Bullet points of the most important observations. Be specific with numbers.

### Surprises
Anything unexpected or that contradicts conventional wisdom about these theories.

### Implications
What do these results mean for the broader question of which theory best models group formation?

### Follow-up Experiments
Suggest 2-3 follow-up experiments based on what you found. Format each as a YAML block
that can be directly appended to our experiment queue. Use this exact format for each:

```yaml
- id: auto-NNN
  question: "Your research question here"
  status: pending
  priority: 3
  created_by: agent
  config:
    type: compare_theories OR parameter_sweep OR temporal_prediction OR test_claim
    simulator:
      n_agents: 40
      scenario: clustered
      k: 3
      T: 30
    n_runs: 10
```

Use IDs starting from auto-{next_auto_id():03d}. Only suggest experiments that would
meaningfully advance understanding - not just variations of what was already run.
"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text


def next_auto_id():
    """Find the next available auto-NNN ID."""
    queue = load_queue()
    max_id = 0
    for exp in queue.get("experiments", []):
        match = re.match(r"auto-(\d+)", exp.get("id", ""))
        if match:
            max_id = max(max_id, int(match.group(1)))
    return max_id + 1


def extract_followups(interpretation_text):
    """Parse follow-up experiment YAML blocks from Claude's response."""
    if not interpretation_text:
        return []

    # Find all YAML code blocks
    yaml_blocks = re.findall(r"```yaml\s*\n(.*?)```", interpretation_text, re.DOTALL)

    followups = []
    for block in yaml_blocks:
        try:
            parsed = yaml.safe_load(block)
            if isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, dict) and "id" in item and "config" in item:
                        # Ensure required fields
                        item.setdefault("status", "pending")
                        item.setdefault("priority", 3)
                        item.setdefault("created_by", "agent")
                        followups.append(item)
            elif isinstance(parsed, dict) and "id" in parsed and "config" in parsed:
                parsed.setdefault("status", "pending")
                parsed.setdefault("priority", 3)
                parsed.setdefault("created_by", "agent")
                followups.append(parsed)
        except yaml.YAMLError:
            continue

    return followups


def load_recent_results(limit=5):
    """Load the most recent result files for context."""
    if not RESULTS_DIR.exists():
        return []

    files = sorted(RESULTS_DIR.glob("*.json"), reverse=True)[:limit]
    results = []
    for f in files:
        try:
            with open(f) as fh:
                results.append(json.load(fh))
        except (json.JSONDecodeError, OSError):
            continue
    return results


# ---------------------------------------------------------------------------
# Log management
# ---------------------------------------------------------------------------

def append_to_log(completed_experiments, interpretation=None):
    """Append a research cycle entry to log.md."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    cycle_num = count_log_cycles() + 1

    entry = f"\n## Cycle {cycle_num} - {ts}\n\n"
    entry += "### Experiments\n"
    for exp, result in completed_experiments:
        # Extract headline metric
        headline = summarize_result(exp, result)
        entry += f"- **{exp['id']}**: {exp['question']}\n  - {headline}\n"

    if interpretation:
        entry += f"\n### Analysis\n\n{interpretation}\n"

    entry += "\n---\n"

    with open(LOG_FILE, "a") as f:
        f.write(entry)


def count_log_cycles():
    """Count existing cycle entries in log.md."""
    if not LOG_FILE.exists():
        return 0
    text = LOG_FILE.read_text()
    return len(re.findall(r"^## Cycle \d+", text, re.MULTILINE))


def summarize_result(exp, result):
    """One-line summary of an experiment result."""
    config = exp["config"]
    exp_type = config["type"]

    if exp_type == "compare_theories":
        if "rankings" in result:
            rankings = result["rankings"]
            if rankings:
                top = rankings[0]
                if isinstance(top, (list, tuple)):
                    return f"Winner: {top[0]} (NMI={top[1]:.3f}), CTAI={result.get('ctai', '?')}"
                elif isinstance(top, dict):
                    name = top.get("theory", top.get("name", "?"))
                    score = top.get("score", top.get("nmi", "?"))
                    return f"Winner: {name} (score={score}), CTAI={result.get('ctai', '?')}"
        if "mean_similarity" in result:
            means = result["mean_similarity"]
            if isinstance(means, dict):
                best = max(means.items(), key=lambda x: x[1])
                return f"Winner: {best[0]} (mean NMI={best[1]:.3f})"
        return "Completed"

    elif exp_type == "parameter_sweep":
        n = len(result) if isinstance(result, list) else "?"
        return f"Swept {config.get('param', '?')} across {n} values"

    elif exp_type == "test_claim":
        passed = result.get("passed", "?")
        return f"Claim {'PASSED' if passed else 'FAILED'}"

    elif exp_type == "temporal_prediction":
        return "Temporal prediction completed"

    return "Completed"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CFT AutoResearch Agent")
    parser.add_argument("--max-experiments", type=int, default=3,
                        help="Max experiments to run per cycle (default: 3)")
    parser.add_argument("--run-only", action="store_true",
                        help="Run experiments but skip Claude interpretation")
    parser.add_argument("--interpret-only", action="store_true",
                        help="Skip running, interpret most recent results")
    args = parser.parse_args()

    queue = load_queue()
    experiments = queue.get("experiments", [])

    completed = []

    if not args.interpret_only:
        # Phase 1: Run pending experiments
        pending = [e for e in experiments if e.get("status") == "pending"]
        pending.sort(key=lambda e: e.get("priority", 3), reverse=True)
        to_run = pending[:args.max_experiments]

        if not to_run:
            print("No pending experiments in queue.")
        else:
            print(f"Running {len(to_run)} experiment(s)...\n")

        for exp in to_run:
            print(f"  [{exp['id']}] {exp['question']}")
            exp["status"] = "running"
            save_queue(queue)

            try:
                result = run_experiment(exp)
                result_path = save_result(exp, result)
                exp["status"] = "completed"
                exp["result_file"] = str(result_path)
                exp["completed_at"] = datetime.now(timezone.utc).isoformat()
                completed.append((exp, result))
                headline = summarize_result(exp, result)
                print(f"    -> {headline}")
            except Exception as e:
                exp["status"] = "failed"
                exp["error"] = str(e)
                print(f"    -> FAILED: {e}")
                traceback.print_exc()

            save_queue(queue)

        print(f"\n{len(completed)}/{len(to_run)} experiments completed.")

    # Phase 2: Interpret results
    if args.run_only or not completed:
        if completed:
            append_to_log(completed)
            print("Results saved (interpretation skipped).")
        return

    print("\nInterpreting results via Claude...")
    try:
        interpretation = interpret_results(completed)
        if interpretation:
            # Phase 3: Extract follow-up suggestions
            followups = extract_followups(interpretation)
            if followups:
                print(f"  Generated {len(followups)} follow-up experiment(s):")
                for fu in followups:
                    print(f"    - [{fu['id']}] {fu.get('question', '?')}")
                    experiments.append(fu)
                save_queue(queue)

            # Phase 4: Update log
            append_to_log(completed, interpretation)
            print("\nLog updated. Done.")
        else:
            append_to_log(completed)
            print("No interpretation available. Results saved.")
    except Exception as e:
        print(f"Interpretation failed: {e}")
        traceback.print_exc()
        append_to_log(completed, f"*Interpretation failed: {e}*")


if __name__ == "__main__":
    main()
