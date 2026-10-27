"""Offline evaluation: random vs semantic-only vs hybrid on synthetic graded relevance.

Run from the repo root:  python eval/evaluate.py
Candidates for every method are the creators passing the campaign's hard filters.
Relevant = graded label >= 2 (precision@k); nDCG uses the 0-3 gain.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from matching.dataset import load_campaigns, load_creators, load_relevance  # noqa: E402
from matching.embedding import get_embedder  # noqa: E402
from matching.index import CandidateFilter, InMemoryIndex  # noqa: E402
from matching.matcher import Matcher  # noqa: E402
from matching.scoring import SEMANTIC_ONLY, audience_overlap  # noqa: E402

K = 10
REL_THRESHOLD = 2
BOOTSTRAPS = 2000


def precision_at_k(ranked: list[str], rel: dict[str, int], k: int) -> float:
    return sum(rel.get(c, 0) >= REL_THRESHOLD for c in ranked[:k]) / k


def dcg(gains: list[float]) -> float:
    return sum(g / np.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked: list[str], rel: dict[str, int], pool: list[str], k: int) -> float:
    ideal = sorted((rel.get(c, 0) for c in pool), reverse=True)[:k]
    best = dcg(ideal)
    return dcg([rel.get(c, 0) for c in ranked[:k]]) / best if best > 0 else 0.0


def bootstrap_ci(diff: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    means = [rng.choice(diff, size=len(diff), replace=True).mean() for _ in range(BOOTSTRAPS)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> dict:
    creators, campaigns, relevance = load_creators(), load_campaigns(), load_relevance()
    corpus = [c.text for c in creators] + [f"{c.title}. {c.brief}" for c in campaigns]
    embedder = get_embedder(corpus, "tfidf")  # fixed to the offline embedder for reproducibility
    matcher = Matcher(embedder, InMemoryIndex())
    matcher.add_creators(creators)
    by_id = {c.id: c for c in creators}
    py_rng = random.Random(0)

    per_method: dict[str, dict[str, list[float]]] = {
        m: {"p": [], "ndcg": [], "aud": [], "eng": []} for m in ("random", "semantic_only", "hybrid")
    }
    pool_sizes = []
    for cp in campaigns:
        rel = relevance[cp.id]
        flt = CandidateFilter.from_campaign(cp)
        pool = [c.id for c in creators if flt.allows(c)]
        if len(pool) < K:
            continue  # too few candidates for a meaningful top-10
        pool_sizes.append(len(pool))
        rankings = {
            "random": py_rng.sample(pool, len(pool)),
            "semantic_only": [m.creator_id for m in matcher.match(cp, K, SEMANTIC_ONLY)],
            "hybrid": [m.creator_id for m in matcher.match(cp, K)],
        }
        for name, ranked in rankings.items():
            per_method[name]["p"].append(precision_at_k(ranked, rel, K))
            per_method[name]["ndcg"].append(ndcg_at_k(ranked, rel, pool, K))
            top = [by_id[c] for c in ranked[:K]]
            per_method[name]["aud"].append(float(np.mean([audience_overlap(c, cp) for c in top])))
            per_method[name]["eng"].append(float(np.mean([c.engagement_rate for c in top])))

    n = len(pool_sizes)
    rng = np.random.default_rng(0)
    results = {
        "embedder": embedder.name, "k": K, "campaigns_evaluated": n,
        "mean_pool_size": round(float(np.mean(pool_sizes)), 1),
        "relevant_threshold": f"grade >= {REL_THRESHOLD}",
        "methods": {m: {f"precision@{K}": round(float(np.mean(v["p"])), 4),
                        f"ndcg@{K}": round(float(np.mean(v["ndcg"])), 4),
                        f"audience_overlap@{K}": round(float(np.mean(v["aud"])), 4),
                        f"engagement_rate@{K}": round(float(np.mean(v["eng"])), 4)}
                    for m, v in per_method.items()},
    }
    for metric, key in ((f"precision@{K}", "p"), (f"ndcg@{K}", "ndcg")):
        d = np.array(per_method["hybrid"][key]) - np.array(per_method["semantic_only"][key])
        lo, hi = bootstrap_ci(d, rng)
        results[f"hybrid_minus_semantic_{metric}"] = {
            "mean_diff": round(float(d.mean()), 4), "ci95": [round(lo, 4), round(hi, 4)]}

    out = ROOT / "reports"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    plot(results, out / "figures" / "retrieval_quality.png")
    print(json.dumps(results, indent=2))
    return results


def plot(results: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = {"random": "Random", "semantic_only": "Semantic only", "hybrid": "Hybrid"}
    colors = {"random": "#9aa5b1", "semantic_only": "#4f7cac", "hybrid": "#e07a1f"}
    metrics = [f"precision@{K}", f"ndcg@{K}"]
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    width = 0.26
    for i, m in enumerate(labels):
        vals = [results["methods"][m][x] for x in metrics]
        xs = [j + (i - 1) * width for j in range(len(metrics))]
        bars = ax.bar(xs, vals, width, label=labels[m], color=colors[m])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(metrics)), [f"Precision@{K}", f"nDCG@{K}"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean over campaigns")
    ax.set_title(f"Synthetic benchmark ({results['campaigns_evaluated']} campaigns, "
                 f"{results['embedder']} embeddings)", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, facecolor="white")


if __name__ == "__main__":
    main()
