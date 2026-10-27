"""Print explained matches for one synthetic campaign.  Usage: python scripts/demo.py [campaign_id] [k]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from matching.api import build_state  # noqa: E402


def main() -> None:
    cid = sys.argv[1] if len(sys.argv) > 1 else "cp_000"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    s = build_state()
    campaign, matcher = s["campaigns"][cid], s["matcher"]
    print(f"Campaign {cid}: {campaign.brief}")
    print(f"Filters: countries={campaign.target_countries} followers="
          f"{campaign.min_followers}-{campaign.max_followers} budget<=${campaign.budget_usd:,.0f}")
    print(f"Embedder: {matcher.embedder.name}\n")
    for m in matcher.match(campaign, k):
        parts = "  ".join(f"{n}={c.contribution:.2f}" for n, c in m.breakdown.components.items())
        print(f"#{m.rank} @{m.handle:<22} score={m.score:.3f}  {parts}")
        print(f"    {m.explanation}")


if __name__ == "__main__":
    main()
