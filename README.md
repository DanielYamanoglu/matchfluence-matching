# Matchfluence Matching Engine

**Explainable brand ↔ creator matching for an influencer-marketing marketplace: embedding retrieval, then a hybrid re-rank that says *why* each creator was picked.** Retrieval runs on an in-memory NumPy index by default or on Postgres + pgvector through a `match_creators()` SQL function; an integration test checks that both return the same rankings.

[![CI](https://github.com/DanielYamanoglu/matchfluence-matching/actions/workflows/ci.yml/badge.svg)](https://github.com/DanielYamanoglu/matchfluence-matching/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-service-009688)
![pgvector](https://img.shields.io/badge/pgvector-integration%20tested-3ECF8E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-TF--IDF-F7931E)
![License](https://img.shields.io/badge/license-MIT-green)

> Matchfluence is a B2B SaaS marketplace I built the backend for (Supabase). The production code is private. This repo rebuilds the **matching approach** from scratch as a standalone demo on **synthetic data**. Every brand, creator and campaign here is generated.

<p align="center"><img src="assets/architecture.svg" width="860" alt="Brief to embedding to vector retrieval to hybrid re-rank to explained matches"></p>

## How it works

1. **Embed:** creator profiles (niche, bio, content topics) and campaign briefs become vectors. A deterministic TF-IDF embedder is used out of the box, so everything runs offline; every number in this README comes from it. `all-MiniLM-L6-v2` via sentence-transformers is an optional drop-in (`requirements-optional.txt`) that was not benchmarked here.
2. **Retrieve (stage 1):** hard filters (audience country share, follower range, budget), then cosine top-k. Two interchangeable backends:
   - in memory (default): NumPy brute force over the creator matrix;
   - Postgres + pgvector (when `DATABASE_URL` is set): the SQL function [`match_creators()`](supabase/migrations/20260101000100_match_function.sql) applies the filters to the campaign row and orders by pgvector cosine distance. It is `security invoker`, so [RLS](supabase/migrations/20260101000200_rls.sql) would apply to the calling brand on Supabase (not tested here, see [Status and limits](#status-and-limits)).
3. **Re-rank (stage 2, in the app):** [`scoring.py`](src/matching/scoring.py) adds the rule features to the semantic score:

   | Feature | Weight | Meaning |
   |---|---|---|
   | semantic | 0.50 | brief ↔ profile similarity, min-max scaled to [0, 1] within the retrieved candidate pool |
   | audience_overlap | 0.20 | share of the creator's audience in the target countries |
   | engagement | 0.10 | engagement rate vs an 8 % benchmark |
   | budget_fit | 0.10 | rate within budget, preferring creators that use it well |
   | past_performance | 0.10 | historical campaign outcome score |

4. **Explain:** every match returns its score breakdown plus a one-line reason.

Because the semantic score is scaled within the candidate pool (5·k candidates), it is a relative rank signal, not a raw cosine similarity, and final scores shift slightly with `k`.

```text
#1 @demo_creator_0700  score=0.788  semantic=0.49  audience_overlap=0.07  engagement=0.10 ...
    Strongest signals: brief/profile similarity (0.99), engagement (1.00). Weakest: audience overlap (0.35).
```

Both backends implement one `VectorIndex` interface (`InMemoryIndex`, `PgVectorIndex` in [`index.py`](src/matching/index.py)). The Postgres one writes the campaign's brief embedding to `campaigns.brief_embedding` and calls `match_creators()`, so on that path the filters exist only in SQL.

## API

```bash
uvicorn matching.api:create_app --factory --app-dir src
# or against a seeded Postgres (see "Postgres + pgvector" below):
DATABASE_URL=postgresql://postgres:postgres@localhost:5433/matching MATCHING_EMBEDDER=tfidf \
  uvicorn matching.api:create_app --factory --app-dir src
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | status, number of creators, active embedder, index backend |
| GET | `/campaigns` | list campaigns |
| POST | `/campaigns/{id}/matches` | top-k explained matches (optional `k`) |
| POST | `/creators` | add a creator; embedded and indexed on insert |

Interactive docs are at `/docs`.

## Results

Offline evaluation on 80 synthetic campaigns. Ground-truth relevance is graded 0–3 by the generator; "relevant" means grade ≥ 2. TF-IDF embedder, k = 10, from [`reports/results.json`](reports/results.json):

| Method | Precision@10 | nDCG@10 | Audience overlap@10 | Engagement@10 |
|---|---|---|---|---|
| Random (filtered pool) | 0.134 | 0.173 | 0.462 | 0.055 |
| Semantic only | **0.836** | 0.847 | 0.446 | 0.057 |
| Hybrid | 0.821 | **0.861** | **0.496** | **0.060** |

Hybrid vs semantic-only, with 95 % bootstrap CIs:
- nDCG@10 is **+0.015 [+0.005, +0.024]**.
- Precision@10 is **−0.015 [−0.026, −0.003]**.

Hybrid ranks the best creators higher and brings in audiences that better match the target countries. It trades a little raw precision for that.

Read these numbers as a sanity check of the pipeline, not as a benchmark. The relevance labels are generated by a formula that uses topical fit plus the same audience, engagement, budget and past-performance features the re-ranker scores, so the hybrid gain is partly circular; bios and briefs draw on shared per-niche vocabularies, and the TF-IDF vocabulary is fitted on creators and evaluation briefs together.

<p align="center"><img src="reports/figures/retrieval_quality.png" width="640" alt="Retrieval quality by method"></p>

## Quickstart

```bash
uv venv --python 3.12 .venv && uv pip install -p .venv -r requirements.txt
# optional: real sentence embeddings
uv pip install -p .venv -r requirements-optional.txt

.venv/bin/python data/generate.py             # synthetic brands / creators / campaigns
.venv/bin/python scripts/demo.py cp_000 3     # explained top-3 for one campaign
.venv/bin/python eval/evaluate.py             # metrics + figure
.venv/bin/python -m pytest -q                 # 21 offline tests (Postgres tests skip)
```

### Postgres + pgvector (local Docker)

```bash
docker compose up -d --wait                    # pgvector/pgvector:0.8.0-pg17 on localhost:5433
uv pip install -p .venv "psycopg[binary]>=3.1"
export DATABASE_URL=postgresql://postgres:postgres@localhost:5433/matching
.venv/bin/python scripts/seed_db.py --reset    # migrations + 40 brands, 80 campaigns, 800 creators
MATCHING_EMBEDDER=tfidf .venv/bin/python scripts/demo.py cp_000 3   # prints "Index: postgres"
TEST_DATABASE_URL=$DATABASE_URL .venv/bin/python -m pytest -q       # 187 tests incl. integration
docker compose down
```

- **Plain Postgres vs Supabase.** The RLS migration uses Supabase's `authenticated` role and `auth.uid()`. [`db/supabase_shim.sql`](db/supabase_shim.sql) creates stand-ins for those two objects (with the same `auth.uid()` contract), so the three migrations apply unchanged on plain Postgres. The shim is for local use only and is never run on Supabase.
- **Ids.** The schema stores brand, campaign and creator ids as `text` instead of `uuid`, so the generated ids (`cr_0000`, `cp_000`) and the API's ids are stored unchanged; that is a column-type change in the migrations rather than an id-mapping layer in the app.
- **Embeddings.** The seed script writes creator embeddings and every campaign's `brief_embedding` with the TF-IDF embedder (384 dimensions, zero-padded, matching the `vector(384)` columns; the script checks this before writing). The API refuses to start against a database seeded with a different embedder.
- `--reset` and the integration test drop and recreate the `public` and `auth` schemas, so point them only at a throwaway database such as the one above.

## Project structure

```
supabase/migrations/   schema (pgvector), match_creators() SQL function, RLS policies
db/supabase_shim.sql   local stand-in for Supabase's auth.uid() and authenticated role
docker-compose.yml     local Postgres + pgvector
src/matching/          embedding.py, index.py, scoring.py, matcher.py, api.py, models.py
data/                  seeded synthetic data generator + generated JSON
eval/evaluate.py       random vs semantic vs hybrid, bootstrap CIs
scripts/demo.py        CLI demo
scripts/seed_db.py     apply migrations and load the synthetic data into Postgres
tests/                 scoring, filters, matcher, API; Postgres-vs-in-memory integration test
```

## Status and limits

- **What the integration test covers** ([`test_pgvector_integration.py`](tests/test_pgvector_integration.py), run in CI against a `pgvector/pgvector:0.8.0-pg17` service container): the migrations apply on plain Postgres 17 with the shim; the seed loads the full synthetic dataset; for all 80 campaigns plus 4 variants with binding filters (narrow follower band, low budget, several countries, no filters), `match_creators()` returns the same top-50 ids in the same order as the in-memory index, with cosine similarities within 1e-6 (observed difference below 1e-7); every returned creator passes the hard filters; the final hybrid top-10 matches for all 80 campaigns; and the API selects Postgres from `DATABASE_URL`. CI also seeds the database and runs the demo against it.
- **Not verified:** RLS under real Supabase auth (the policies are created, but the tests connect as the table owner, which bypasses RLS, and no JWTs are issued); anything on a hosted Supabase project; the `all-MiniLM-L6-v2` embedder on the Postgres path; latency or behaviour beyond 800 creators.
- `match_creators()` does an exact scan: the query vector comes from a join with `campaigns` and ties are broken by id, so the HNSW index in the schema is not used by it. That is fine at 800 rows; at catalogue scale it would need a query-vector parameter so the index applies, and an approximate index would no longer match the in-memory ranking exactly.
- The evaluation (`eval/evaluate.py`) runs on the in-memory index only.
- Weights are hand-set. Learning them from campaign outcomes (learning-to-rank) is the obvious next step.

## What I'd do next

- Test the RLS policies with real Supabase auth (a local Supabase stack with signed-in brand users) and make `match_creators()` use the HNSW index with a recall check against the exact ranking.
- Learn the re-rank weights from accepted/declined offers (LambdaMART or a small pairwise model).
- Add an LLM pass that rewrites the explanation for the brand and flags brand-safety risks.
