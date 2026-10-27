-- Schema for the matching demo: Postgres + pgvector (Supabase-compatible).
create extension if not exists vector;

create table brands (
  id          uuid primary key default gen_random_uuid(),
  name        text not null,
  industry    text,
  created_at  timestamptz not null default now()
);

-- Which authenticated users belong to which brand (used by RLS policies).
create table brand_members (
  brand_id uuid not null references brands(id) on delete cascade,
  user_id  uuid not null,
  primary key (brand_id, user_id)
);

create table creators (
  id                 uuid primary key default gen_random_uuid(),
  handle             text not null unique,
  platform           text not null,
  bio                text not null,
  tags               text[] not null default '{}',
  -- share of the audience per ISO country code, e.g. {"US": 0.6, "GB": 0.2}
  audience_countries jsonb not null default '{}',
  followers          integer not null check (followers >= 0),
  engagement_rate    numeric(5,4) not null check (engagement_rate between 0 and 1),
  rate_usd           numeric(10,2) not null check (rate_usd >= 0),
  past_performance   numeric(4,3) not null default 0 check (past_performance between 0 and 1),
  created_at         timestamptz not null default now()
);

create table campaigns (
  id               uuid primary key default gen_random_uuid(),
  brand_id         uuid not null references brands(id) on delete cascade,
  title            text not null,
  brief            text not null,
  target_countries text[] not null default '{}',
  min_followers    integer not null default 0,
  max_followers    integer not null default 2147483647,
  budget_usd       numeric(10,2) not null check (budget_usd > 0),  -- per creator
  brief_embedding  vector(384),
  created_at       timestamptz not null default now()
);

create table creator_embeddings (
  creator_id uuid primary key references creators(id) on delete cascade,
  model      text not null,           -- e.g. all-MiniLM-L6-v2 or tfidf
  embedding  vector(384) not null,
  updated_at timestamptz not null default now()
);

create table match_results (
  id          uuid primary key default gen_random_uuid(),
  campaign_id uuid not null references campaigns(id) on delete cascade,
  creator_id  uuid not null references creators(id) on delete cascade,
  score       numeric(6,5) not null,
  breakdown   jsonb not null,         -- per-feature value / weight / contribution
  created_at  timestamptz not null default now(),
  unique (campaign_id, creator_id)
);

-- Approximate nearest-neighbour index for cosine distance.
create index creator_embeddings_hnsw
  on creator_embeddings using hnsw (embedding vector_cosine_ops);
create index creators_followers_idx on creators (followers);
create index creators_audience_gin on creators using gin (audience_countries);
create index campaigns_brand_idx on campaigns (brand_id);
create index match_results_campaign_idx on match_results (campaign_id, score desc);
