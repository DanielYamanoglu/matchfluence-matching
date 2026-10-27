-- Stage 1 (retrieval): top-k creators by cosine similarity to the campaign brief,
-- after hard filters (audience country, follower range, budget).
-- Stage 2 (hybrid re-rank + explanation) happens in the application layer.
create or replace function match_creators(
  p_campaign_id uuid,
  p_k int default 50,
  p_min_audience_share numeric default 0.20
)
returns table (
  creator_id uuid,
  similarity double precision,
  followers integer,
  rate_usd numeric
)
language sql
stable
security invoker          -- RLS of the calling user applies
as $$
  select
    cr.id,
    1 - (ce.embedding <=> c.brief_embedding) as similarity,   -- cosine similarity
    cr.followers,
    cr.rate_usd
  from campaigns c
  join creators cr on true
  join creator_embeddings ce on ce.creator_id = cr.id
  where c.id = p_campaign_id
    and c.brief_embedding is not null
    and cr.followers between c.min_followers and c.max_followers
    and cr.rate_usd <= c.budget_usd
    and (
      cardinality(c.target_countries) = 0
      or exists (
        select 1
        from unnest(c.target_countries) as tc
        where coalesce((cr.audience_countries ->> tc)::numeric, 0) >= p_min_audience_share
      )
    )
  order by ce.embedding <=> c.brief_embedding
  limit least(greatest(p_k, 1), 500);
$$;
