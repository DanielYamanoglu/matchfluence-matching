-- Row Level Security examples (Supabase: auth.uid() is the signed-in user).
alter table brands             enable row level security;
alter table brand_members      enable row level security;
alter table creators           enable row level security;
alter table creator_embeddings enable row level security;
alter table campaigns          enable row level security;
alter table match_results      enable row level security;

-- A user sees only the brands they belong to.
create policy brands_member_read on brands
  for select to authenticated
  using (exists (
    select 1 from brand_members bm
    where bm.brand_id = brands.id and bm.user_id = auth.uid()
  ));

create policy brand_members_self_read on brand_members
  for select to authenticated
  using (user_id = auth.uid());

-- The creator catalogue is readable by any signed-in brand user; writes are
-- reserved for the service role (which bypasses RLS), so there is no write policy.
create policy creators_read on creators
  for select to authenticated using (true);

create policy creator_embeddings_read on creator_embeddings
  for select to authenticated using (true);

-- Campaigns and match results are private to the owning brand's members.
create policy campaigns_member_all on campaigns
  for all to authenticated
  using (exists (
    select 1 from brand_members bm
    where bm.brand_id = campaigns.brand_id and bm.user_id = auth.uid()
  ))
  with check (exists (
    select 1 from brand_members bm
    where bm.brand_id = campaigns.brand_id and bm.user_id = auth.uid()
  ));

create policy match_results_member_read on match_results
  for select to authenticated
  using (exists (
    select 1
    from campaigns c
    join brand_members bm on bm.brand_id = c.brand_id
    where c.id = match_results.campaign_id and bm.user_id = auth.uid()
  ));
