-- Local stand-in for the two Supabase objects the RLS migration references, so the
-- migrations apply unchanged on plain Postgres (docker compose / CI). Do NOT run this on
-- Supabase, which already provides both. It does not emulate Supabase auth: nothing here
-- issues JWTs, and the demo connects as the table owner, which bypasses RLS.

do $$
begin
  if not exists (select 1 from pg_roles where rolname = 'authenticated') then
    create role authenticated nologin;
  end if;
end
$$;

create schema if not exists auth;

-- Same contract as Supabase's auth.uid(): the `sub` claim of the request JWT, or null.
create or replace function auth.uid() returns uuid
language sql stable
as $$
  select nullif(
    coalesce(
      current_setting('request.jwt.claim.sub', true),
      current_setting('request.jwt.claims', true)::jsonb ->> 'sub'
    ), ''
  )::uuid
$$;
