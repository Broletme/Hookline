-- ============================================================
-- Hookline — Supabase schema migration
-- Run this once in: Supabase Dashboard → SQL Editor → New query
-- ============================================================

-- ── jobs table ────────────────────────────────────────────────────────────────
create table if not exists public.jobs (
    id          uuid primary key default gen_random_uuid(),
    youtube_url text        not null,
    status      text        not null default 'queued',
    -- Possible statuses: queued | downloading | transcribing | scoring | clipping | done | error
    error       text,
    transcript  jsonb,
    clips       jsonb,
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

-- Auto-update updated_at on every row change
create or replace function public.set_updated_at()
returns trigger language plpgsql as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists jobs_set_updated_at on public.jobs;
create trigger jobs_set_updated_at
    before update on public.jobs
    for each row execute function public.set_updated_at();

-- ── Row-Level Security ────────────────────────────────────────────────────────
-- The backend uses the service-role key so it bypasses RLS.
-- Enable RLS anyway so the anon key cannot read/write directly.
alter table public.jobs enable row level security;

-- ── Storage bucket ────────────────────────────────────────────────────────────
-- Create the bucket via: Storage → New bucket → name: clips → Public: ON
-- Then add this policy so the service-role key can upload freely:

-- Allow public read of clips (so the frontend <video> src works)
insert into storage.buckets (id, name, public)
values ('clips', 'clips', true)
on conflict (id) do nothing;

drop policy if exists "clips_public_read" on storage.objects;
create policy "clips_public_read" on storage.objects
    for select
    using (bucket_id = 'clips');

drop policy if exists "clips_service_insert" on storage.objects;
create policy "clips_service_insert" on storage.objects
    for insert
    with check (bucket_id = 'clips');
