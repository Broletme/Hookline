-- Run this in the Supabase SQL editor to create the jobs table.
-- Also create a public Storage bucket named "clips" in the Supabase dashboard:
--   Storage → New Bucket → name: clips → toggle Public ON

create table if not exists jobs (
  id uuid primary key,
  youtube_url text not null,
  status text not null default 'queued',
  title text,
  transcript jsonb,
  clips jsonb,
  error text,
  created_at timestamptz default now()
);
