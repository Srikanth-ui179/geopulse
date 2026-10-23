-- Geopulse initial schema. Review before applying to a Supabase project.
create extension if not exists pgcrypto;

create table if not exists public.sources (
    id uuid primary key default gen_random_uuid(),
    name text not null check (length(trim(name)) > 0),
    feed_url text not null unique check (length(trim(feed_url)) > 0),
    enabled boolean not null default true,
    created_at timestamptz not null default now()
);

create table if not exists public.articles (
    id uuid primary key default gen_random_uuid(),
    source_id uuid not null references public.sources(id) on delete cascade,
    url text not null,
    canonical_url text not null unique,
    title text not null,
    author text,
    published_at timestamptz,
    fetched_at timestamptz not null default now(),
    snippet text,
    content_hash text,
    language text,
    relevance_status text not null default 'pending'
        check (relevance_status in ('pending', 'relevant', 'irrelevant', 'review')),
    created_at timestamptz not null default now()
);

create index if not exists articles_published_idx
    on public.articles (published_at desc);
create index if not exists articles_source_published_idx
    on public.articles (source_id, published_at desc);
create index if not exists articles_relevance_idx
    on public.articles (relevance_status);

create table if not exists public.events (
    id uuid primary key default gen_random_uuid(),
    title text not null,
    category text not null,
    region text,
    first_seen_at timestamptz not null default now(),
    last_seen_at timestamptz not null default now(),
    summary text,
    significance text,
    confidence_label text not null default 'uncertain'
        check (confidence_label in ('corroborated', 'single_source', 'disputed', 'uncertain')),
    created_at timestamptz not null default now(),
    check (last_seen_at >= first_seen_at)
);

create index if not exists events_last_seen_idx
    on public.events (last_seen_at desc);

create table if not exists public.event_articles (
    event_id uuid not null references public.events(id) on delete cascade,
    article_id uuid not null references public.articles(id) on delete cascade,
    relationship_type text not null default 'reports_on',
    created_at timestamptz not null default now(),
    primary key (event_id, article_id)
);

create index if not exists event_articles_article_id_idx
    on public.event_articles (article_id);

create table if not exists public.briefings (
    id uuid primary key default gen_random_uuid(),
    briefing_date date not null unique,
    status text not null default 'pending'
        check (status in ('pending', 'processing', 'sent', 'partial', 'failed')),
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    telegram_message_count integer not null default 0,
    error_message text,
    created_at timestamptz not null default now(),
    check (completed_at is null or completed_at >= started_at)
);

create table if not exists public.briefing_events (
    briefing_id uuid not null references public.briefings(id) on delete cascade,
    event_id uuid not null references public.events(id) on delete cascade,
    rank integer not null check (rank > 0),
    summary text not null,
    why_it_matters text,
    created_at timestamptz not null default now(),
    primary key (briefing_id, event_id),
    unique (briefing_id, rank)
);

create index if not exists briefing_events_event_id_idx
    on public.briefing_events (event_id);

create table if not exists public.briefing_event_evidence (
    briefing_id uuid not null,
    event_id uuid not null,
    article_id uuid not null references public.articles(id) on delete cascade,
    primary key (briefing_id, event_id, article_id),
    foreign key (briefing_id, event_id)
        references public.briefing_events(briefing_id, event_id) on delete cascade
);

create index if not exists briefing_event_evidence_article_id_idx
    on public.briefing_event_evidence (article_id);

create table if not exists public.pipeline_runs (
    id uuid primary key default gen_random_uuid(),
    run_type text not null,
    status text not null default 'running'
        check (status in ('running', 'success', 'partial', 'failed')),
    articles_fetched integer not null default 0,
    articles_new integer not null default 0,
    events_created integer not null default 0,
    started_at timestamptz not null default now(),
    completed_at timestamptz,
    error_details jsonb,
    check (completed_at is null or completed_at >= started_at)
);

create index if not exists pipeline_runs_started_idx
    on public.pipeline_runs (started_at desc);
