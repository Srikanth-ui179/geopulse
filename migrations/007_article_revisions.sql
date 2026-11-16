-- Geopulse Phase 2: Article Revision History & Historical Provenance
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Draft migration for offline review.
--
-- This migration establishes an immutable, append-only history of article versions,
-- enabling precise claim-to-evidence provenance across revisions, corrections, and retractions.

-- 1. Create the immutable article_revisions table
create table if not exists public.article_revisions (
    id uuid primary key default gen_random_uuid(),
    article_id uuid not null references public.articles(id) on delete restrict,
    canonical_url text not null,
    content_hash text not null,
    title text not null,
    publisher text not null,
    published_at timestamptz,
    first_seen_at timestamptz not null default now(),
    retrieved_at timestamptz not null default now(),
    last_modified text,
    etag text,
    body_text text,
    extraction_status text not null default 'success'
        check (extraction_status in ('success', 'cached', 'updated', 'paywall_detected', 'robots_denied', 'empty', 'failed')),
    extraction_method text not null default 'trafilatura',
    revision_sequence integer not null check (revision_sequence > 0),
    previous_revision_id uuid references public.article_revisions(id) on delete set null,
    revision_classification text not null default 'initial_publication'
        check (revision_classification in ('initial_publication', 'content_update', 'metadata_update', 'explicit_correction', 'explicit_retraction', 'unavailability')),
    relevant_metadata jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    unique (article_id, revision_sequence)
);

-- 2. Foreign keys and provenance tracking on parent articles table
alter table public.articles
    add column if not exists current_revision_id uuid references public.article_revisions(id) on delete set null,
    add column if not exists revision_sequence integer not null default 1;

-- 3. Link historical claims to specific article revisions in briefing evidence
alter table public.briefing_event_evidence
    add column if not exists revision_id uuid references public.article_revisions(id) on delete set null,
    add column if not exists content_hash text,
    add column if not exists start_char integer,
    add column if not exists end_char integer,
    add column if not exists section_label text;

-- 4. Fast lookup indexes
create index if not exists article_revisions_article_seq_idx
    on public.article_revisions (article_id, revision_sequence);

create index if not exists article_revisions_content_hash_idx
    on public.article_revisions (content_hash);

create index if not exists article_revisions_retrieved_idx
    on public.article_revisions (retrieved_at desc);

create index if not exists article_revisions_classification_idx
    on public.article_revisions (revision_classification);

create index if not exists articles_current_revision_idx
    on public.articles (current_revision_id);

create index if not exists briefing_evidence_revision_idx
    on public.briefing_event_evidence (revision_id);

-- 5. Concurrency-safe atomic revision append procedure (SECURITY DEFINER with strict validation)
create or replace function public.append_article_revision(
    p_revision_id uuid,
    p_article_id uuid,
    p_canonical_url text,
    p_content_hash text,
    p_title text,
    p_publisher text,
    p_published_at timestamptz,
    p_first_seen_at timestamptz,
    p_retrieved_at timestamptz,
    p_last_modified text,
    p_etag text,
    p_body_text text,
    p_extraction_status text,
    p_extraction_method text,
    p_classification text,
    p_metadata jsonb
)
returns uuid
language plpgsql
security definer
set search_path = public, pg_temp
as $$
declare
    v_rev_id uuid := coalesce(p_revision_id, gen_random_uuid());
    v_prev_rev_id uuid;
    v_expected_url text;
    v_next_seq integer;
    v_latest_hash text;
    v_latest_title text;
    v_latest_status text;
begin
    -- 1. Validate required inputs
    if p_article_id is null then
        raise exception 'Invalid revision: p_article_id cannot be null';
    end if;

    if p_canonical_url is null or length(trim(p_canonical_url)) = 0 then
        raise exception 'Invalid revision: p_canonical_url cannot be empty';
    end if;

    if p_content_hash is null or length(trim(p_content_hash)) = 0 then
        raise exception 'Invalid revision: p_content_hash cannot be empty';
    end if;

    -- 2. Verify article exists and lock parent row to serialize revision generation
    select current_revision_id, canonical_url
    into v_prev_rev_id, v_expected_url
    from public.articles
    where id = p_article_id
    for update;

    if not found then
        raise exception 'Article % not found', p_article_id;
    end if;

    -- 3. Validate article identity (canonical_url ownership match)
    if trim(v_expected_url) <> trim(p_canonical_url) then
        raise exception 'Article identity mismatch: canonical URL % does not match registered URL % for article %',
            p_canonical_url, v_expected_url, p_article_id;
    end if;

    -- 4. Content hash consistency & idempotency guard
    if v_prev_rev_id is not null then
        select content_hash, title, extraction_status
        into v_latest_hash, v_latest_title, v_latest_status
        from public.article_revisions
        where id = v_prev_rev_id;

        if v_latest_hash = p_content_hash
           and coalesce(v_latest_title, '') = coalesce(p_title, '')
           and coalesce(v_latest_status, 'success') = coalesce(p_extraction_status, 'success')
        then
            -- Exact duplicate of current revision: return existing revision ID without creating redundant sequence
            return v_prev_rev_id;
        end if;
    end if;

    -- 5. Calculate strictly monotonic next sequence
    select coalesce(max(revision_sequence), 0) + 1 into v_next_seq
    from public.article_revisions
    where article_id = p_article_id;

    -- 6. Insert immutable revision record
    insert into public.article_revisions (
        id,
        article_id,
        canonical_url,
        content_hash,
        title,
        publisher,
        published_at,
        first_seen_at,
        retrieved_at,
        last_modified,
        etag,
        body_text,
        extraction_status,
        extraction_method,
        revision_sequence,
        previous_revision_id,
        revision_classification,
        relevant_metadata
    ) values (
        v_rev_id,
        p_article_id,
        p_canonical_url,
        p_content_hash,
        coalesce(p_title, 'Untitled Article'),
        coalesce(p_publisher, 'Unknown Publisher'),
        p_published_at,
        coalesce(p_first_seen_at, now()),
        coalesce(p_retrieved_at, now()),
        p_last_modified,
        p_etag,
        p_body_text,
        coalesce(p_extraction_status, 'success'),
        coalesce(p_extraction_method, 'trafilatura'),
        v_next_seq,
        v_prev_rev_id,
        coalesce(p_classification, 'initial_publication'),
        coalesce(p_metadata, '{}'::jsonb)
    );

    -- 7. Update pointer on parent article
    update public.articles
    set current_revision_id = v_rev_id,
        revision_sequence = v_next_seq
    where id = p_article_id;

    return v_rev_id;
end;
$$;

comment on function public.append_article_revision is
    'Appends an immutable revision record for an article with strict sequence ordering and idempotency.';

-- Explicit privilege restriction: revoke default execution from PUBLIC, anon, and authenticated
revoke execute on function public.append_article_revision(
    uuid, uuid, text, text, text, text, timestamptz, timestamptz, timestamptz,
    text, text, text, text, text, text, jsonb
) from public;

revoke execute on function public.append_article_revision(
    uuid, uuid, text, text, text, text, timestamptz, timestamptz, timestamptz,
    text, text, text, text, text, text, jsonb
) from anon, authenticated;

-- Grant execution strictly to backend service_role
grant execute on function public.append_article_revision(
    uuid, uuid, text, text, text, text, timestamptz, timestamptz, timestamptz,
    text, text, text, text, text, text, jsonb
) to service_role;

-- 6. Row Level Security (RLS) enforcement on article_revisions
alter table public.article_revisions enable row level security;

-- Revoke default table access from public and untrusted roles
revoke all on table public.article_revisions from public, anon, authenticated;

-- Read-only policy for authenticated analysts
create policy article_revisions_select_policy
    on public.article_revisions
    for select
    to authenticated
    using (true);

grant select on table public.article_revisions to authenticated;

-- Backend service role permissions (select, insert, update for retention scrubbing)
grant select, insert, update on table public.article_revisions to service_role;

-- 7. Immutability trigger: prohibits DELETE and enforces strict metadata immutability on UPDATE
create or replace function public.enforce_article_revision_immutability()
returns trigger
language plpgsql
security definer
set search_path = public, pg_temp
as $$
begin
    -- 1. DELETE and TRUNCATE operations are strictly prohibited to preserve permanent historical provenance
    if tg_op in ('DELETE', 'TRUNCATE') then
        raise exception 'article_revisions is append-only: % operations are prohibited. Historical provenance is permanent.', tg_op;
    end if;

    -- 2. UPDATE operations: enforce provenance immutability while allowing retention scrubbing
    if tg_op = 'UPDATE' then
        if new.id <> old.id
           or new.article_id <> old.article_id
           or new.canonical_url <> old.canonical_url
           or new.content_hash <> old.content_hash
           or new.revision_sequence <> old.revision_sequence
           or coalesce(new.previous_revision_id, '00000000-0000-0000-0000-000000000000'::uuid) <>
              coalesce(old.previous_revision_id, '00000000-0000-0000-0000-000000000000'::uuid)
           or new.created_at <> old.created_at
           or new.first_seen_at <> old.first_seen_at
           or new.retrieved_at <> old.retrieved_at
           or new.revision_classification <> old.revision_classification
           or coalesce(new.title, '') <> coalesce(old.title, '')
           or coalesce(new.publisher, '') <> coalesce(old.publisher, '')
           or coalesce(new.published_at, '1970-01-01'::timestamptz) <> coalesce(old.published_at, '1970-01-01'::timestamptz)
        then
            raise exception 'article_revisions provenance metadata is immutable: modification of hashes, sequences, titles, dates, or URLs is prohibited.';
        end if;

        -- Copyright retention compliance:
        -- Body text may only be scrubbed/reduced, never altered or expanded
        if new.body_text is not null and old.body_text is not null and length(new.body_text) > length(old.body_text) then
            raise exception 'article_revisions body_text cannot be expanded or altered: only retention scrubbing/redaction is permitted.';
        end if;

        return new;
    end if;

    return new;
end;
$$;

create or replace trigger trg_enforce_article_revision_immutability
before update or delete on public.article_revisions
for each row execute function public.enforce_article_revision_immutability();

create or replace trigger trg_enforce_article_revision_immutability_truncate
before truncate on public.article_revisions
for each statement execute function public.enforce_article_revision_immutability();
