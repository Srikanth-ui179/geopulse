-- Geopulse Phase 2 Rollback: Article Revision History & Historical Provenance
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY.

-- 1. Drop immutability triggers and function
drop trigger if exists trg_enforce_article_revision_immutability_truncate on public.article_revisions;
drop trigger if exists trg_enforce_article_revision_immutability on public.article_revisions;
drop function if exists public.enforce_article_revision_immutability();

-- 2. Drop RLS policy
drop policy if exists article_revisions_select_policy on public.article_revisions;

-- 3. Drop stored procedure
drop function if exists public.append_article_revision(
    uuid, uuid, text, text, text, text, timestamptz, timestamptz, timestamptz,
    text, text, text, text, text, text, jsonb
);

-- 4. Drop lookup indexes on altered core tables
drop index if exists public.briefing_evidence_revision_idx;
drop index if exists public.articles_current_revision_idx;

-- 5. Drop columns from briefing_event_evidence
alter table public.briefing_event_evidence
    drop column if exists revision_id,
    drop column if exists content_hash,
    drop column if exists start_char,
    drop column if exists end_char,
    drop column if exists section_label;

-- 6. Drop columns from articles
alter table public.articles
    drop column if exists current_revision_id,
    drop column if exists revision_sequence;

-- 7. Drop article_revisions table and its indexes
drop table if exists public.article_revisions cascade;
