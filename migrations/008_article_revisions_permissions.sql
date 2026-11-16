-- Geopulse Phase 2 Security Hardening: Article Revisions Column-Level Permissions
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Draft migration for offline review.
--
-- This additive migration hardens public.article_revisions against unauthorized data exposure:
-- 1. Revokes table-level SELECT privilege on public.article_revisions from the authenticated role.
-- 2. Grants column-level SELECT strictly on 17 non-sensitive provenance metadata columns to authenticated.
-- 3. Explicitly excludes body_text (preventing unauthorized access to full extracted article text).
-- 4. Explicitly excludes relevant_metadata (preventing exposure of internal crawler and retention state).
-- 5. Reaffirms backend service_role full table permissions (SELECT, INSERT, UPDATE) for ingestion,
--    summarization, NLI verification, and retention scrubbing.
-- 6. Retains the existing article_revisions_select_policy USING (true) for authenticated users.
--    Rationale: In Geopulse's intelligence provenance architecture, the historical sequence of
--    public news headlines, publication dates, publishers, hashes, and revision classifications across
--    all ingested articles represents non-sensitive public record data necessary for cross-feed event auditing.
--    Limiting rows (e.g. hiding pending or unclustered articles) would introduce artificial gaps in
--    provenance chains and break client-side revision-graph rendering. The sensitive assets are exclusively
--    body_text (copyright/full text) and relevant_metadata (internal crawler data), which are strictly
--    protected at the column level.

-- 1. Revoke broad table-level SELECT privilege from authenticated role
revoke select on table public.article_revisions from authenticated;

-- 2. Grant column-level SELECT strictly on non-sensitive provenance metadata columns to authenticated role
grant select (
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
    extraction_status,
    extraction_method,
    revision_sequence,
    previous_revision_id,
    revision_classification,
    created_at
) on table public.article_revisions to authenticated;

-- 3. Reaffirm backend service_role table permissions (SELECT, INSERT, UPDATE for retention scrubbing)
grant select, insert, update on table public.article_revisions to service_role;

