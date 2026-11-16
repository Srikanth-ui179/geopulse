-- Geopulse Phase 2 Security Hardening Rollback: Article Revisions Column-Level Permissions
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY.
--
-- Restores the permissions state prior to migration 008 (as established in migration 007):
-- 1. Revokes column-level and table-level privileges on public.article_revisions from authenticated.
-- 2. Restores broad table-level SELECT privilege on public.article_revisions to authenticated (matching 007 state).
-- 3. Reaffirms backend service_role operational permissions (SELECT, INSERT, UPDATE).
-- 4. Preserves all tables, constraints, functions, triggers, and RLS policies from migration 007 intact.

-- 1. Revoke column-level and any residual table privileges from authenticated role
revoke all on table public.article_revisions from authenticated;

-- 2. Restore table-level SELECT privilege to authenticated role (matching migration 007 state)
grant select on table public.article_revisions to authenticated;

-- 3. Reaffirm backend service_role table permissions
grant select, insert, update on table public.article_revisions to service_role;

