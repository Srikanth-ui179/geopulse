-- Geopulse Phase 5: Telegram Delivery State & Auditability Enhancements
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Review and approve before applying.
--
-- This migration enhances the existing public.briefings table with Telegram-specific
-- auditability fields: target chat, message IDs returned by Telegram, content hash,
-- and multi-part chunk delivery progress.

alter table if exists public.briefings
    add column if not exists telegram_chat_id text,
    add column if not exists telegram_message_ids jsonb default '[]'::jsonb,
    add column if not exists digest_hash text,
    add column if not exists chunks_delivered integer not null default 0;

-- Index for querying delivery runs by content hash
create index if not exists briefings_digest_hash_idx
    on public.briefings (digest_hash);

-- Index for auditing deliveries by recipient chat
create index if not exists briefings_chat_id_idx
    on public.briefings (telegram_chat_id);
