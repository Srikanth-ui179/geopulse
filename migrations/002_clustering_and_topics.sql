-- Geopulse Phase 3: Classification and Clustering Schema Enhancements
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Review before applying.

-- 1. Optional topic column on articles to persist classified primary topic
alter table if exists public.articles
    add column if not exists topic text check (
        topic is null or topic in (
            'armed_conflict',
            'military_operations',
            'defence_policy',
            'weapons_procurement',
            'diplomacy',
            'sanctions',
            'terrorism_insurgency',
            'cyber_hybrid_warfare',
            'security_incident',
            'other_geopolitical',
            'uncertain'
        )
    );

-- 2. Index for filtering and ordering articles by topic
create index if not exists articles_topic_idx
    on public.articles (topic, published_at desc);

-- 3. Composite index for filtering events by category and chronology
create index if not exists events_category_first_seen_idx
    on public.events (category, first_seen_at desc);

-- 4. Index for filtering events by confidence label (corroborated, single_source, disputed)
create index if not exists events_confidence_label_idx
    on public.events (confidence_label);
