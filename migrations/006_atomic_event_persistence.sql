-- Geopulse Phase 6C: Atomic Event and Association Persistence Function
-- DO NOT APPLY TO LIVE DATABASE AUTOMATICALLY. Draft migration for offline review.
--
-- This migration creates a PostgreSQL function to transactionally upsert events
-- and their corresponding event_articles within a single atomic database transaction.
-- Depends on migrations 001 and 004.

create or replace function public.upsert_events_with_articles(
    p_events jsonb,
    p_event_articles jsonb
)
returns jsonb
language plpgsql
security invoker
set search_path = public, pg_temp
as $$
declare
    v_events_count integer := 0;
    v_articles_linked_count integer := 0;
begin
    -- 1. Guard against empty or null event payloads
    if p_events is null or jsonb_typeof(p_events) <> 'array' or jsonb_array_length(p_events) = 0 then
        return jsonb_build_object('events_persisted', 0, 'articles_linked', 0);
    end if;

    if p_event_articles is null or jsonb_typeof(p_event_articles) <> 'array' or jsonb_array_length(p_event_articles) = 0 then
        raise exception 'Cannot persist events without article associations';
    end if;

    -- 2. Validate events payload structure and canonical UUID format (8-4-4-4-12 hex)
    if exists (
        select 1
        from jsonb_array_elements(p_events) as e
        where e->>'id' is null or length(trim(e->>'id')) = 0
           or e->>'title' is null or length(trim(e->>'title')) = 0
           or e->>'category' is null or length(trim(e->>'category')) = 0
           or e->>'first_seen_at' is null or length(trim(e->>'first_seen_at')) = 0
           or e->>'last_seen_at' is null or length(trim(e->>'last_seen_at')) = 0
           or trim(e->>'id') !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    ) then
        raise exception 'Malformed event payload: missing required fields or invalid UUID format';
    end if;

    -- 3. Reject duplicate event IDs in the same batch to prevent ambiguous metadata loss
    if exists (
        select 1
        from jsonb_array_elements(p_events) as e
        group by trim(e->>'id')
        having count(*) > 1
    ) then
        raise exception 'Duplicate event ID in batch: each event in p_events must have a unique id';
    end if;

    -- 4. Validate event timestamps constraint (last_seen_at >= first_seen_at)
    if exists (
        select 1
        from jsonb_array_elements(p_events) as e
        where (e->>'last_seen_at')::timestamptz < (e->>'first_seen_at')::timestamptz
    ) then
        raise exception 'Invalid timestamps: last_seen_at must be greater than or equal to first_seen_at';
    end if;

    -- 5. Validate association payload structure and canonical UUID formats
    if exists (
        select 1
        from jsonb_array_elements(p_event_articles) as a
        where a->>'event_id' is null or length(trim(a->>'event_id')) = 0
           or a->>'article_id' is null or length(trim(a->>'article_id')) = 0
           or trim(a->>'event_id') !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
           or trim(a->>'article_id') !~* '^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$'
    ) then
        raise exception 'Malformed association payload: event_id and article_id must be valid non-empty UUIDs';
    end if;

    -- 6. Validate that every association references an event in the requested batch
    if exists (
        select 1
        from jsonb_array_elements(p_event_articles) as a
        where not exists (
            select 1
            from jsonb_array_elements(p_events) as e
            where (trim(e->>'id'))::uuid = (trim(a->>'event_id'))::uuid
        )
    ) then
        raise exception 'Unmatched association: association references event_id not present in requested events batch';
    end if;

    -- 7. Validate that every requested event has at least one article association
    if exists (
        select 1
        from jsonb_array_elements(p_events) as e
        where not exists (
            select 1
            from jsonb_array_elements(p_event_articles) as a
            where (trim(a->>'event_id'))::uuid = (trim(e->>'id'))::uuid
        )
    ) then
        raise exception 'Orphan event rejected: every event must have at least one article association';
    end if;

    -- 8. Upsert events atomically
    with upserted_events as (
        insert into public.events (
            id,
            title,
            category,
            region,
            first_seen_at,
            last_seen_at,
            published_at,
            is_analysis,
            confidence_label
        )
        select distinct on ((trim(e->>'id'))::uuid)
            (trim(e->>'id'))::uuid,
            e->>'title',
            e->>'category',
            e->>'region',
            (e->>'first_seen_at')::timestamptz,
            (e->>'last_seen_at')::timestamptz,
            case
                when e->>'published_at' is not null and length(trim(e->>'published_at')) > 0
                then (e->>'published_at')::timestamptz
                else null
            end,
            coalesce((e->>'is_analysis')::boolean, false),
            coalesce(e->>'confidence_label', 'uncertain')
        from jsonb_array_elements(p_events) as e
        order by (trim(e->>'id'))::uuid
        on conflict (id) do update set
            title = excluded.title,
            category = excluded.category,
            region = coalesce(excluded.region, public.events.region),
            first_seen_at = least(public.events.first_seen_at, excluded.first_seen_at),
            last_seen_at = greatest(public.events.last_seen_at, excluded.last_seen_at),
            published_at = coalesce(excluded.published_at, public.events.published_at),
            is_analysis = excluded.is_analysis,
            confidence_label = excluded.confidence_label
        returning id
    )
    select count(*) into v_events_count from upserted_events;

    -- 9. Upsert event_articles atomically
    with inserted_assocs as (
        insert into public.event_articles (
            event_id,
            article_id,
            relationship_type
        )
        select distinct
            (trim(a->>'event_id'))::uuid,
            (trim(a->>'article_id'))::uuid,
            coalesce(a->>'relationship_type', 'reports_on')
        from jsonb_array_elements(p_event_articles) as a
        on conflict (event_id, article_id) do nothing
        returning event_id
    )
    select count(*) into v_articles_linked_count from inserted_assocs;

    return jsonb_build_object(
        'events_persisted', v_events_count,
        'articles_linked', v_articles_linked_count
    );
end;
$$;

comment on function public.upsert_events_with_articles(jsonb, jsonb) is
    'Transactionally upserts events and event_articles associations within a single transaction.';

-- Revoke default execution privileges from PUBLIC to close anonymous access
revoke execute on function public.upsert_events_with_articles(jsonb, jsonb) from public;

-- Explicitly revoke execution from unprivileged roles (defense-in-depth)
revoke execute on function public.upsert_events_with_articles(jsonb, jsonb) from anon, authenticated;

-- Grant execution strictly to the backend service role
grant execute on function public.upsert_events_with_articles(jsonb, jsonb) to service_role;
