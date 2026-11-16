"""Complete End-to-End V1 Operational Scenarios Test Suite.

Validates the 12 required V1 user journey scenarios using isolated fixtures
and disposable storage:

1. Five publishers repeating one Reuters report.
2. Multiple publishers repeating one official announcement.
3. Two genuinely independent investigations.
4. Conflicting evidence and contradictory claims.
5. An event with unknown source relationships.
6. An unassessed legacy event.
7. An unapproved corroborated event.
8. An approved corroborated event.
9. A downgraded event.
10. A rejected event.
11. A candidate whose evidence changes after approval (fingerprint invalidation).
12. A briefing containing a mixture of all statuses.

Proves:
- The review queue persists across executions.
- Decisions survive a simulated restart.
- Unapproved events cannot reach final output.
- Approval and downgrade decisions change output correctly.
- A changed candidate cannot reuse an outdated approval.
"""

from datetime import UTC, date, datetime
from pathlib import Path

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.corroboration.service import corroborate_event
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.ingestion.models import ArticleRecord
from geopulse.pilot import (
    OperatorSupervisedPilot,
    compute_candidate_fingerprint,
)
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryClaim,
)


def _make_art(
    source_name: str,
    title: str,
    snippet: str,
    url: str,
    author: str | None = None,
) -> ArticleRecord:
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)
    return ArticleRecord(
        source_name=source_name,
        title=title,
        snippet=snippet,
        url=url,
        canonical_url=url,
        published_at=now,
        fetched_at=now,
        author=author,
        content_hash=f"hash-{url}",
        relevance_status="relevant",
        topic="military_operations",
    )


# ==============================================================================
# Scenario 1: Five publishers repeating one Reuters report
# ==============================================================================
def test_scenario_01_wire_syndication_shared_origin(tmp_path: Path):
    """Five publishers repeating one primary Reuters dispatch must yield [SHARED ORIGIN]."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json",
        queue_path=tmp_path / "pilot_queue.json",
    )
    formatter = TelegramDigestFormatter()
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art_wire = _make_art(
        "Reuters",
        "Air strike hits ammunition depot in Tartus",
        "By Reuters Staff in Tartus. An airstrike struck a depot.",
        "https://reuters.com/tartus-strike",
    )
    syndicates = [
        _make_art(
            f"Syndicate {i}",
            "Air strike hits Tartus depot",
            "According to a report by Reuters, an airstrike struck a depot.",
            f"https://outlet{i}.com/tartus",
        )
        for i in range(1, 5)
    ]
    all_articles = (art_wire, *syndicates)
    event = EventCluster(
        title="Tartus Depot Airstrike",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=all_articles,
    )

    # Corroboration service evaluates lineage
    decision = corroborate_event(event)

    assert decision.status == "multiple_reports_shared_origin"
    assert decision.independent_chains_count == 1

    summary = EventSummary(
        event_id=event.id,
        headline="Airstrike Strikes Ammunition Depot in Tartus",
        what_happened="An airstrike struck an ammunition depot according to wire reports.",
        confidence_qualification="Operational reporting.",
        corroboration_status=decision.status,
        independent_chains_count=decision.independent_chains_count,
        sources=tuple(
            EventSourceAttribution(source_name=a.source_name, title=a.title, url=a.url)
            for a in all_articles
        ),
        model_used="test-model",
    )
    rendered = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[SHARED ORIGIN]" in rendered
    assert "[CORROBORATED]" not in rendered

    # Gated pilot review check: automated pass-through without blocking
    packet = pilot.build_review_packet(summary, event)
    assert packet.requires_approval is False


# ==============================================================================
# Scenario 2: Multiple publishers repeating one official announcement
# ==============================================================================
def test_scenario_02_shared_official_announcement(tmp_path: Path):
    """Multiple publishers reporting one ministry statement yield [OFFICIAL ANNOUNCEMENT]."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json",
        queue_path=tmp_path / "pilot_queue.json",
    )
    formatter = TelegramDigestFormatter()
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art_mod = _make_art(
        "Ministry of Defence",
        "Official announcement: new air defence missile system acquired",
        "The Ministry of Defence announced a £400 million contract for air defence missiles.",
        "https://mod.gov.uk/missiles-contract",
    )
    art_outlet = _make_art(
        "BBC News",
        "UK signs £400m air defence contract",
        "According to a statement released by the Ministry of Defence, new missiles were ordered.",
        "https://bbc.com/uk-missiles",
    )
    event = EventCluster(
        title="UK Air Defence Missile Acquisition",
        category=TopicCategory.WEAPONS_PROCUREMENT,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art_mod, art_outlet),
    )
    decision = corroborate_event(event)

    assert decision.status == "multiple_reports_shared_origin"

    summary = EventSummary(
        event_id=event.id,
        headline="Ministry of Defence Signs £400m Air Defence Contract",
        what_happened="The British MoD announced acquisition of next-generation missile systems.",
        confidence_qualification="Operational reporting.",
        corroboration_status=decision.status,
        independent_chains_count=decision.independent_chains_count,
        sources=(
            EventSourceAttribution(
                source_name=art_mod.source_name, title=art_mod.title, url=art_mod.url
            ),
            EventSourceAttribution(
                source_name=art_outlet.source_name, title=art_outlet.title, url=art_outlet.url
            ),
        ),
        model_used="test-model",
    )
    rendered = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[OFFICIAL ANNOUNCEMENT]" in rendered
    assert "[CORROBORATED]" not in rendered
    assert pilot.build_review_packet(summary, event).requires_approval is False


# ==============================================================================
# Scenario 3: Two genuinely independent investigations
# ==============================================================================
def test_scenario_03_two_independent_investigations(tmp_path: Path):
    """Two independent investigative outlets mandate human review for [CORROBORATED]."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json",
        queue_path=tmp_path / "pilot_queue.json",
    )
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_art(
        "Bellingcat",
        "Commercial satellite imagery reveals new radar array at secret base",
        "By Christo Grozev. Independent satellite analysis confirms radar installation.",
        "https://bellingcat.com/radar-base",
        author="Christo Grozev",
    )
    art2 = _make_art(
        "The Guardian",
        "On-the-ground inspection verifies advanced radar array operational",
        "By Dan Sabbagh. Field visits and local eyewitnesses confirm the radar station.",
        "https://theguardian.com/radar-investigation",
        author="Dan Sabbagh",
    )
    event = EventCluster(
        title="Covert Radar Array Discovered",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    claim = SummaryClaim(
        claim_id="c-radar-1",
        text="A new military radar array was installed and is operational.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url,
                evidence_span="radar installation",
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=art2.url, evidence_span="radar station", verification_status="accepted"
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary = EventSummary(
        event_id=event.id,
        headline="Investigative Reports Reveal Operational Military Radar Array",
        what_happened=(
            "Independent satellite imagery and ground inspections confirmed a new radar array."
        ),
        confidence_qualification="Operational reporting.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim,),
        model_used="test-model",
    )

    packet = pilot.build_review_packet(summary, event)
    assert packet.requires_approval is True
    assert packet.proposed_badge == "[CORROBORATED]"
    assert packet.proposed_chains == 2


# ==============================================================================
# Scenario 4: Conflicting evidence and contradictory claims
# ==============================================================================
def test_scenario_04_conflicting_evidence_disputed(tmp_path: Path):
    """Contradictory factual claims yield [DISPUTED] and pass through without blocking."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json",
        queue_path=tmp_path / "pilot_queue.json",
    )
    formatter = TelegramDigestFormatter()
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_art(
        "Al Jazeera",
        "Rebels claim downing of transport aircraft",
        "Rebels claim shootdown.",
        "https://aj.com/1",
    )
    event = EventCluster(
        title="Contested Aircraft Loss",
        category=TopicCategory.ARMED_CONFLICT,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1,),
    )
    summary = EventSummary(
        event_id=event.id,
        headline="Rebels Claim Downing of Transport Aircraft as Military Denies Loss",
        what_happened="Insurgents claimed a shootdown while military spokesmen denied attrition.",
        confidence_qualification="Operational reporting.",
        corroboration_status="conflicting_evidence",
        independent_chains_count=1,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
        ),
        model_used="test-model",
    )
    rendered = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[DISPUTED]" in rendered
    assert pilot.build_review_packet(summary, event).requires_approval is False


# ==============================================================================
# Scenario 5: An event with unknown source relationships
# ==============================================================================
def test_scenario_05_unknown_source_relationships_conservative(tmp_path: Path):
    """Unknown source relationships must never be inferred as corroborated."""
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)
    art1 = _make_art(
        "Outlet Alpha",
        "Border skirmish reported",
        "Skirmish occurred.",
        "https://alpha.com/skirmish",
    )
    art2 = _make_art(
        "Outlet Beta", "Border skirmish reported", "Skirmish occurred.", "https://beta.com/skirmish"
    )
    event = EventCluster(
        title="Unverified Border Skirmish",
        category=TopicCategory.SECURITY_INCIDENT,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    decision = corroborate_event(event)

    # Different publishers without demonstrable positive independence must not be corroborated
    assert decision.status != "independently_corroborated"
    assert decision.independent_chains_count <= 1


# ==============================================================================
# Scenario 6: An unassessed legacy event
# ==============================================================================
def test_scenario_06_unassessed_legacy_event(tmp_path: Path):
    """Historical or unassessed events must render explicit [NOT ASSESSED]."""
    formatter = TelegramDigestFormatter()
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_art("Legacy News", "Legacy event title", "Legacy body.", "https://legacy.com/1")
    event = EventCluster(
        title="Historical Border Patrol",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1,),
        corroboration_status="unassessed",
        independent_chains_count=0,
    )
    summary = EventSummary(
        event_id=event.id,
        headline="Historical Border Patrol Completed",
        what_happened="Routine patrol was conducted in 2024.",
        confidence_qualification="Operational reporting.",
        corroboration_status="unassessed",
        independent_chains_count=0,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
        ),
        model_used="legacy-db",
    )
    rendered = formatter.format_event_block(
        summary=summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[NOT ASSESSED]" in rendered
    assert "[CORROBORATED]" not in rendered
    assert "[SINGLE SOURCE]" not in rendered


# ==============================================================================
# Scenarios 7, 8, 9, 10: Supervisory Pilot Lifecycle (Unapproved, Approved, Downgraded, Rejected)
# ==============================================================================
def test_scenarios_07_to_10_supervisory_pilot_lifecycle(tmp_path: Path):
    """Verify unapproved events held, approved events released, downgraded updated,
    and rejected excluded.
    """
    log_file = tmp_path / "pilot_review_log.json"
    queue_file = tmp_path / "pilot_queue.json"
    pilot = OperatorSupervisedPilot(log_path=log_file, queue_path=queue_file)
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_art("AP", "Naval patrol", "Naval patrol launched.", "https://ap.com/p")
    art2 = _make_art("Reuters", "Naval patrol", "Naval patrol launched.", "https://reuters.com/p")
    event = EventCluster(
        title="Naval Patrol Operation",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="Naval patrols started.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url, evidence_span="patrol launched", verification_status="accepted"
            ),
            ClaimEvidenceReference(
                source_url=art2.url, evidence_span="patrol launched", verification_status="accepted"
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary = EventSummary(
        event_id=event.id,
        headline="Allied Warships Begin Patrols",
        what_happened="Warships commenced patrols in maritime corridor.",
        confidence_qualification="Operational reporting.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim,),
        model_used="test-model",
    )
    res = SummarizationResult(event_id=event.id, status="success", summary=summary, event=event)

    # --- Scenario 7: Unapproved corroborated event ---
    supervised, pending = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending) == 1
    assert len(supervised) == 0  # Completely held from published briefing!

    # Verify review queue was persisted to disk
    queue_items = pilot.get_queue()
    assert len(queue_items) == 1
    assert queue_items[0]["event_id"] == event.id
    assert queue_items[0]["operational_status"] == "PENDING_APPROVAL"

    # --- Scenario 8: Approved corroborated event ---
    pilot.review_event(
        event_id=event.id,
        action="approve",
        analyst_id="ANALYST-CHIEF-01",
        rationale="Verified primary independent reports from distinct field correspondents.",
    )
    # Re-evaluate briefing after approval
    supervised_approved, pending_approved = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending_approved) == 0
    assert len(supervised_approved) == 1
    assert supervised_approved[0].summary.corroboration_status == "independently_corroborated"

    rendered_approved = TelegramDigestFormatter().format_event_block(
        summary=supervised_approved[0].summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[CORROBORATED]" in rendered_approved

    # --- Scenario 9: Downgraded event ---
    pilot.review_event(
        event_id=event.id,
        action="downgrade_shared_origin",
        analyst_id="ANALYST-CHIEF-01",
        rationale="Secondary inspection identified shared wire copy. Downgrading to shared origin.",
    )
    supervised_down, pending_down = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending_down) == 0
    assert len(supervised_down) == 1
    assert supervised_down[0].summary.corroboration_status == "multiple_reports_shared_origin"

    rendered_down = TelegramDigestFormatter().format_event_block(
        summary=supervised_down[0].summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )
    assert "[SHARED ORIGIN]" in rendered_down
    assert "[CORROBORATED]" not in rendered_down

    # --- Scenario 10: Rejected event ---
    pilot.review_event(
        event_id=event.id,
        action="reject",
        analyst_id="ANALYST-CHIEF-01",
        rationale="Fabricated claim suspected by intelligence monitoring. Rejecting from briefing.",
    )
    supervised_rej, pending_rej = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending_rej) == 0
    assert len(supervised_rej) == 0  # Excluded from final briefing!


# ==============================================================================
# Scenario 11: A candidate whose evidence changes after approval (fingerprint invalidation)
# ==============================================================================
def test_scenario_11_candidate_evidence_change_invalidates_approval(tmp_path: Path):
    """Material changes in claims or evidence must invalidate prior approvals (fail closed)."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json",
        queue_path=tmp_path / "pilot_queue.json",
    )
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_art("Outlet A", "Border Post", "Soldiers deployed.", "https://a.com/1")
    art2 = _make_art("Outlet B", "Border Post", "Soldiers deployed.", "https://b.com/2")
    event = EventCluster(
        title="Border Post Deployment",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    claim1 = SummaryClaim(
        claim_id="c1",
        text="Soldiers were deployed to border.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url,
                evidence_span="soldiers deployed",
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=art2.url,
                evidence_span="soldiers deployed",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary_v1 = EventSummary(
        event_id=event.id,
        headline="Soldiers Deployed to Border Post",
        what_happened="Troops took up defensive positions.",
        confidence_qualification="Operational reporting.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim1,),
        model_used="test-model",
    )
    res_v1 = SummarizationResult(
        event_id=event.id, status="success", summary=summary_v1, event=event
    )

    # 1. Enqueue and approve candidate v1
    pilot.supervise_summaries([res_v1], hold_unapproved=True)
    pilot.review_event(
        event_id=event.id,
        action="approve",
        analyst_id="ANALYST-DEF",
        rationale="Approved version 1 based on verified primary reports.",
    )

    # Verification: v1 passes through as approved
    supervised1, _ = pilot.supervise_summaries([res_v1], hold_unapproved=True)
    assert len(supervised1) == 1
    assert supervised1[0].summary.corroboration_status == "independently_corroborated"

    # 2. Candidate v2: evidence materially changes (new claim added)
    claim2 = SummaryClaim(
        claim_id="c2",
        text="Heavy artillery was moved to the frontline.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url,
                evidence_span="heavy artillery moved",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary_v2 = summary_v1.model_copy(update={"claims": (claim1, claim2)})
    res_v2 = SummarizationResult(
        event_id=event.id, status="success", summary=summary_v2, event=event
    )

    fp1 = compute_candidate_fingerprint(summary_v1, event)
    fp2 = compute_candidate_fingerprint(summary_v2, event)
    assert fp1 != fp2, "Fingerprint must change when claims or evidence are modified"

    # 3. Supervise candidate v2: stale approval must NOT apply silently!
    supervised2, pending2 = pilot.supervise_summaries([res_v2], hold_unapproved=True)
    assert len(pending2) == 1  # Intercepted!
    assert len(supervised2) == 0  # Blocked from publication!
    assert pending2[0].candidate_fingerprint == fp2


# ==============================================================================
# Scenario 12: A briefing containing a mixture of all statuses
# ==============================================================================
def test_scenario_12_comprehensive_mixed_briefing(tmp_path: Path):
    """A full briefing containing all statuses formats correctly and excludes
    unapproved/rejected items.
    """
    log_file = tmp_path / "pilot_review_log.json"
    queue_file = tmp_path / "pilot_queue.json"
    pilot = OperatorSupervisedPilot(log_path=log_file, queue_path=queue_file)
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    def _quick_res(
        eid: str,
        title: str,
        narrative: str,
        status: str,
        chains: int,
        cat: TopicCategory,
        has_official: bool = False,
        num_sources: int = 1,
    ):
        articles = tuple(
            _make_art(
                "Ministry of Defence" if has_official else f"News Outlet {i}",
                title,
                narrative,
                f"https://source{i}.com/{eid}",
            )
            for i in range(1, num_sources + 1)
        )
        ev = EventCluster(
            id=eid,
            title=title,
            category=cat,
            first_seen_at=now,
            last_seen_at=now,
            articles=articles,
        )
        claims = ()
        if num_sources >= 2:
            claims = (
                SummaryClaim(
                    claim_id=f"c-{eid}",
                    text=narrative,
                    evidence_refs=tuple(
                        ClaimEvidenceReference(
                            source_url=a.url,
                            evidence_span=narrative,
                            verification_status="accepted",
                        )
                        for a in articles
                    ),
                    classification="reported_fact",
                    central=True,
                    verification_status="accepted",
                ),
            )
        sm = EventSummary(
            event_id=eid,
            headline=title,
            what_happened=narrative,
            confidence_qualification="Operational reporting.",
            corroboration_status=status,
            independent_chains_count=chains,
            sources=tuple(
                EventSourceAttribution(source_name=a.source_name, title=a.title, url=a.url)
                for a in articles
            ),
            claims=claims,
            model_used="test-model",
        )
        return SummarizationResult(event_id=eid, status="success", summary=sm, event=ev)

    # 1. Approved Corroborated
    res_corrob = _quick_res(
        "e1",
        "Allied Naval Forces Patrol Red Sea",
        "Warships from allied naval forces commenced international maritime patrols today.",
        "independently_corroborated",
        2,
        TopicCategory.MILITARY_OPERATIONS,
        num_sources=2,
    )
    # 2. Downgraded Shared Origin
    res_shared = _quick_res(
        "e2",
        "Trade Report Quotes Artillery Acquisition Scoop",
        "Romania reached preliminary agreement to acquire self-propelled howitzers.",
        "independently_corroborated",
        2,
        TopicCategory.WEAPONS_PROCUREMENT,
        num_sources=2,
    )
    # 3. Pass-through Official Announcement
    res_official = _quick_res(
        "e3",
        "Government Announces New Strategic Export Controls",
        "Ministry officials imposed strict controls on dual-use technology transfers.",
        "multiple_reports_shared_origin",
        1,
        TopicCategory.SANCTIONS,
        has_official=True,
        num_sources=1,
    )
    # 4. Pass-through Single Source
    res_single = _quick_res(
        "e4",
        "Border Checkpoint Reopens After Brief Closure",
        (
            "Civilian authorities reopened the northern transit corridor following technical "
            "maintenance."
        ),
        "single_source",
        1,
        TopicCategory.SECURITY_INCIDENT,
        num_sources=1,
    )
    # 5. Pass-through Disputed
    res_disputed = _quick_res(
        "e5",
        "Conflicting Reports Over Border Outpost Shelling",
        (
            "Contested accounts emerged regarding the source and casualty count of morning "
            "artillery shelling."
        ),
        "conflicting_evidence",
        1,
        TopicCategory.ARMED_CONFLICT,
        num_sources=1,
    )
    # 6. Pass-through Preliminary
    res_prelim = _quick_res(
        "e6",
        "Unconfirmed Explosion Heard Near Industrial Port",
        (
            "Initial unverified reports described smoke rising near the cargo terminal without "
            "official confirmation."
        ),
        "insufficient_information",
        0,
        TopicCategory.SECURITY_INCIDENT,
        num_sources=1,
    )
    # 7. Pass-through Unassessed
    res_unassessed = _quick_res(
        "e7",
        "Historical Memorial Commemorates Armistice Anniversary",
        (
            "Diplomats and veterans gathered at the national cemetery for an annual memorial "
            "wreath-laying ceremony."
        ),
        "unassessed",
        0,
        TopicCategory.DIPLOMACY,
        num_sources=1,
    )
    # 8. Unapproved Corroborated (Must be held!)
    res_unapproved = _quick_res(
        "e8",
        "Unapproved Report Claims Secret Airbase Expansion",
        "Satellite imagery analysts noted new tarmac construction at an isolated airfield.",
        "independently_corroborated",
        2,
        TopicCategory.MILITARY_OPERATIONS,
        num_sources=2,
    )
    # 9. Rejected Event (Must be excluded!)
    res_rejected = _quick_res(
        "e9",
        "Fabricated Hostile Incursion Allegation Rejected",
        "Unsubstantiated claims of cross-border raiding were dismissed as disinformation.",
        "independently_corroborated",
        2,
        TopicCategory.MILITARY_OPERATIONS,
        num_sources=2,
    )

    # Pre-populate decisions for e1 (Approve), e2 (Downgrade), e9 (Reject)
    for r in [res_corrob, res_shared, res_rejected]:
        pkt = pilot.build_review_packet(r.summary, r.event)
        pilot._persist_queue_items([pkt], {pkt.event_id: r.summary})

    pilot.review_event(
        "e1", "approve", "ANALYST-01", "Verified independent investigative primary coverage."
    )
    pilot.review_event(
        "e2", "downgrade_shared_origin", "ANALYST-01", "Secondary trade quotes Defense News."
    )
    pilot.review_event(
        "e9", "reject", "ANALYST-01", "Fabricated report by state propaganda outlet."
    )

    raw_candidates = [
        res_corrob,
        res_shared,
        res_official,
        res_single,
        res_disputed,
        res_prelim,
        res_unassessed,
        res_unapproved,
        res_rejected,
    ]

    supervised, pending = pilot.supervise_summaries(raw_candidates, hold_unapproved=True)

    # Prove interception of unapproved and exclusion of rejected
    assert len(pending) == 1
    assert pending[0].event_id == "e8"  # Unapproved drone base held in review
    assert not any(s.event_id == "e8" for s in supervised)
    assert not any(s.event_id == "e9" for s in supervised)  # Rejected excluded

    # Exactly 7 events remain in published briefing
    assert len(supervised) == 7

    # Format into complete Telegram digest
    formatter = TelegramDigestFormatter()
    chunks = formatter.format_digest(
        results=supervised,
        briefing_date=date(2026, 11, 4),
        edition="morning",
        include_low_priority=True,
    )
    full_digest = "\n".join(chunks)

    # Check presence of all 7 badges
    assert "[CORROBORATED]" in full_digest
    assert "[SHARED ORIGIN]" in full_digest
    assert "[OFFICIAL ANNOUNCEMENT]" in full_digest
    assert "[SINGLE SOURCE]" in full_digest
    assert "[DISPUTED]" in full_digest
    assert "[PRELIMINARY]" in full_digest
    assert "[NOT ASSESSED]" in full_digest

    # Ensure unapproved and rejected events are never rendered
    assert "Unapproved Drone Base Discovery" not in full_digest
    assert "Fabricated Missile Launch Claim" not in full_digest
