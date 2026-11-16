"""Unit and regression tests for Geopulse Operator-Supervised Pilot Workflow.



Tests:

1. Mandatory approval requirement on [CORROBORATED] events.

2. Direct pass-through preservation of [NOT ASSESSED], [SHARED ORIGIN], [DISPUTED], [SINGLE SOURCE].

3. Regression Case 1: EVT-15 (Romanian K9 scoop) downgrade to [SHARED ORIGIN] with rationale.

4. Regression Case 2: EVT-20 (Prosperity Guardian coalition) downgrade to [OFFICIAL ANNOUNCEMENT].

5. Audit log recording of analyst identity, rationale, claim spans, and lineage inspection.

"""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import EventCluster
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.ingestion.models import ArticleRecord
from geopulse.pilot import OperatorSupervisedPilot, ReviewAction
from geopulse.summarization.models import (
    ClaimEvidenceReference,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
    SummaryClaim,
)


@pytest.fixture(autouse=True)
def isolate_pilot_storage_for_all_tests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Safety guard: default OperatorSupervisedPilot paths to tmp_path."""
    orig_init = OperatorSupervisedPilot.__init__

    def safe_init(self, *args, **kwargs):
        if not kwargs.get("log_path"):
            kwargs["log_path"] = tmp_path / "pilot_review_log.json"
        if not kwargs.get("queue_path"):
            kwargs["queue_path"] = tmp_path / "pilot_queue.json"
        orig_init(self, *args, **kwargs)

    monkeypatch.setattr(OperatorSupervisedPilot, "__init__", safe_init)


def _make_article(
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
        topic="defence_policy",
    )


def test_pilot_requires_approval_on_corroborated_events_only(tmp_path: Path):
    """Every event claiming [CORROBORATED] must mandate human approval."""

    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json", queue_path=tmp_path / "pilot_queue.json"
    )

    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_article("Outlet A", "Naval Operation", "Patrols began.", "https://a.com/1")

    art2 = _make_article("Outlet B", "Naval Operation", "Patrols began.", "https://b.com/2")

    event = EventCluster(
        title="Naval Operation",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )

    # 1. Corroborated event

    claim_corrob = SummaryClaim(
        claim_id="c1",
        text="Naval patrol started.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url, evidence_span="patrols began", verification_status="accepted"
            ),
            ClaimEvidenceReference(
                source_url=art2.url, evidence_span="patrols began", verification_status="accepted"
            ),
        ),
        classification="reported_fact",
        central=True,
    )

    summary_corrob = EventSummary(
        event_id=event.id,
        headline="Naval Patrol Operations Began",
        what_happened="Naval forces started patrols.",
        confidence_qualification="Independently corroborated.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim_corrob,),
        model_used="test-model",
    )

    packet_corrob = pilot.build_review_packet(summary_corrob, event)

    assert packet_corrob.requires_approval is True

    assert packet_corrob.proposed_badge == "[CORROBORATED]"

    assert "Mandatory operator review" in packet_corrob.review_trigger_reason

    # 2. Unassessed event does NOT claim corroboration

    summary_unassessed = summary_corrob.model_copy(
        update={
            "corroboration_status": "unassessed",
            "independent_chains_count": 0,
        }
    )

    packet_unassessed = pilot.build_review_packet(summary_unassessed, event)

    assert packet_unassessed.requires_approval is False

    assert packet_unassessed.proposed_badge == "[NOT ASSESSED]"


def test_pilot_regression_case_1_evt_15_trade_scoop_downgrade(tmp_path: Path):
    """Regression Case 1 (EVT-15): Secondary trade quotes must be downgraded to [SHARED ORIGIN]."""

    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json", queue_path=tmp_path / "pilot_queue.json"
    )

    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art_defnews = _make_article(
        "Defense News",
        "Romania finalizing $920 million acquisition of South Korean K9 howitzers",
        (
            "By Jaroslaw Adamowski in Warsaw. Defense ministry sources confirmed "
            "negotiations concluded."
        ),
        "https://defensenews.com/k9-romania",
        author="Jaroslaw Adamowski",
    )

    art_romania = _make_article(
        "Romania Insider",
        "Romania prepares major artillery acquisition from South Korea",
        "According to a report by Defense News, Romania concluded negotiations for K9 howitzers.",
        "https://romania-insider.com/k9-howitzers",
    )

    event = EventCluster(
        title="Romanian K9 Howitzer Acquisition",
        category=TopicCategory.DEFENCE_POLICY,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art_defnews, art_romania),
    )

    claim = SummaryClaim(
        claim_id="c15-1",
        text="Romania concluded negotiations for 54 K9 howitzers.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art_defnews.url,
                evidence_span="concluded negotiations",
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=art_romania.url,
                evidence_span="concluded negotiations",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
    )

    # Erroneously proposed as corroborated by autonomous pipeline

    summary_proposed = EventSummary(
        event_id=event.id,
        headline="Romania Concludes Negotiations for 54 K9 Howitzers",
        what_happened="Romania agreed to acquire 54 K9 artillery systems.",
        confidence_qualification="Independently corroborated.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(
                source_name=art_defnews.source_name, title=art_defnews.title, url=art_defnews.url
            ),
            EventSourceAttribution(
                source_name=art_romania.source_name, title=art_romania.title, url=art_romania.url
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )

    packet = pilot.build_review_packet(summary_proposed, event)

    assert packet.requires_approval is True

    assert packet.proposed_badge == "[CORROBORATED]"

    # Analyst inspects evidence and identifies Romania Insider quoting Defense News

    analyst_rationale = (
        "Human review of claim evidence: Romania Insider explicitly cites 'According to a "
        "report by Defense News'. This is derivative reporting of a single primary trade scoop, "
        "not independent corroboration. Downgrading to shared origin."
    )

    approved_summary, log_entry = pilot.apply_analyst_decision(
        packet=packet,
        summary=summary_proposed,
        action=ReviewAction.DOWNGRADE_SHARED_ORIGIN,
        analyst_id="ANALYST-DEFENCE-04",
        analyst_rationale=analyst_rationale,
    )

    assert approved_summary is not None

    assert approved_summary.corroboration_status == "multiple_reports_shared_origin"

    assert approved_summary.independent_chains_count == 1

    # Formatted Telegram digest renders [SHARED ORIGIN]

    rendered = TelegramDigestFormatter().format_event_block(
        summary=approved_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )

    assert "[SHARED ORIGIN]" in rendered

    assert "[CORROBORATED]" not in rendered

    # Audit log entry verification

    assert log_entry.action == "DOWNGRADE_SHARED_ORIGIN"

    assert log_entry.final_badge == "[SHARED ORIGIN]"

    assert log_entry.final_chains == 1

    assert log_entry.analyst_id == "ANALYST-DEFENCE-04"


def test_pilot_regression_case_2_evt_20_coalition_statement_downgrade(tmp_path: Path):
    """Regression Case 2 (EVT-20): Multi-agency allied coalition releases downgrade to

    [OFFICIAL ANNOUNCEMENT]."""

    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json", queue_path=tmp_path / "pilot_queue.json"
    )

    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art_centcom = _make_article(
        "U.S. Central Command",
        "Operation Prosperity Guardian marks 500th successful merchant escort",
        "In an official statement released by U.S. Central Command, warships escorted 500 ships.",
        "https://centcom.mil/release-500",
    )

    art_mod = _make_article(
        "Ministry of Defence",
        "Royal Navy warship continues Red Sea escort operations",
        (
            "According to a statement released by the UK Ministry of Defence, warships "
            "escorted 500 ships."
        ),
        "https://gov.uk/uk-mod-red-sea",
    )

    event = EventCluster(
        title="Operation Prosperity Guardian Escort Milestone",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art_centcom, art_mod),
    )

    claim = SummaryClaim(
        claim_id="c20-1",
        text="Coalition warships escorted over 500 merchant vessels.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art_centcom.url,
                evidence_span="warships escorted 500 ships",
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=art_mod.url,
                evidence_span="warships escorted 500 ships",
                verification_status="accepted",
            ),
        ),
        classification="attributed_statement",
        attribution="Operation Prosperity Guardian Coalition",
        central=True,
    )

    # Erroneously proposed as corroborated by autonomous pipeline

    summary_proposed = EventSummary(
        event_id=event.id,
        headline="Coalition Escorts Over 500 Vessels in Red Sea",
        what_happened="Operation Prosperity Guardian completed 500 merchant escorts.",
        confidence_qualification="Independently corroborated.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(
                source_name=art_centcom.source_name, title=art_centcom.title, url=art_centcom.url
            ),
            EventSourceAttribution(
                source_name=art_mod.source_name, title=art_mod.title, url=art_mod.url
            ),
        ),
        claims=(claim,),
        model_used="test-model",
    )

    packet = pilot.build_review_packet(summary_proposed, event)

    assert packet.requires_approval is True

    # Analyst identifies allied partner releases of the same coalition communique

    analyst_rationale = (
        "Human review: US Central Command and UK Ministry of Defence are allied military "
        "partners announcing joint statistics for the same unified coalition operation. "
        "This is a shared official announcement, not independent journalistic verification."
    )

    approved_summary, log_entry = pilot.apply_analyst_decision(
        packet=packet,
        summary=summary_proposed,
        action=ReviewAction.DOWNGRADE_SHARED_ORIGIN,
        analyst_id="ANALYST-MARITIME-02",
        analyst_rationale=analyst_rationale,
    )

    assert approved_summary is not None

    assert approved_summary.corroboration_status == "multiple_reports_shared_origin"

    rendered = TelegramDigestFormatter().format_event_block(
        summary=approved_summary,
        category=event.category,
        confidence_label=event.confidence_label,
        rank=1,
        total=1,
    )

    assert "[OFFICIAL ANNOUNCEMENT]" in rendered

    assert "[CORROBORATED]" not in rendered

    assert log_entry.final_badge == "[OFFICIAL ANNOUNCEMENT]"


def test_pilot_log_persistence_and_retrieval(tmp_path: Path):
    """Audit log accurately persists inspection claims, lineages, and analyst rationales."""

    log_file = tmp_path / "pilot_review_log.json"

    pilot = OperatorSupervisedPilot(log_path=log_file, queue_path=tmp_path / "pilot_queue.json")

    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_article(
        "Reuters", "Reconnaissance Mission", "Mission launched.", "https://r.com/1"
    )

    art2 = _make_article("AP", "Reconnaissance Mission", "Mission launched.", "https://ap.com/2")

    event = EventCluster(
        title="Reconnaissance Mission",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )

    claim = SummaryClaim(
        claim_id="c1",
        text="Mission launched.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url,
                evidence_span="mission launched",
                verification_status="accepted",
            ),
            ClaimEvidenceReference(
                source_url=art2.url,
                evidence_span="mission launched",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
    )

    summary = EventSummary(
        event_id=event.id,
        headline="Reconnaissance Mission Launched",
        what_happened="Allied aircraft conducted reconnaissance.",
        confidence_qualification="Independently corroborated.",
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

    # Analyst approves confirmed independent reporting

    approved, log_entry = pilot.apply_analyst_decision(
        packet=packet,
        summary=summary,
        action=ReviewAction.APPROVE,
        analyst_id="ANALYST-CHIEF-01",
        analyst_rationale=(
            "Inspected verbatim quotes from independent Reuters and AP field correspondents. "
            "Primary reporting verified from distinct flight monitoring units."
        ),
    )

    assert approved is not None

    assert approved.corroboration_status == "independently_corroborated"

    history = pilot.get_review_history()

    assert len(history) == 1

    assert history[0]["review_id"] == log_entry.review_id

    assert history[0]["action"] == "APPROVE"

    assert history[0]["final_badge"] == "[CORROBORATED]"

    assert len(history[0]["inspection_claims"]) == 1

    assert len(history[0]["inspection_lineages"]) == 2


def test_supervise_summaries_intercepts_unapproved_corroborated_events(tmp_path: Path):
    """Confirm unapproved corroborated events are intercepted and held from publication."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json", queue_path=tmp_path / "pilot_queue.json"
    )
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_article("Outlet A", "Naval Operation", "Patrols began.", "https://a.com/1")
    art2 = _make_article("Outlet B", "Naval Operation", "Patrols began.", "https://b.com/2")
    event_corrob = EventCluster(
        title="Naval Operation",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    claim_corrob = SummaryClaim(
        claim_id="c1",
        text="Naval patrol started.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url, evidence_span="patrols began", verification_status="accepted"
            ),
            ClaimEvidenceReference(
                source_url=art2.url, evidence_span="patrols began", verification_status="accepted"
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary_corrob = EventSummary(
        event_id=event_corrob.id,
        headline="Naval Patrol Operations Began",
        what_happened="Naval forces started patrols.",
        confidence_qualification="Independently corroborated.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim_corrob,),
        model_used="test-model",
    )
    res_corrob = SummarizationResult(
        event_id=event_corrob.id,
        status="success",
        summary=summary_corrob,
        event=event_corrob,
    )

    art3 = _make_article("Outlet C", "Border Shelling", "Shelling occurred.", "https://c.com/1")
    event_single = EventCluster(
        title="Border Shelling",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art3,),
    )
    claim_single = SummaryClaim(
        claim_id="c2",
        text="Shelling occurred.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art3.url,
                evidence_span="shelling occurred",
                verification_status="accepted",
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary_single = EventSummary(
        event_id=event_single.id,
        headline="Border Shelling Occurred",
        what_happened="Artillery shelled border.",
        confidence_qualification="Single source.",
        corroboration_status="single_source",
        independent_chains_count=1,
        sources=(
            EventSourceAttribution(source_name=art3.source_name, title=art3.title, url=art3.url),
        ),
        claims=(claim_single,),
        model_used="test-model",
    )
    res_single = SummarizationResult(
        event_id=event_single.id,
        status="success",
        summary=summary_single,
        event=event_single,
    )

    # Supervise with hold_unapproved=True
    supervised, pending = pilot.supervise_summaries([res_corrob, res_single], hold_unapproved=True)
    assert len(pending) == 1
    assert pending[0].event_id == event_corrob.id
    assert len(supervised) == 1
    assert supervised[0].event_id == event_single.id

    # Supervise with hold_unapproved=False (safe downgrade to unassessed)
    supervised_downgrade, pending2 = pilot.supervise_summaries(
        [res_corrob, res_single], hold_unapproved=False
    )
    assert len(pending2) == 1
    assert len(supervised_downgrade) == 2
    corrob_downgraded = next(s for s in supervised_downgrade if s.event_id == event_corrob.id)
    assert corrob_downgraded.summary.corroboration_status == "unassessed"
    assert corrob_downgraded.summary.independent_chains_count == 0


def test_supervise_summaries_releases_approved_and_downgraded_events(tmp_path: Path):
    """Events with logged analyst decisions are released accordingly."""
    pilot = OperatorSupervisedPilot(
        log_path=tmp_path / "pilot_log.json", queue_path=tmp_path / "pilot_queue.json"
    )
    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)

    art1 = _make_article("Outlet A", "Naval Patrol", "Patrols began.", "https://a.com/1")
    art2 = _make_article("Outlet B", "Naval Patrol", "Patrols began.", "https://b.com/2")
    event = EventCluster(
        title="Naval Patrol",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="Naval patrol started.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url, evidence_span="patrols began", verification_status="accepted"
            ),
            ClaimEvidenceReference(
                source_url=art2.url, evidence_span="patrols began", verification_status="accepted"
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary = EventSummary(
        event_id=event.id,
        headline="Naval Patrol Began",
        what_happened="Naval forces started patrols.",
        confidence_qualification="Independently corroborated.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim,),
        model_used="test-model",
    )
    res = SummarizationResult(
        event_id=event.id,
        status="success",
        summary=summary,
        event=event,
    )
    packet = pilot.build_review_packet(summary, event)

    # 1. Analyst approves -> preserved as corroborated
    pilot.apply_analyst_decision(
        packet=packet,
        summary=summary,
        action=ReviewAction.APPROVE,
        analyst_id="ANALYST-01",
        analyst_rationale="Verified primary independent reporting from both outlets.",
    )
    supervised, pending = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending) == 0
    assert len(supervised) == 1
    assert supervised[0].summary.corroboration_status == "independently_corroborated"

    # 2. Analyst downgrades -> modified to shared origin
    pilot.apply_analyst_decision(
        packet=packet,
        summary=summary,
        action=ReviewAction.DOWNGRADE_SHARED_ORIGIN,
        analyst_id="ANALYST-01",
        analyst_rationale="Identified shared syndicate wire copy between outlets.",
    )
    supervised2, pending2 = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending2) == 0
    assert len(supervised2) == 1
    assert supervised2[0].summary.corroboration_status == "multiple_reports_shared_origin"
    assert supervised2[0].summary.independent_chains_count == 1

    # 3. Analyst rejects -> excluded
    pilot.apply_analyst_decision(
        packet=packet,
        summary=summary,
        action=ReviewAction.REJECT,
        analyst_id="ANALYST-01",
        analyst_rationale="Fabricated claims detected, rejecting from briefing.",
    )
    supervised3, pending3 = pilot.supervise_summaries([res], hold_unapproved=True)
    assert len(pending3) == 0
    assert len(supervised3) == 0


def test_cli_brief_pilot_supervision_blocks_send(capsys):
    """The --pilot-supervision flag strictly blocks --send with a descriptive error."""
    from geopulse.cli import main

    exit_code = main(["brief", "--pilot-supervision", "--send"])
    captured = capsys.readouterr().out
    assert exit_code == 1
    assert (
        "Error: Pilot supervision mode is strictly dry-run. Live Telegram delivery is prohibited."
        in captured
    )


def test_cli_brief_pilot_supervision_interception_alert(capsys, monkeypatch, tmp_path: Path):
    """CLI brief under --pilot-supervision intercepts unapproved [CORROBORATED] events."""
    from geopulse.cli import main

    now = datetime(2026, 11, 4, 10, 0, tzinfo=UTC)
    art1 = _make_article("Outlet A", "Naval Patrol", "Patrols began.", "https://a.com/1")
    art2 = _make_article("Outlet B", "Naval Patrol", "Patrols began.", "https://b.com/2")
    event = EventCluster(
        title="Intercepted Naval Operation",
        category=TopicCategory.MILITARY_OPERATIONS,
        first_seen_at=now,
        last_seen_at=now,
        articles=(art1, art2),
    )
    claim = SummaryClaim(
        claim_id="c1",
        text="Naval patrol started.",
        evidence_refs=(
            ClaimEvidenceReference(
                source_url=art1.url, evidence_span="patrols began", verification_status="accepted"
            ),
            ClaimEvidenceReference(
                source_url=art2.url, evidence_span="patrols began", verification_status="accepted"
            ),
        ),
        classification="reported_fact",
        central=True,
    )
    summary = EventSummary(
        event_id=event.id,
        headline="Intercepted Naval Operation Began",
        what_happened="Naval forces started patrols.",
        confidence_qualification="Independently corroborated.",
        corroboration_status="independently_corroborated",
        independent_chains_count=2,
        sources=(
            EventSourceAttribution(source_name=art1.source_name, title=art1.title, url=art1.url),
            EventSourceAttribution(source_name=art2.source_name, title=art2.title, url=art2.url),
        ),
        claims=(claim,),
        model_used="test-model",
    )
    mock_result = SummarizationResult(
        event_id=event.id,
        status="success",
        summary=summary,
        event=event,
    )

    from geopulse.clustering import ClusteringSummary

    monkeypatch.setattr(
        "geopulse.cli.cluster_articles",
        lambda *args, **kwargs: ClusteringSummary(
            total_articles=2,
            total_clusters=1,
            corroborated_clusters=1,
            single_source_clusters=0,
            disputed_clusters=0,
            uncertain_clusters=0,
            clusters=(event,),
        ),
    )
    monkeypatch.setattr(
        "geopulse.cli.EventSummarizer.summarize_events",
        lambda *args, **kwargs: [mock_result],
    )

    exit_code = main(["brief", "--fixture", "--mock-model", "--pilot-supervision"])
    captured = capsys.readouterr().out
    assert exit_code == 0
    assert (
        "[PILOT SUPERVISION ALERT]: Intercepted 1 unapproved [CORROBORATED] event(s)." in captured
    )
    assert "Held in review queue: 'Intercepted Naval Operation'" in captured
    assert "These events are held from the digest until approved by a human analyst." in captured
    assert "Intercepted Naval Operation Began" not in captured

    # Verify test event was safely isolated to tmp_path and never touched .geopulse/pilot_queue.json
    isolated_queue = tmp_path / "pilot_queue.json"
    assert isolated_queue.exists()
    isolated_items = json.loads(isolated_queue.read_text(encoding="utf-8"))
    assert len(isolated_items) == 1
    assert isolated_items[0]["event_title"] == "Intercepted Naval Operation"
