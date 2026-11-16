"""Operator-Supervised Pilot Workflow for Geopulse.

Enforces human-in-the-loop review for every event claiming independent corroboration
before digest publication. Analysts inspect structured claim-level evidence and
reporting lineages rather than raw publisher counts.

Maintains strict dry-run mode with zero autonomous publishing or Telegram transmission.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from geopulse.clustering.models import EventCluster
from geopulse.corroboration.lineage import build_cluster_lineages
from geopulse.delivery.formatter import TelegramDigestFormatter
from geopulse.summarization.models import (
    EventSummary,
)


class ReviewAction(StrEnum):
    """Analyst supervisory actions for candidate briefing events."""

    APPROVE = "APPROVE"
    DOWNGRADE_SHARED_ORIGIN = "DOWNGRADE_SHARED_ORIGIN"
    DOWNGRADE_OFFICIAL_ANNOUNCEMENT = "DOWNGRADE_OFFICIAL_ANNOUNCEMENT"
    DOWNGRADE_SINGLE_SOURCE = "DOWNGRADE_SINGLE_SOURCE"
    DOWNGRADE_DISPUTED = "DOWNGRADE_DISPUTED"
    DOWNGRADE_PRELIMINARY = "DOWNGRADE_PRELIMINARY"
    REJECT = "REJECT"

    @classmethod
    def from_str(cls, val: str) -> ReviewAction:
        """Parse action string case-insensitively, supporting hyphens and underscores."""
        norm = val.strip().upper().replace("-", "_")
        if norm in ("APPROVE", "APPROVED"):
            return cls.APPROVE
        if norm in ("DOWNGRADE_SHARED_ORIGIN", "SHARED_ORIGIN", "SHARED"):
            return cls.DOWNGRADE_SHARED_ORIGIN
        if norm in (
            "DOWNGRADE_OFFICIAL_ANNOUNCEMENT",
            "DOWNGRADE_OFFICIAL",
            "OFFICIAL_ANNOUNCEMENT",
            "OFFICIAL",
        ):
            return cls.DOWNGRADE_OFFICIAL_ANNOUNCEMENT
        if norm in ("DOWNGRADE_SINGLE_SOURCE", "SINGLE_SOURCE", "SINGLE"):
            return cls.DOWNGRADE_SINGLE_SOURCE
        if norm in ("DOWNGRADE_DISPUTED", "DISPUTED", "CONFLICT"):
            return cls.DOWNGRADE_DISPUTED
        if norm in ("DOWNGRADE_PRELIMINARY", "PRELIMINARY", "UNCONFIRMED"):
            return cls.DOWNGRADE_PRELIMINARY
        if norm in ("REJECT", "REJECTED"):
            return cls.REJECT
        raise ValueError(
            f"Unknown review action '{val}'. Valid actions: approve, downgrade_shared_origin, "
            "downgrade_official, downgrade_single_source, downgrade_disputed, "
            "downgrade_preliminary, reject."
        )


def compute_candidate_fingerprint(summary: EventSummary, event: EventCluster) -> str:
    """Compute deterministic cryptographic fingerprint of a candidate event's claims and evidence.

    Binds analyst approvals to exact factual claims, evidence spans, and source lineages.
    If claims, evidence spans, or source URLs change, the fingerprint changes, invalidating
    stale operator approvals.
    """
    h = hashlib.sha256()
    h.update(str(event.id).encode("utf-8"))
    h.update((summary.headline or "").strip().encode("utf-8"))
    h.update((summary.what_happened or "").strip().encode("utf-8"))

    # Claims & Evidence Spans
    for c in sorted(summary.claims or (), key=lambda x: str(x.claim_id)):
        h.update(str(c.claim_id).encode("utf-8"))
        h.update(str(c.text).strip().encode("utf-8"))
        h.update(str(c.classification).encode("utf-8"))
        h.update(b"1" if c.central else b"0")
        for ref in sorted(
            c.evidence_refs or (),
            key=lambda r: (str(r.source_url), str(r.evidence_span)),
        ):
            h.update(str(ref.source_url).encode("utf-8"))
            h.update((ref.evidence_span or "").strip().encode("utf-8"))
            h.update(str(ref.verification_status).encode("utf-8"))

    # Article URLs & snippets
    for art in sorted(event.articles or (), key=lambda a: str(a.url)):
        h.update(str(art.url).encode("utf-8"))
        h.update((art.snippet or "").strip().encode("utf-8"))
        if getattr(art, "author", None):
            h.update(str(art.author).strip().encode("utf-8"))

    return h.hexdigest()[:16]


@dataclass(frozen=True)
class ReviewEvidenceInspection:
    """Inspection packet of claim-level evidence presented to the human analyst."""

    claim_id: str
    claim_text: str
    classification: str
    central: bool
    evidence_spans: tuple[str, ...]
    supporting_urls: tuple[str, ...]


@dataclass(frozen=True)
class ReviewLineageInspection:
    """Inspection packet of article reporting origins presented to the human analyst."""

    publisher: str
    url: str
    lineage_classification: str
    inferred_original_source: str
    author: str | None = None


@dataclass(frozen=True)
class CandidateReviewPacket:
    """Complete supervisory review packet for a candidate event."""

    event_id: str
    event_title: str
    category: str
    headline: str
    proposed_status: str
    proposed_badge: str
    proposed_chains: int
    requires_approval: bool
    review_trigger_reason: str
    claims_inspection: tuple[ReviewEvidenceInspection, ...]
    lineages_inspection: tuple[ReviewLineageInspection, ...]
    source_count: int
    article_count: int
    candidate_fingerprint: str = ""


@dataclass
class PilotReviewLogEntry:
    """Audit log entry recording an analyst's decision, rationale, and modifications."""

    review_id: str
    timestamp: str
    event_id: str
    event_title: str
    proposed_status: str
    proposed_badge: str
    proposed_chains: int
    action: str
    final_status: str
    final_badge: str
    final_chains: int
    analyst_id: str
    analyst_rationale: str
    source_count: int
    article_count: int
    inspection_claims: list[dict[str, Any]]
    inspection_lineages: list[dict[str, Any]]
    candidate_fingerprint: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class OperatorSupervisedPilot:
    """Supervisory workflow engine managing candidate event review and decision logging."""

    def __init__(
        self,
        log_path: Path | str | None = None,
        queue_path: Path | str | None = None,
        formatter: TelegramDigestFormatter | None = None,
    ) -> None:
        self.log_path = Path(log_path or ".geopulse/pilot_review_log.json")
        default_queue = (
            self.log_path.parent / "pilot_queue.json" if log_path else ".geopulse/pilot_queue.json"
        )
        self.queue_path = Path(queue_path or default_queue)
        self.formatter = formatter or TelegramDigestFormatter()
        self._ensure_storage_files()

    def _ensure_storage_files(self) -> None:
        """Ensure review audit log and queue files exist."""
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            self.log_path.write_text("[]", encoding="utf-8")
        self.queue_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.queue_path.exists():
            self.queue_path.write_text("[]", encoding="utf-8")

    def build_review_packet(
        self,
        summary: EventSummary,
        event: EventCluster,
    ) -> CandidateReviewPacket:
        """Construct a structured supervisory review packet for an analyst."""
        lineages_map = build_cluster_lineages(event.articles)
        lineage_inspections = tuple(
            ReviewLineageInspection(
                publisher=art.source_name,
                url=art.url,
                lineage_classification=(
                    lineages_map[art.url].lineage_classification
                    if art.url in lineages_map
                    else "unknown"
                ),
                inferred_original_source=(
                    lineages_map[art.url].original_source
                    if art.url in lineages_map
                    else art.source_name
                ),
                author=art.author,
            )
            for art in event.articles
        )

        claims_inspections = tuple(
            ReviewEvidenceInspection(
                claim_id=c.claim_id,
                claim_text=c.text,
                classification=c.classification,
                central=c.central,
                evidence_spans=tuple(r.evidence_span for r in c.evidence_refs if r.evidence_span),
                supporting_urls=tuple(r.source_url for r in c.evidence_refs),
            )
            for c in summary.claims
        )

        proposed_status = summary.corroboration_status or "unassessed"
        proposed_chains = summary.independent_chains_count

        rendered = self.formatter.format_event_block(
            summary=summary,
            category=event.category,
            confidence_label=event.confidence_label,
            rank=1,
            total=1,
        )

        badge_m = re.search(
            r"<b>\[#\d+/\d+\]\s+[^·]+·\s+(?:\[ANALYSIS\]\s+·\s+)?(\[[A-Z\s]+\])</b>",
            rendered,
        )
        proposed_badge = badge_m.group(1) if badge_m else "[REPORT]"

        # Human approval is strictly mandatory for every [CORROBORATED] event
        requires_approval = proposed_badge == "[CORROBORATED]" or proposed_status == (
            "independently_corroborated"
        )
        trigger_reason = (
            "Mandatory operator review: event claims independent corroboration across "
            f"{proposed_chains} reporting chains."
            if requires_approval
            else f"Automated pass-through with explicit badge {proposed_badge}."
        )

        fingerprint = compute_candidate_fingerprint(summary, event)

        return CandidateReviewPacket(
            event_id=event.id,
            event_title=event.title,
            category=getattr(event.category, "value", str(event.category)),
            headline=summary.headline,
            proposed_status=proposed_status,
            proposed_badge=proposed_badge,
            proposed_chains=proposed_chains,
            requires_approval=requires_approval,
            review_trigger_reason=trigger_reason,
            claims_inspection=claims_inspections,
            lineages_inspection=lineage_inspections,
            source_count=event.source_count,
            article_count=len(event.articles),
            candidate_fingerprint=fingerprint,
        )

    def apply_analyst_decision(
        self,
        packet: CandidateReviewPacket,
        summary: EventSummary,
        action: ReviewAction | str,
        analyst_id: str,
        analyst_rationale: str,
    ) -> tuple[EventSummary | None, PilotReviewLogEntry]:
        """Apply the analyst's supervisory decision and record it in the review log."""
        if isinstance(action, str):
            action = ReviewAction.from_str(action)

        if not analyst_id or not analyst_id.strip():
            raise ValueError("Analyst ID must be provided for auditable review decisions.")

        if not analyst_rationale or len(analyst_rationale.strip()) < 10:
            raise ValueError("Analyst rationale must be a meaningful explanation (>= 10 chars).")

        now_str = datetime.now(UTC).isoformat()
        review_id = f"REV-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{packet.event_id[:8]}"

        final_status = packet.proposed_status
        final_chains = packet.proposed_chains
        final_badge = packet.proposed_badge
        approved_summary: EventSummary | None = summary

        if action == ReviewAction.APPROVE:
            final_status = "independently_corroborated"
            final_chains = max(packet.proposed_chains, 2)
            final_badge = "[CORROBORATED]"
            approved_summary = summary.model_copy(
                update={
                    "corroboration_status": final_status,
                    "independent_chains_count": final_chains,
                }
            )

        elif action in (
            ReviewAction.DOWNGRADE_SHARED_ORIGIN,
            ReviewAction.DOWNGRADE_OFFICIAL_ANNOUNCEMENT,
        ):
            final_status = "multiple_reports_shared_origin"
            final_chains = 1
            if (
                action == ReviewAction.DOWNGRADE_OFFICIAL_ANNOUNCEMENT
                or "official" in packet.proposed_badge.lower()
            ):
                final_badge = "[OFFICIAL ANNOUNCEMENT]"
            else:
                final_badge = "[SHARED ORIGIN]"
            approved_summary = summary.model_copy(
                update={
                    "corroboration_status": final_status,
                    "independent_chains_count": final_chains,
                }
            )

        elif action == ReviewAction.DOWNGRADE_SINGLE_SOURCE:
            final_status = "single_source"
            final_chains = 1
            final_badge = "[SINGLE SOURCE]"
            approved_summary = summary.model_copy(
                update={
                    "corroboration_status": final_status,
                    "independent_chains_count": final_chains,
                }
            )

        elif action == ReviewAction.DOWNGRADE_DISPUTED:
            final_status = "conflicting_evidence"
            final_chains = max(packet.proposed_chains, 1)
            final_badge = "[DISPUTED]"
            approved_summary = summary.model_copy(
                update={
                    "corroboration_status": final_status,
                    "independent_chains_count": final_chains,
                }
            )

        elif action == ReviewAction.DOWNGRADE_PRELIMINARY:
            final_status = "insufficient_information"
            final_chains = 0
            final_badge = "[PRELIMINARY]"
            approved_summary = summary.model_copy(
                update={
                    "corroboration_status": final_status,
                    "independent_chains_count": final_chains,
                }
            )

        elif action == ReviewAction.REJECT:
            final_status = "rejected_by_operator"
            final_chains = 0
            final_badge = "[REJECTED]"
            approved_summary = None

        log_entry = PilotReviewLogEntry(
            review_id=review_id,
            timestamp=now_str,
            event_id=packet.event_id,
            event_title=packet.event_title,
            proposed_status=packet.proposed_status,
            proposed_badge=packet.proposed_badge,
            proposed_chains=packet.proposed_chains,
            action=action.value,
            final_status=final_status,
            final_badge=final_badge,
            final_chains=final_chains,
            analyst_id=analyst_id.strip(),
            analyst_rationale=analyst_rationale.strip(),
            source_count=packet.source_count,
            article_count=packet.article_count,
            inspection_claims=[asdict(c) for c in packet.claims_inspection],
            inspection_lineages=[asdict(lin) for lin in packet.lineages_inspection],
            candidate_fingerprint=packet.candidate_fingerprint,
        )

        self._append_log(log_entry)
        return approved_summary, log_entry

    def _append_log(self, entry: PilotReviewLogEntry) -> None:
        """Append review log entry to disk."""
        try:
            records = json.loads(self.log_path.read_text(encoding="utf-8"))
        except Exception:
            records = []
        records.append(entry.to_dict())
        self.log_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    def get_latest_decision_for_event(self, event_id: str) -> dict[str, Any] | None:
        """Find the latest logged analyst decision for an event, if any."""
        for entry in reversed(self.get_review_history()):
            if entry.get("event_id") == event_id:
                return entry
        return None

    def _persist_queue_items(
        self,
        packets: Sequence[CandidateReviewPacket],
        summaries: dict[str, EventSummary],
    ) -> None:
        """Persist candidate items to the review queue file."""
        try:
            queue_list = json.loads(self.queue_path.read_text(encoding="utf-8"))
            queue_map = {item["event_id"]: item for item in queue_list if "event_id" in item}
        except Exception:
            queue_map = {}

        now_str = datetime.now(UTC).isoformat()
        for pkt in packets:
            sm = summaries.get(pkt.event_id)
            queue_map[pkt.event_id] = {
                "event_id": pkt.event_id,
                "event_title": pkt.event_title,
                "category": pkt.category,
                "candidate_fingerprint": pkt.candidate_fingerprint,
                "proposed_badge": pkt.proposed_badge,
                "proposed_status": pkt.proposed_status,
                "proposed_chains": pkt.proposed_chains,
                "requires_approval": pkt.requires_approval,
                "review_trigger_reason": pkt.review_trigger_reason,
                "source_count": pkt.source_count,
                "article_count": pkt.article_count,
                "enqueued_at": now_str,
                "packet": asdict(pkt),
                "summary": sm.model_dump(mode="json") if sm else None,
            }
        self.queue_path.write_text(
            json.dumps(list(queue_map.values()), indent=2, default=str), encoding="utf-8"
        )

    def supervise_summaries(
        self,
        results: Sequence[Any],
        hold_unapproved: bool = True,
    ) -> tuple[tuple[Any, ...], list[CandidateReviewPacket]]:
        """Supervise candidate briefing results before digest formatting.

        Guarantees that:
        1. Every event claiming [CORROBORATED] is intercepted before publication.
        2. Events with logged approval pass through as [CORROBORATED] ONLY IF candidate
           fingerprint matches (detecting material modifications to claims/evidence).
        3. Events with logged downgrades pass through with their downgraded badge.
        4. Events with logged rejections are excluded.
        5. Unapproved corroborated events CANNOT bypass approval: they are
           held in the review queue (or downgraded to unassessed if hold_unapproved=False).
        """
        supervised: list[Any] = []
        pending_review_packets: list[CandidateReviewPacket] = []
        all_candidate_packets: list[CandidateReviewPacket] = []
        summaries_by_id: dict[str, EventSummary] = {}

        for res in results:
            if getattr(res, "status", None) != "success" or getattr(res, "summary", None) is None:
                supervised.append(res)
                continue

            event = getattr(res, "event", None)
            if event is None:
                supervised.append(res)
                continue

            packet = self.build_review_packet(res.summary, event)
            all_candidate_packets.append(packet)
            summaries_by_id[packet.event_id] = res.summary

            if not packet.requires_approval:
                # Direct pass-through for unassessed, shared origin, disputed, single source
                supervised.append(res)
                continue

            # Check if this event has an analyst decision in the audit log
            decision_entry = self.get_latest_decision_for_event(event.id)

            # Fingerprint validation: check if evidence or claims have changed since review
            if decision_entry is not None:
                logged_fp = decision_entry.get("candidate_fingerprint")
                if logged_fp and logged_fp != packet.candidate_fingerprint:
                    # Material change detected: stale approval is invalidated (fail-closed)
                    decision_entry = None

            if decision_entry is not None:
                action = decision_entry.get("action")
                if action == ReviewAction.APPROVE.value:
                    # Approved by analyst: preserve [CORROBORATED]
                    supervised.append(res)
                elif action == ReviewAction.REJECT.value:
                    # Rejected by analyst: omit from briefing
                    continue
                else:
                    # Downgraded by analyst: apply modified status and chains
                    final_status = decision_entry.get(
                        "final_status", "multiple_reports_shared_origin"
                    )
                    final_chains = decision_entry.get("final_chains", 1)
                    updated_summary = res.summary.model_copy(
                        update={
                            "corroboration_status": final_status,
                            "independent_chains_count": final_chains,
                        }
                    )
                    supervised.append(res.model_copy(update={"summary": updated_summary}))
            else:
                # Unapproved or stale corroborated event: mandatory interception
                pending_review_packets.append(packet)
                if hold_unapproved:
                    # Held in queue: omitted from published briefing until analyst approves
                    continue
                else:
                    # Safely downgraded to unassessed so it cannot claim corroboration
                    downgraded_summary = res.summary.model_copy(
                        update={
                            "corroboration_status": "unassessed",
                            "independent_chains_count": 0,
                            "confidence_qualification": "Pending operator corroboration review.",
                        }
                    )
                    supervised.append(res.model_copy(update={"summary": downgraded_summary}))

        # Persist candidate packets to review queue
        if all_candidate_packets:
            self._persist_queue_items(all_candidate_packets, summaries_by_id)

        return tuple(supervised), pending_review_packets

    def get_review_history(self) -> list[dict[str, Any]]:
        """Retrieve complete review audit history."""
        try:
            return json.loads(self.log_path.read_text(encoding="utf-8"))
        except Exception:
            return []

    def get_queue(self, status_filter: str | None = None) -> list[dict[str, Any]]:
        """Retrieve items from the review queue with their current operational status.

        Status filters:
        - None / 'all': all enqueued items
        - 'pending': items requiring approval without a valid decision
        - 'approved': approved items
        - 'downgraded': downgraded items
        - 'rejected': rejected items
        - 'stale': items whose claims/evidence changed after prior approval
        """
        try:
            items = json.loads(self.queue_path.read_text(encoding="utf-8"))
        except Exception:
            return []

        enriched: list[dict[str, Any]] = []
        for item in items:
            event_id = item.get("event_id")
            packet_fp = item.get("candidate_fingerprint") or ""
            requires_approval = item.get("requires_approval", False)

            decision = self.get_latest_decision_for_event(event_id)
            if decision is not None:
                dec_fp = decision.get("candidate_fingerprint") or ""
                if dec_fp and packet_fp and dec_fp != packet_fp:
                    computed_status = "STALE_RE_REVIEW_REQUIRED"
                else:
                    action = decision.get("action", "")
                    if action == ReviewAction.APPROVE.value:
                        computed_status = "APPROVED"
                    elif action == ReviewAction.REJECT.value:
                        computed_status = "REJECTED"
                    elif "DOWNGRADE" in action:
                        computed_status = "DOWNGRADED"
                    else:
                        computed_status = action
            else:
                computed_status = "PENDING_APPROVAL" if requires_approval else "PASS_THROUGH"

            item_copy = dict(item)
            item_copy["operational_status"] = computed_status
            item_copy["latest_decision"] = decision

            if status_filter:
                f_norm = status_filter.strip().upper()
                is_pending = (
                    f_norm in ("PENDING", "PENDING_APPROVAL")
                    and computed_status == "PENDING_APPROVAL"
                )
                if f_norm == "ALL" or is_pending:
                    enriched.append(item_copy)
                elif f_norm == "APPROVED" and computed_status == "APPROVED":
                    enriched.append(item_copy)
                elif f_norm == "DOWNGRADED" and computed_status == "DOWNGRADED":
                    enriched.append(item_copy)
                elif f_norm == "REJECTED" and computed_status == "REJECTED":
                    enriched.append(item_copy)
                elif f_norm == "STALE" and computed_status == "STALE_RE_REVIEW_REQUIRED":
                    enriched.append(item_copy)
            else:
                enriched.append(item_copy)

        return enriched

    def inspect_event(self, event_id: str) -> dict[str, Any] | None:
        """Retrieve complete inspection dossier for an event in the queue."""
        queue_items = self.get_queue()
        matched = next((item for item in queue_items if item.get("event_id") == event_id), None)
        if matched is not None:
            return matched

        # Fallback to review history if not currently in queue
        history = [e for e in self.get_review_history() if e.get("event_id") == event_id]
        if history:
            latest = history[-1]
            return {
                "event_id": event_id,
                "event_title": latest.get("event_title"),
                "operational_status": f"HISTORICAL_{latest.get('action')}",
                "latest_decision": latest,
                "history": history,
            }
        return None

    def review_event(
        self,
        event_id: str,
        action: ReviewAction | str,
        analyst_id: str,
        rationale: str,
    ) -> tuple[EventSummary | None, PilotReviewLogEntry]:
        """Apply an analyst decision to a queued event by event_id."""
        item = self.inspect_event(event_id)
        if not item or "packet" not in item:
            raise KeyError(f"Event ID '{event_id}' was not found in the pilot review queue.")

        pkt_data = item["packet"]
        claims = tuple(ReviewEvidenceInspection(**c) for c in pkt_data["claims_inspection"])
        lineages = tuple(ReviewLineageInspection(**lin) for lin in pkt_data["lineages_inspection"])

        packet = CandidateReviewPacket(
            event_id=pkt_data["event_id"],
            event_title=pkt_data["event_title"],
            category=pkt_data["category"],
            headline=pkt_data["headline"],
            proposed_status=pkt_data["proposed_status"],
            proposed_badge=pkt_data["proposed_badge"],
            proposed_chains=pkt_data["proposed_chains"],
            requires_approval=pkt_data["requires_approval"],
            review_trigger_reason=pkt_data["review_trigger_reason"],
            claims_inspection=claims,
            lineages_inspection=lineages,
            source_count=pkt_data["source_count"],
            article_count=pkt_data["article_count"],
            candidate_fingerprint=pkt_data.get("candidate_fingerprint", ""),
        )

        raw_summary = item.get("summary")
        if raw_summary:
            summary = EventSummary.model_validate(raw_summary)
        else:
            summary = EventSummary(
                event_id=packet.event_id,
                headline=packet.headline,
                what_happened=packet.headline,
                confidence_qualification=packet.proposed_badge,
                corroboration_status=packet.proposed_status,
                independent_chains_count=packet.proposed_chains,
                model_used="operator-pilot-review",
            )

        return self.apply_analyst_decision(
            packet=packet,
            summary=summary,
            action=action,
            analyst_id=analyst_id,
            analyst_rationale=rationale,
        )
