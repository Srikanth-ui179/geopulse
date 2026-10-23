"""Dedicated Telegram digest formatter producing source-grounded HTML briefings."""

import html
from collections.abc import Sequence
from datetime import date, datetime

from geopulse.classification.models import TopicCategory
from geopulse.clustering.models import ConfidenceLabel
from geopulse.summarization.models import EventSummary, SummarizationResult

# Maximum character length per Telegram message
MAX_TELEGRAM_MESSAGE_LENGTH: int = 4096
# Target safe limit per chunk to leave headroom for Telegram metadata
SAFE_CHUNK_LIMIT: int = 4000

CATEGORY_LABELS: dict[TopicCategory, str] = {
    TopicCategory.ARMED_CONFLICT: "⚔️ ARMED CONFLICT",
    TopicCategory.MILITARY_OPERATIONS: "🛡️ MILITARY OPERATIONS",
    TopicCategory.DEFENCE_POLICY: "🏛️ DEFENCE POLICY",
    TopicCategory.WEAPONS_PROCUREMENT: "🚀 WEAPONS PROCUREMENT",
    TopicCategory.DIPLOMACY: "🕊️ DIPLOMACY",
    TopicCategory.SANCTIONS: "⚖️ SANCTIONS",
    TopicCategory.TERRORISM_INSURGENCY: "⚠️ TERRORISM & INSURGENCY",
    TopicCategory.CYBER_HYBRID_WARFARE: "💻 CYBER & HYBRID WARFARE",
    TopicCategory.SECURITY_INCIDENT: "🚨 SECURITY INCIDENT",
    TopicCategory.OTHER_GEOPOLITICAL: "🌐 GEOPOLITICAL",
    TopicCategory.UNCERTAIN: "❓ UNCERTAIN",
}

CONFIDENCE_BADGES: dict[ConfidenceLabel, str] = {
    "corroborated": "[CORROBORATED]",
    "single_source": "[SINGLE SOURCE]",
    "disputed": "[DISPUTED]",
    "uncertain": "[PRELIMINARY]",
}


def escape_html(text: str | None) -> str:
    """Safely escape plain text for Telegram HTML parse mode."""
    if not text:
        return ""
    return html.escape(str(text), quote=True)


class TelegramDigestFormatter:
    """Formats event summarization results into clean, length-bounded Telegram HTML messages."""

    def __init__(self, safe_limit: int = SAFE_CHUNK_LIMIT) -> None:
        self.safe_limit = safe_limit

    def format_event_block(
        self,
        summary: EventSummary,
        category: TopicCategory,
        confidence_label: ConfidenceLabel,
        rank: int,
        total: int,
        is_analysis: bool = False,
        published_at: datetime | None = None,
    ) -> str:
        """Format a single event summary into a structured HTML block."""
        cat_badge = CATEGORY_LABELS.get(category, "🌐 GEOPOLITICAL")
        conf_badge = CONFIDENCE_BADGES.get(confidence_label, "[REPORT]")

        if is_analysis:
            header_badge = f"<b>[#{rank}/{total}] {cat_badge} · [ANALYSIS] · {conf_badge}</b>"
        else:
            header_badge = f"<b>[#{rank}/{total}] {cat_badge} · {conf_badge}</b>"

        lines: list[str] = [
            header_badge,
            f"<b>{escape_html(summary.headline)}</b>\n",
            escape_html(summary.what_happened),
        ]

        # Key details (omit absent or unspecified fields)
        kd = summary.key_details
        kd_bullets: list[str] = []
        if kd.actors:
            clean_actors = [escape_html(a) for a in kd.actors if a and a.lower() != "unspecified"]
            if clean_actors:
                kd_bullets.append(f"• <b>Actors:</b> {', '.join(clean_actors)}")
        if kd.location and kd.location.lower() not in {"unspecified", "unspecified in reporting"}:
            kd_bullets.append(f"• <b>Location:</b> {escape_html(kd.location)}")

        # For analysis, explicitly display verified original publication date
        if is_analysis:
            if published_at is not None:
                kd_bullets.append(f"• <b>Published:</b> {published_at.strftime('%B %d, %Y')}")
            else:
                kd_bullets.append("• <b>Published:</b> Date not specified in source")
        elif kd.timing and kd.timing.lower() not in {"unspecified", "unspecified in reporting"}:
            kd_bullets.append(f"• <b>Timing:</b> {escape_html(kd.timing)}")

        if kd.action and kd.action.lower() not in {"unspecified", "unspecified in reporting"}:
            kd_bullets.append(f"• <b>Action:</b> {escape_html(kd.action)}")

        if kd_bullets:
            lines.append("\n<b>Key Details:</b>\n" + "\n".join(kd_bullets))

        # Strategic significance (omit if null or unspecified)
        if summary.significance:
            sig_clean = summary.significance.strip()
            if sig_clean.lower() not in {
                "null",
                "none",
                "n/a",
                "nil",
                "unspecified",
                "unspecified in reporting",
            }:
                lines.append(f"\n<b>Significance:</b> {escape_html(sig_clean)}")

        # Confidence and uncertainty assessment (enforcing badge and assessment consistency)
        if summary.confidence_qualification:
            assessment_text = summary.confidence_qualification.strip()
            # Guardrail: single source must never be described as corroborated
            if confidence_label == "single_source" and "corroborat" in assessment_text.lower():
                assessment_text = "Single-source reporting; unverified independently."
            lines.append(f"\n<b>Assessment:</b> {escape_html(assessment_text)}")

        # Source attribution with clickable hyperlinks
        if summary.sources:
            lines.append("\n<b>Sources:</b>")
            for src in summary.sources:
                escaped_url = escape_html(src.url)
                escaped_outlet = escape_html(src.source_name)
                escaped_title = escape_html(src.title)
                lines.append(f'• <a href="{escaped_url}">{escaped_outlet}</a>: {escaped_title}')

        return "\n".join(lines)

    def format_digest(
        self,
        results: Sequence[SummarizationResult],
        briefing_date: date | None = None,
    ) -> tuple[str, ...]:
        """Format multiple summarization results into one or more Telegram message chunks.

        Splits across message boundaries cleanly between events, ensuring no message
        exceeds safe character limits and no HTML tags are broken.
        """
        successful = [r for r in results if r.status == "success" and r.summary is not None]
        # Ensure operational breaking news precedes analytical commentary
        ordered_successful = sorted(
            successful,
            key=lambda r: 1 if getattr(r.event, "is_analysis", False) else 0,
        )
        target_date = briefing_date or date.today()
        date_str = target_date.strftime("%B %d, %Y")

        if not ordered_successful:
            empty_msg = (
                f"<b>🌍 GEOPULSE DAILY BRIEFING</b> — {date_str}\n\n"
                "<i>No corroborated or verified geopolitical events met briefing thresholds today. "
                "Monitoring continues across all configured defence feeds.</i>\n\n"
                "<i>Geopulse Intelligence • Source-grounded briefing</i>"
            )
            return (empty_msg,)

        total_events = len(ordered_successful)
        event_blocks: list[str] = []
        for idx, res in enumerate(ordered_successful, start=1):
            assert res.summary is not None
            evt = res.event
            block = self.format_event_block(
                summary=res.summary,
                category=evt.category,
                confidence_label=evt.confidence_label,
                rank=idx,
                total=total_events,
                is_analysis=getattr(evt, "is_analysis", False),
                published_at=getattr(evt, "published_at", None),
            )
            event_blocks.append(block)

        # Build chunks respecting self.safe_limit
        return self._pack_chunks(event_blocks, date_str, total_events)

    def _pack_chunks(
        self,
        event_blocks: list[str],
        date_str: str,
        total_events: int,
    ) -> tuple[str, ...]:
        """Pack event blocks into message chunks bounded by character limits."""
        # Preliminary test: can everything fit in a single message?
        single_header = (
            f"<b>🌍 GEOPULSE DAILY INTELLIGENCE BRIEFING</b>\n"
            f"<i>{date_str} • {total_events} event(s) briefed</i>\n\n"
        )
        footer = "\n\n<i>Geopulse Intelligence • Source-grounded briefing</i>"
        full_single = single_header + "\n\n".join(event_blocks) + footer

        if len(full_single) <= self.safe_limit:
            return (full_single,)

        # Multi-part packing: assign event blocks into chunk groups
        chunks_events: list[list[str]] = []
        current_group: list[str] = []

        for block in event_blocks:
            current_group.append(block)
            # Estimate chunk size with conservative header/footer overhead
            estimated_size = (
                len(f"<b>🌍 GEOPULSE BRIEFING (Part 99/99) — {date_str}</b>\n\n")
                + sum(len(b) + 2 for b in current_group)
                + len(footer)
            )
            if estimated_size > self.safe_limit and len(current_group) > 1:
                # Pop the block that caused overflow and start next group
                current_group.pop()
                chunks_events.append(list(current_group))
                current_group = [block]

        if current_group:
            chunks_events.append(current_group)

        total_parts = len(chunks_events)
        result_chunks: list[str] = []

        for part_idx, group in enumerate(chunks_events, start=1):
            if total_parts > 1:
                header = (
                    f"<b>🌍 GEOPULSE BRIEFING (Part {part_idx}/{total_parts})</b>\n"
                    f"<i>{date_str}</i>\n\n"
                )
            else:
                header = single_header

            is_last = part_idx == total_parts
            body = "\n\n".join(group)
            part_msg = header + body + (footer if is_last else "")
            result_chunks.append(part_msg)

        return tuple(result_chunks)


def format_telegram_digest(
    results: Sequence[SummarizationResult],
    briefing_date: date | None = None,
    safe_limit: int = SAFE_CHUNK_LIMIT,
) -> tuple[str, ...]:
    """Convenience helper to format summarization results into Telegram HTML chunks."""
    formatter = TelegramDigestFormatter(safe_limit=safe_limit)
    return formatter.format_digest(results=results, briefing_date=briefing_date)
