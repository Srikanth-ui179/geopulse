"""Service coordinating local Ollama event summarization, validation, and fallback."""

import json
from typing import Any

from geopulse.clustering.models import EventCluster
from geopulse.config import Settings, get_settings
from geopulse.summarization.client import BaseOllamaClient, OllamaClient
from geopulse.summarization.exceptions import (
    OllamaModelNotFoundError,
    OllamaResponseError,
    OllamaTimeoutError,
    OllamaUnavailableError,
)
from geopulse.summarization.models import (
    EventKeyDetails,
    EventSourceAttribution,
    EventSummary,
    SummarizationResult,
)
from geopulse.summarization.prompts import SYSTEM_PROMPT, build_event_prompt


class EventSummarizer:
    """Orchestrates source-grounded LLM summarization for clustered geopolitical events."""

    def __init__(
        self,
        client: BaseOllamaClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.client = client or OllamaClient(
            base_url=self.settings.ollama_base_url,
            model=self.settings.ollama_model,
        )

    def summarize_event(self, event: EventCluster) -> SummarizationResult:
        """Summarize a single EventCluster using Ollama."""
        # 1. Preserve original source attribution from source records
        sources = tuple(
            EventSourceAttribution(
                source_name=art.source_name,
                title=art.title,
                url=art.url,
                published_at=art.published_at,
            )
            for art in event.articles
        )

        # 2. Build strict source-grounded prompt
        prompt = build_event_prompt(event)

        # 3. Request LLM completion with graceful error handling
        try:
            raw_output = self.client.generate(
                prompt=prompt,
                system=SYSTEM_PROMPT,
                format_type="json",
            )
        except OllamaUnavailableError as exc:
            return SummarizationResult(
                event_id=event.id,
                status="unavailable",
                error_message=str(exc),
                event=event,
            )
        except (OllamaTimeoutError, OllamaModelNotFoundError, OllamaResponseError) as exc:
            return SummarizationResult(
                event_id=event.id,
                status="failed",
                error_message=str(exc),
                event=event,
            )
        except Exception as exc:
            return SummarizationResult(
                event_id=event.id,
                status="failed",
                error_message=f"Unexpected error during summarization: {exc}",
                event=event,
            )

        # 4. Parse and validate JSON structure
        parsed: dict[str, Any]
        try:
            clean_output = raw_output.strip()
            # Strip markdown code blocks if the model wrapped output in ```json
            if clean_output.startswith("```"):
                lines = clean_output.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                clean_output = "\n".join(lines).strip()

            parsed = json.loads(clean_output)
            if not isinstance(parsed, dict):
                raise ValueError("Model output did not parse into a JSON object dictionary")
        except Exception as exc:
            return SummarizationResult(
                event_id=event.id,
                status="failed",
                error_message=f"Model output could not be parsed as valid JSON: {exc}",
                event=event,
            )

        # 5. Extract and validate required textual fields
        headline = str(parsed.get("headline") or "").strip()
        what_happened = str(parsed.get("what_happened") or "").strip()
        if len(headline) < 5 or len(what_happened) < 10:
            return SummarizationResult(
                event_id=event.id,
                status="failed",
                error_message=(
                    "Summary validation failed: 'headline' or 'what_happened' is missing or empty"
                ),
                event=event,
            )

        # 6. Extract key details
        raw_key_details = parsed.get("key_details")
        if isinstance(raw_key_details, dict):
            raw_actors = raw_key_details.get("actors", [])
            if isinstance(raw_actors, list):
                actors = tuple(str(a).strip() for a in raw_actors if str(a).strip())
            else:
                actors = ()
            location = str(raw_key_details.get("location") or "").strip() or None
            timing = str(raw_key_details.get("timing") or "").strip() or None
            action = str(raw_key_details.get("action") or "").strip() or None
            key_details = EventKeyDetails(
                actors=actors,
                location=location,
                timing=timing,
                action=action,
            )
        else:
            key_details = EventKeyDetails()

        # 7. Extract confidence qualification or fallback deterministically
        confidence_qual = str(parsed.get("confidence_qualification") or "").strip()
        slug_norm = confidence_qual.lower().replace("-", "_").replace(" ", "_").strip()
        slugs_to_expand = {
            "single_source",
            "single_source_report",
            "corroborated",
            "multi_source_corroboration",
            "multi_source_corroborated",
            "disputed",
            "disputed_report",
            "uncertain",
            "unconfirmed",
        }

        if not confidence_qual or len(confidence_qual) < 15 or slug_norm in slugs_to_expand:
            if event.confidence_label == "disputed":
                confidence_qual = (
                    "Disputed reporting: contradictory claims reported across belligerents."
                )
            elif event.confidence_label == "single_source":
                source_name = event.articles[0].source_name if event.articles else "Single source"
                confidence_qual = (
                    f"Single-source report from {source_name}; unverified independently."
                )
            elif event.confidence_label == "corroborated":
                confidence_qual = (
                    f"Corroborated across {event.source_count} reporting outlets; "
                    "military claims unconfirmed independently."
                )
            else:
                confidence_qual = "Reporting details remain preliminary or unconfirmed."

        # Enforce consistency between event confidence label, source count, and qualification
        conf_lower = confidence_qual.lower()
        if event.confidence_label == "single_source":
            multi_mentioned = (
                "multiple outlets" in conf_lower or "multiple sources" in conf_lower
            )
            if "corroborat" in conf_lower or (event.source_count <= 1 and multi_mentioned):
                source_name = event.articles[0].source_name if event.articles else "Single source"
                if event.source_count >= 2:
                    confidence_qual = (
                        "Single-origin reporting across outlets; unverified independently."
                    )
                else:
                    confidence_qual = (
                        f"Single-source report from {source_name}; unverified independently."
                    )
        elif event.confidence_label == "corroborated" and event.source_count >= 2:
            if (
                "unconfirmed" not in conf_lower
                and "unverified" not in conf_lower
                and "not" not in conf_lower
            ):
                confidence_qual = (
                    f"Corroborated across {event.source_count} reporting outlets; "
                    "military claims unconfirmed independently."
                )

        raw_sig = parsed.get("significance")
        if raw_sig is not None:
            cleaned_sig = str(raw_sig).strip()
            if cleaned_sig.lower() in {
                "null",
                "none",
                "n/a",
                "nil",
                "unspecified",
                "unspecified in reporting",
            } or not cleaned_sig:
                significance = None
            else:
                significance = cleaned_sig
        else:
            significance = None

        # 8. Construct validated EventSummary
        summary = EventSummary(
            event_id=event.id,
            headline=headline,
            what_happened=what_happened,
            key_details=key_details,
            significance=significance,
            confidence_qualification=confidence_qual,
            sources=sources,
            model_used=self.client.model,
        )

        return SummarizationResult(
            event_id=event.id,
            status="success",
            summary=summary,
            event=event,
        )

    def summarize_events(
        self,
        events: list[EventCluster] | tuple[EventCluster, ...],
    ) -> tuple[SummarizationResult, ...]:
        """Summarize multiple EventClusters sequentially."""
        return tuple(self.summarize_event(event) for event in events)
