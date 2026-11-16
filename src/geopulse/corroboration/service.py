"""Corroboration evaluation service for claims and geopolitical events."""

from collections.abc import Sequence

from geopulse.clustering.models import EventCluster
from geopulse.corroboration.lineage import build_cluster_lineages, extract_article_lineage
from geopulse.corroboration.models import (
    ArticleLineage,
    ClaimCorroboration,
    CorroborationStatus,
    EventCorroboration,
    EvidenceLabel,
)
from geopulse.ingestion.models import ArticleRecord
from geopulse.summarization.models import EventSummary, SummaryClaim
from geopulse.summarization.validator import normalize_url


def corroborate_claim(
    claim: SummaryClaim,
    articles: Sequence[ArticleRecord],
    lineages: dict[str, ArticleLineage] | None = None,
    contradictory_claims: Sequence[SummaryClaim] | None = None,
) -> ClaimCorroboration:
    """Evaluate demonstrable independent corroboration for a single claim.

    Strict rules:
    - Does not equate article count with independent confirmation.
    - Multiple articles citing the same wire or official statement collapse to 1 chain.
    - Unknown sources NEVER count as independent confirmation.
    - Contradictory evidence triggers 'conflicting_evidence'.
    """
    if lineages is None:
        lineages = build_cluster_lineages(articles)

    article_by_url: dict[str, ArticleRecord] = {}
    for a in articles:
        if a.url:
            article_by_url[normalize_url(a.url)] = a
        if a.canonical_url:
            article_by_url[normalize_url(a.canonical_url)] = a

    # 1. Identify supporting articles from accepted evidence references
    supporting_urls: list[str] = []
    for ref in claim.evidence_refs:
        if ref.verification_status == "rejected":
            continue
        norm_ref = normalize_url(ref.source_url)
        if norm_ref in article_by_url and norm_ref not in supporting_urls:
            supporting_urls.append(norm_ref)

    # 2. Extract lineages for supporting articles
    supporting_lineages: list[ArticleLineage] = []
    for url in supporting_urls:
        lin = lineages.get(url)
        if lin is None:
            # Fallback
            art = article_by_url.get(url)
            if art:
                lin = extract_article_lineage(art)
        if lin:
            supporting_lineages.append(lin)

    # 3. Partition into independent chains vs derivative/syndicated vs unknown
    chains_by_origin: dict[str, list[ArticleLineage]] = {}
    derivative_urls: list[str] = []
    unresolved_sources: list[str] = []

    for lin in supporting_lineages:
        if lin.lineage_classification == "unknown" or lin.original_source.strip().lower() in (
            "unknown",
            "",
        ):
            unresolved_sources.append(lin.publisher or lin.article_url)
            continue

        orig = lin.original_source.strip()
        if lin.lineage_classification in (
            "syndicated",
            "derivative",
            "wire_service_citation",
        ):
            derivative_urls.append(lin.article_url)

        chains_by_origin.setdefault(orig.lower(), []).append(lin)

    # Distinct demonstrable independent chains (excluding unknown)
    independent_origins = list(chains_by_origin.keys())
    independent_chains_count = len(independent_origins)

    # 4. Check for contradictory evidence
    contradictions: list[str] = []
    for ref in claim.evidence_refs:
        if ref.nli_classification == "contradiction":
            contradictions.append(
                f"Contradictory evidence span in {ref.source_url}: {ref.evidence_span[:100]}"
            )

    if claim.classification == "disputed_claim":
        contradictions.append(f"Claim explicitly flagged as disputed: {claim.text[:100]}")

    if contradictory_claims:
        for c in contradictory_claims:
            if c.claim_id != claim.claim_id and (
                c.classification == "disputed_claim" or (c.nli_classification == "contradiction")
            ):
                contradictions.append(f"Contradicting claim [{c.claim_id}]: {c.text[:100]}")

    # 5. Determine CorroborationStatus and Explanation
    status: CorroborationStatus
    explanation: str
    orig_names = [chains_by_origin[k][0].original_source for k in independent_origins]

    if contradictions:
        status = "conflicting_evidence"
        explanation = (
            f"Contradictory evidence identified across sources: "
            f"{len(contradictions)} conflicting accounts."
        )
    elif independent_chains_count >= 2:
        status = "independently_corroborated"
        explanation = (
            f"Independently corroborated across {independent_chains_count} distinct reporting "
            f"chains ({', '.join(sorted(orig_names))})."
        )
    elif independent_chains_count == 1:
        origin_name = orig_names[0]
        chain_lineages = chains_by_origin[independent_origins[0]]
        has_shared_repetition = len(supporting_urls) >= 2 or any(
            lin.lineage_classification
            in ("wire_service_citation", "official_statement", "syndicated", "derivative")
            for lin in chain_lineages
        )
        if has_shared_repetition and len(supporting_urls) >= 2:
            status = "multiple_reports_shared_origin"
            first_class = chain_lineages[0].lineage_classification
            if first_class == "wire_service_citation":
                explanation = (
                    f"Reported across {len(supporting_urls)} publishers, but all rely on a single "
                    f"shared wire report ({origin_name}); 1 independent evidence chain."
                )
            elif first_class == "official_statement":
                explanation = (
                    f"Reported across {len(supporting_urls)} publishers relying on the same "
                    f"official announcement ({origin_name}); not independently confirmed."
                )
            else:
                explanation = (
                    f"Multiple reports ({len(supporting_urls)}) originating from a single "
                    f"primary source ({origin_name})."
                )
        else:
            status = "single_source"
            explanation = f"Single-source reporting from {origin_name}; unverified independently."
    elif len(supporting_urls) == 1:
        # A single supporting article with unproven lineage is still a single-source report
        lin = supporting_lineages[0] if supporting_lineages else None
        origin_name = (
            lin.publisher
            if lin and lin.publisher and lin.publisher.strip().lower() != "unknown"
            else "Single source"
        )
        status = "single_source"
        independent_chains_count = 1
        orig_names = [origin_name]
        explanation = f"Single-source reporting from {origin_name}; unverified independently."
    else:
        # Multiple supporting articles but source provenance unresolved or zero proven chains
        status = "insufficient_information"
        explanation = (
            "Source provenance unresolved or insufficient evidence to verify independent reporting."
        )

    # Determine transparent editorial evidence label
    if status == "independently_corroborated":
        evidence_label: EvidenceLabel = "Independently corroborated"
    elif status in ("multiple_reports_shared_origin", "conflicting_evidence") or (
        len(supporting_urls) > 1 and independent_chains_count == 1
    ):
        evidence_label = "Partially corroborated"
    else:
        evidence_label = "Single-source"

    return ClaimCorroboration(
        claim_id=claim.claim_id,
        claim_text=claim.text,
        status=status,
        evidence_label=evidence_label,
        independent_evidence_chains=independent_chains_count,
        original_sources=tuple(orig_names),
        supporting_articles=tuple(supporting_urls),
        derivative_articles=tuple(derivative_urls),
        contradictory_evidence=tuple(contradictions),
        unresolved_relationships=tuple(unresolved_sources),
        explanation=explanation,
    )


def corroborate_event(
    event: EventCluster,
    summary: EventSummary | None = None,
) -> EventCorroboration:
    """Evaluate aggregate independent corroboration for an entire event cluster."""
    lineages_map = build_cluster_lineages(event.articles)
    all_lineages = tuple(
        lineages_map[a.canonical_url or a.url]
        for a in event.articles
        if (a.canonical_url or a.url) in lineages_map
    )

    # 1. Evaluate claim-level corroboration if summary claims exist
    claims_corrob: list[ClaimCorroboration] = []
    if summary and summary.claims:
        for claim in summary.claims:
            c_res = corroborate_claim(
                claim=claim,
                articles=event.articles,
                lineages=lineages_map,
                contradictory_claims=[c for c in summary.claims if c.claim_id != claim.claim_id],
            )
            claims_corrob.append(c_res)

    # 2. Determine aggregate event status and independent chains count
    if claims_corrob:
        # Prioritize central claims if available
        central_claim_results = [
            c_res
            for c_res, claim in zip(claims_corrob, summary.claims)
            if getattr(claim, "central", False)
        ]
        target_results = central_claim_results if central_claim_results else claims_corrob

        has_conflict = any(c.status == "conflicting_evidence" for c in target_results)
        all_corroborated = all(c.status == "independently_corroborated" for c in target_results)
        has_shared_origin = any(
            c.status == "multiple_reports_shared_origin" for c in target_results
        )
        all_insufficient = all(c.status == "insufficient_information" for c in target_results)

        # Max independent chains across target claims
        chains_count = (
            min(c.independent_evidence_chains for c in target_results) if target_results else 1
        )

        if has_conflict:
            event_status = "conflicting_evidence"
            summary_expl = "Contested reporting; conflicting accounts identified across evidence."
            primary_origin = None
        elif all_corroborated and chains_count >= 2:
            event_status = "independently_corroborated"
            summary_expl = (
                f"Independently corroborated across {chains_count} distinct reporting chains."
            )
            primary_origin = None
        elif has_shared_origin:
            event_status = "multiple_reports_shared_origin"
            shared_c = next(
                (c for c in target_results if c.status == "multiple_reports_shared_origin"), None
            )
            primary_origin = (
                shared_c.original_sources[0] if shared_c and shared_c.original_sources else None
            )
            summary_expl = (
                f"Multiple articles reporting from shared origin: "
                f"{primary_origin or 'shared source'}."
            )
        elif all_insufficient:
            event_status = "insufficient_information"
            summary_expl = (
                "Preliminary reporting; insufficient information to establish independent sources."
            )
            primary_origin = None
        else:
            event_status = "single_source"
            primary_origin = (
                target_results[0].original_sources[0]
                if target_results[0].original_sources
                else None
            )
            summary_expl = "Single-source reporting; not independently corroborated."
    else:
        # Cluster-level assessment directly from articles
        chains_by_origin: dict[str, list[ArticleLineage]] = {}
        unresolved: list[str] = []

        for lin in all_lineages:
            if lin.lineage_classification == "unknown" or lin.original_source.strip().lower() in (
                "unknown",
                "",
            ):
                unresolved.append(lin.publisher)
            else:
                chains_by_origin.setdefault(lin.original_source.lower(), []).append(lin)

        chains_count = len(chains_by_origin)
        conf_label = getattr(event, "confidence_label", None)
        if conf_label == "disputed":
            event_status = "conflicting_evidence"
            primary_origin = None
            summary_expl = "Contested reporting; conflicting accounts identified across evidence."
        elif chains_count == 0:
            if len(event.articles) == 1:
                event_status = "single_source"
                chains_count = 1
                primary_origin = event.articles[0].source_name
                summary_expl = f"Single-source from {primary_origin}; unverified independently."
            else:
                event_status = "insufficient_information"
                primary_origin = None
                summary_expl = "Unresolved source provenance; independent confirmation unavailable."
        elif chains_count == 1:
            orig = next(iter(chains_by_origin.values()))[0].original_source
            primary_origin = orig
            if len(event.articles) > 1:
                event_status = "multiple_reports_shared_origin"
                summary_expl = (
                    f"Multiple reports ({len(event.articles)}) originating from shared source: "
                    f"{orig}."
                )
            else:
                event_status = "single_source"
                summary_expl = f"Single-source report from {orig}; unverified independently."
        elif chains_count >= 2:
            if conf_label == "single_source":
                event_status = "single_source"
                primary_origin = next(iter(chains_by_origin.values()))[0].original_source
                summary_expl = "Single-source reporting; not independently corroborated."
            else:
                event_status = "independently_corroborated"
                primary_origin = None
                summary_expl = (
                    f"Independently corroborated across {chains_count} distinct reporting chains."
                )
        else:
            event_status = "insufficient_information"
            primary_origin = None
            summary_expl = "Unresolved source provenance; independent confirmation unavailable."

    if event_status == "independently_corroborated" and chains_count >= 2:
        event_evidence_label: EvidenceLabel = "Independently corroborated"
    elif event_status in ("multiple_reports_shared_origin", "conflicting_evidence") or (
        len(event.articles) > 1 and chains_count == 1
    ):
        event_evidence_label = "Partially corroborated"
    else:
        event_evidence_label = "Single-source"

    return EventCorroboration(
        event_id=event.id,
        status=event_status,
        evidence_label=event_evidence_label,
        independent_chains_count=chains_count,
        primary_origin=primary_origin,
        claims_corroboration=tuple(claims_corrob),
        lineages=all_lineages,
        summary_explanation=summary_expl,
    )
