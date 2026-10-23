"""Prompt templates and builders for source-grounded event summarization."""

from geopulse.clustering.models import EventCluster

SYSTEM_PROMPT = """You are a senior geopolitical and defence intelligence analyst for Geopulse.
Your mission is to produce strictly factual, source-grounded event summaries for a concise
intelligence briefing.

CRITICAL RULES:
1. Output MUST be valid JSON only, with no markdown code fences, comments, or extra text.
2. NEVER invent casualties, dates, locations, military outcomes, quotations, or strategic motives.
3. If a detail is not explicitly supported by the provided sources, state
   'Unspecified in reporting' or omit it.
4. If sources conflict or an actor denies a claim, clearly preserve the disagreement.
   Never merge conflicting accounts into false certainty.
5. If only one source reported the event, clearly state that this is a single-source
   report that is not independently verified.
6. If the event is corroborated by multiple independent sources, note that it has
   multi-source corroboration.
7. Be concise, objective, and neutral. Target a compact format suitable for a Telegram digest.
8. If an event is marked as Strategic Analysis / Commentary, do NOT present it as a
   newly occurring military strike, operation, or breaking battlefield event. Summarize
   the analytical thesis, core strategic arguments, and policy implications. Never treat
   historical case studies or illustrative vignettes cited in an essay as newly occurring news.
9. For timing, if the source article does not specify a publication date, state
   'Date not specified in source'. NEVER assume the current date or invent a timestamp.
"""


def build_event_prompt(event: EventCluster) -> str:
    """Build the user prompt for an EventCluster using all attached source articles."""
    articles_text_blocks = []
    for idx, art in enumerate(event.articles, start=1):
        pub_str = (
            art.published_at.strftime("%Y-%m-%d %H:%M UTC")
            if art.published_at
            else "Date not specified in source (do NOT assume current date)"
        )
        snippet_str = art.snippet.strip() if art.snippet else "No snippet available."
        block = (
            f"--- Source Article {idx} ---\n"
            f"Outlet: {art.source_name}\n"
            f"Published: {pub_str}\n"
            f"Title: {art.title}\n"
            f"Snippet: {snippet_str}\n"
        )
        articles_text_blocks.append(block)

    articles_section = "\n".join(articles_text_blocks)

    event_nature = (
        "Strategic Analysis / Commentary (NOT a breaking news event)"
        if getattr(event, "is_analysis", False)
        else "Breaking News Development"
    )

    prompt = f"""Summarize this geopolitical event based SOLELY on the provided source articles.

EVENT METADATA:
Category: {event.category.value}
Event Nature: {event_nature}
Reported Region: {event.region or 'Unspecified'}
Clustering Confidence Label: {event.confidence_label}
Distinct Reporting Sources: {event.source_count} ({event.article_count} total reports)

SOURCE ARTICLES:
{articles_section}

REQUIRED JSON STRUCTURE:
{{
  "headline": "<concise, factual headline>",
  "what_happened": "<2-3 factual sentences summarizing the reported event or analytical thesis>",
  "key_details": {{
    "actors": ["<actor 1>", "<actor 2>"],
    "location": "<specific geographic location or 'Unspecified in reporting'>",
    "timing": "<original publication date, historical period, or 'Date not specified in source'>",
    "action": "<reported action or core strategic policy subject>"
  }},
  "significance": "<why this may matter strategically if supported, or null>",
  "confidence_qualification": "<explicit statement on single-source, disputed, or corroborated>"
}}
"""
    return prompt
