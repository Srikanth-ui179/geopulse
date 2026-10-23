"""Configurable rule-based geopolitical and defence relevance classifier."""

import re
from typing import Literal

RelevanceStatus = Literal["pending", "relevant", "irrelevant", "review"]

# Sources that exclusively cover military affairs, defence, and geopolitics
DEDICATED_DEFENCE_SOURCES: frozenset[str] = frozenset(
    {
        "defense news",
        "breaking defense",
        "war on the rocks",
        "uk ministry of defence",
    }
)

# High-signal conflict, military, and defence keywords (word-boundary matched)
DEFENCE_KEYWORDS: tuple[str, ...] = (
    r"\bwar\b",
    r"\bwars\b",
    r"\bwarfare\b",
    r"\bconflict\b",
    r"\bconflicts\b",
    r"\bmilitary\b",
    r"\bdefence\b",
    r"\bdefense\b",
    r"\bmissile\b",
    r"\bmissiles\b",
    r"\bairstrike\b",
    r"\bairstrikes\b",
    r"\bair strike\b",
    r"\bair strikes\b",
    r"\btroop\b",
    r"\btroops\b",
    r"\bsoldier\b",
    r"\bsoldiers\b",
    r"\barmy\b",
    r"\bnavy\b",
    r"\bnaval\b",
    r"\bair force\b",
    r"\bceasefire\b",
    r"\bdrone strike\b",
    r"\bdrone strikes\b",
    r"\bweapons\b",
    r"\bammunition\b",
    r"\bnato\b",
    r"\bpentagon\b",
    r"\bcounteroffensive\b",
    r"\bartillery\b",
    r"\bballistic\b",
    r"\bnuclear\b",
    r"\binsurgency\b",
    r"\bsanctions\b",
    r"\bfrontline\b",
    r"\bfrontlines\b",
    r"\bpeacekeeping\b",
    r"\bcombat\b",
    r"\bshelling\b",
    r"\binvasion\b",
    r"\brebel\b",
    r"\brebels\b",
    r"\bmilitia\b",
    r"\bmilitias\b",
    r"\barmed forces\b",
)

# Irrelevant topic signals (sports, entertainment, lifestyle, medical therapy,
# civilian wildfires, routine regulations, consumer tech, and routine domestic politics)
IRRELEVANT_KEYWORDS: tuple[str, ...] = (
    # Sports & entertainment
    r"\bfootball\b",
    r"\bsoccer\b",
    r"\bcricket\b",
    r"\bpremier league\b",
    r"\bnfl\b",
    r"\bnba\b",
    r"\btennis\b",
    r"\bgolf\b",
    r"\bbox office\b",
    r"\bcelebrity\b",
    r"\bhollywood\b",
    r"\boscar\b",
    r"\bgrammy\b",
    r"\bfilm festival\b",
    r"\bolympics?\b",
    r"\bdoping\b",
    r"\bmarathon runner\b",
    r"\brace walker\b",
    # Lifestyle & wellness
    r"\brecipe\b",
    r"\brecipes\b",
    r"\blifestyle\b",
    r"\bhoroscope\b",
    r"\bfashion week\b",
    r"\bdating\b",
    # Medical, therapy, and health
    r"\bpost-traumatic stress\b",
    r"\bptsd\b",
    r"\btreating post-traumatic\b",
    r"\btherapy\b",
    r"\btherapies\b",
    r"\bpsychotherapy\b",
    r"\bmental health\b",
    r"\btrauma recovery\b",
    r"\bclinical trial\b",
    r"\bcancer\b",
    r"\bbreast cancer\b",
    r"\bchemotherapy\b",
    # Civilian emergency, weather, wildfires, natural disaster relief
    r"\bfight wildfires\b",
    r"\bwildfires?\b",
    r"\bbushfires?\b",
    r"\bforest fires?\b",
    r"\bfirefighters?\b",
    r"\bfirefighting\b",
    r"\btyphoon\b",
    r"\bhurricane\b",
    r"\bcyclone\b",
    r"\btornado\b",
    r"\bearthquake\b",
    r"\btsunami\b",
    r"\bflights? cancelled\b",
    # Routine civil demonstrations & transit protests
    r"\bweekend protests?\b",
    # Routine administrative regulations & paperwork
    r"\bnotice of proposed amendments\b",
    r"\bnpa documents?\b",
    r"\bregulatory compliance notice\b",
    # Routine administrative court notices, court listings, and generic sentencing transcripts
    r"\btransparency data:\s*military court\b",
    r"\b(?:court listings? (?:for|at|of)?\s*military court(?: centres?)?|"
    r"military court(?: centres?)? court listings?)\b",
    r"\b(?:military )?court[- ]martial (?:court )?listings?\b",
    r"\bsentencing remarks in (?:the )?court[- ]martial\b",
    r"\bmilitary court service sentencing remarks\b",
    r"\btranscripts of sentencing remarks\b",
    # Personal conduct and domestic scandals
    r"\binappropriate relationship\b",
    r"\bresigns over inappropriate\b",
    # Routine domestic politics & electoral campaigns (without armed crisis/coup)
    r"\bsnap election\b",
    r"\bgeneral election\b",
    r"\belection campaign\b",
    r"\bparliamentary elections?\b",
    r"\bvoters? head to polls?\b",
    r"\bstudents? unseat\b",
    r"\bopinion polls?\b",
    r"\bmayoral race\b",
    # Consumer electronics & retail gadgets
    r"\bnew electronics reach\b",
    r"\belectronics reach\b",
    r"\bsmartphones?\b",
    r"\bconsumer devices?\b",
    r"\bprices soar as gov['’]?t fees\b",
    # Routine financial & business inquiries
    r"\binvestigation into financial affairs\b",
    r"\bfinancial affairs of\b",
    r"\btax fraud probe\b",
    # Sponsored marketing, advertorials, and vendor promotions (no timely intelligence value)
    r"\[sponsored\]",
    r"\bsponsored content\b",
    r"\badvertorial\b",
    r"\bpartner content\b",
    r"\bsponsored by\b",
    # Pure abstract doctrine monographs & historical strategy essays without timely hook
    r"\bdeception in mass autonomous warfare\b",
    r"\bdoctrinal essay\b",
    r"\btheoretical monograph\b",
    r"\bhistorical case studies in warfare\b",
    # Retrospective overviews and comparative lookback essays without timely operational hook
    r"\bthen and now: how the landscape has changed\b",
    r"\bhow the landscape has changed since\b",
    r"\bretrospective overview\b",
    r"\bhistorical lookback on the war\b",
)

_DEFENCE_PATTERN = re.compile("|".join(DEFENCE_KEYWORDS), re.IGNORECASE)
_IRRELEVANT_PATTERN = re.compile("|".join(IRRELEVANT_KEYWORDS), re.IGNORECASE)
_KINETIC_OVERRIDE_PATTERN = re.compile(
    r"\b(soldiers? killed|soldiers? wounded|casualties|killed in action|"
    r"gun battle|gunmen open fire|fatal shooting|shooting attack|attackers? killed|"
    r"attackers? shot dead|airstrike|bombing|shelling|militants ambush)\b",
    re.IGNORECASE,
)
_SPORTS_OR_LIFESTYLE_PATTERN = re.compile(
    r"\b(premier league|football|soccer|nba|nfl|tennis|golf|celebrity|"
    r"hollywood|box office|horoscope|lifestyle|recipe|dating)\b",
    re.IGNORECASE,
)
_STRATEGIC_LEGAL_OVERRIDE_PATTERN = re.compile(
    r"\b(treason|espionage|classified|secrets?|war crimes?|atrocit\w*|"
    r"homicide|murder|bribery|corruption|reform|mutiny|sabotage|trafficking)\b",
    re.IGNORECASE,
)


def classify_relevance(
    title: str,
    snippet: str | None = None,
    source_name: str | None = None,
) -> RelevanceStatus:
    """Classify an article's initial relevance using source attribution and keyword heuristics."""
    clean_source = (source_name or "").strip().lower()
    combined_text = f"{title} {snippet or ''}"

    # Check for non-geopolitical / irrelevant domains first (even for dedicated defence outlets)
    if _IRRELEVANT_PATTERN.search(combined_text):
        # Allow kinetic attacks or high-consequence legal cases to override incidental hits,
        # but never override genuine sports, celebrity, or lifestyle reporting.
        has_kinetic = bool(_KINETIC_OVERRIDE_PATTERN.search(combined_text))
        has_strategic_legal = bool(_STRATEGIC_LEGAL_OVERRIDE_PATTERN.search(combined_text))
        is_sports = bool(_SPORTS_OR_LIFESTYLE_PATTERN.search(combined_text))
        if (has_kinetic or has_strategic_legal) and not is_sports:
            is_defence = clean_source in DEDICATED_DEFENCE_SOURCES
            if is_defence or bool(_DEFENCE_PATTERN.search(combined_text)):
                return "relevant"
            return "pending"
        return "irrelevant"

    # Dedicated defence and military outlets are relevant by default for military coverage
    if clean_source in DEDICATED_DEFENCE_SOURCES:
        return "relevant"

    # Check for direct geopolitical and defence keywords
    if _DEFENCE_PATTERN.search(combined_text):
        return "relevant"

    # General news without explicit keyword matches remains pending for later analysis
    return "pending"
