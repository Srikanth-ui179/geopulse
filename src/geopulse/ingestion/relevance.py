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
    r"\banti[- ]aircraft\b",
    r"\bshot down\b",
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
    r"(?<!\braf\s)(?<!\bbritish\s)(?<!\buk\s)(?<!\beurofighter\s)\btyphoons?\b(?!\s+(?:fighters?|jets?|aircraft|fleet|squadron|pilot|pilots))\b",
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
    # Routine service personnel administrative guidance & domestic HR processes
    r"\b(?:established long-term relationship|relationship status for service personnel|"
    r"service family accommodation|civilian accommodation allowance|forces help to buy|"
    r"relocation allowance|pension scheme updates?|service personnel welfare benefits|"
    r"recording and approving a defence-recognised)\b",
    # Routine environmental estate management, forestry, flora & fauna guidance
    r"\b(?:ash dieback|estate tree safety|invasive non-native species|woodland management plan|"
    r"forestry on the .*?estate|deer management|conservation on the .*?estate|"
    r"ash dieback on the (?:ministry of defence|mod) estate|"
    r"woodlands across the (?:ministry of defence|mod) training estate)\b",
    # Routine civilian firing range clearance notices & public access schedules
    r"\b(?:firing times|range firing times|firing notice|firing schedule|"
    r"range clearance times|byelaws? for .*? range|public access to .*? range|"
    r"dates and times when .*? ranges? (?:are|is) being used for live firing)\b",
    # Routine evergreen public guidance, flight complaints, and administrative advisory pages
    r"\b(?:military\s+)?low[- ]flying\s+(?:military\s+)?(?:aircraft|training|operations?|"
    r"exercises?|timetables?|schedules?)\b",
    r"\bguidance(?:\s*:)?\s*(?:military\s+)?low[- ]flying\b",
    r"\b(?:military\s+)?low[- ]flying(?:\s+system)?\s*(?::|\b).*?"
    r"(?:exercises?|timetables?|schedules?)\b",
    r"\bmod[- ]sponsored\s+(?:air\s+)?exercises?\b",
    r"\boperational\s+low\s+flying\s+training\s+timetable\b",
    # Routine administrative guidance, standards, manuals, safety frameworks, and pensions
    r"\bguidance(?:\s*:)?\s*(?:defence\s+)?safety\s+management\s+system\b",
    r"\bdefence\s+safety\s+management\s+system(?:\s*\(jsp\s*\d+\))?\b",
    r"\b(?:joint\s+service\s+publication|jsp)\s+\d+\b",
    r"\b(?:defence\s+standard|def\s+stan)\s+\d+\b",
    r"\barmed\s+forces\s+pensions?(?:\s+schemes?)?\b",
    r"\bpensions\s+and\s+compensation\s+for\s+veterans\b",
    r"\b(?:complaints?|enquiries|questions and answers|faqs?|faq)\s+"
    r"(?:about|on|regarding)?\s*(?:military\s+)?low[- ]flying\b",
    r"\b(?:military\s+)?low[- ]flying(?:\s*:\s*|\s+)"
    r"(?:questions and answers|complaints|enquiries|faqs?|faq|guidance|advisory|public notice)\b",
    r"\bpublic\s+(?:information|guidance|advisory)\s+on\s+(?:military\s+)?low[- ]flying\b",
    r"\b(?:frequently asked questions|faqs?|complaints procedure|how to complain)\s+"
    r"(?:about|on|regarding)\s+(?:military|defence|defense|low flying)\b",
    # Seasonal/holiday-themed discussion roundtables, costume pieces, and reading lists
    r"\bhalloween\b",
    r"\bspooky\s+season\b",
    r"\bholiday\s+reading\s+list\b",
    r"\bholiday\s+gift\s+guide\b",
    r"\bholiday\s+special\s+edition\b",
    r"\bcostume\s+party\b",
    r"\btrick[- ]or[- ]treat\b",
    # Generic newsletter digests, podcast roundtables, and weekly reading roundups
    r"\b(?:weekly\s+roundup|daily\s+debrief\s+newsletter|weekend\s+reading\s+list|"
    r"in\s+case\s+you\s+missed\s+it:\s*our\s+top\s+stories)\b",
    r"\broundtable\s+discussion:\s*(?:halloween|holiday|spooky)\b",
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
    # Broad opinion pieces, book reviews, and author promotional retrospectives
    # without a timely defence hook
    r"\bhow semiconductors became the world['’]?s most powerful resource\b",
    r"\b(?:interview is with\s+[\w\s'’-]+?,\s+the\s+author of)\b",
    # Routine contractor personnel & executive appointments (no procurement value)
    r"\b(?:announces?|names?|appoints?)\s+[\w\s'’-]+?\s+"
    r"(?:as\s+(?:the\s+)?(?:new\s+)?(?:president|ceo|cfo|coo|chief executive|"
    r"managing director|board member))\b",
    r"\b(?:names?|appoints?|hires?)\s+(?:new\s+)?"
    r"(?:president|ceo|cfo|coo|chief executive|managing director)\b",
    r"\bexecutive leadership (?:changes?|appointments?|transition)\b",
)

HISTORICAL_CONFLICT_KEYWORDS: tuple[str, ...] = (
    # Named historical conflicts, battles, and campaigns
    r"\b(?:battle of\s+[\w\s'-]+?|bail[eé]n|waterloo|trafalgar|gettysburg|antietam|"
    r"verdun|somme|passchendaele|gallipoli|stalingrad|normandy landing|"
    r"crimean war|boer war|peloponnesian war|punic war|crusades|ancient warfare|"
    r"roman (?:legion|army|empire)|medieval (?:warfare|knights|siege)|"
    r"napoleon(?:ic)?(?:\s+wars?|\s+era)?)\b",
    r"\b(?:historical case study|historical perspective on the battle|"
    r"lessons from the battle of|historical military strategy of)\b",
    r"\b(?:in\s+(?:1[4-9]\d\d|1[0-3]\d\d)|"
    r"on the morning of [a-z]+ \d{1,2},\s*(?:1[4-9]\d\d|1[0-3]\d\d))\b",
)

HISTORICAL_CONTEMPORARY_HOOK_KEYWORDS: tuple[str, ...] = (
    r"\b(?:dna (?:testing|analysis|match|identification)|"
    r"remains? (?:identified|laid to rest|unearthed|discovered|recovered|found)|"
    r"laid to rest|unearthed|archaeolog\w*|ceremony|memorial service|commemorat\w*|"
    r"newly declassified|unsealed (?:records|files)|inquest|inquiry (?:finds|reveals|concludes)|"
    r"new evidence uncovered|posthumous|apology|pardon|museum exhibit)\b",
)

_DEFENCE_PATTERN = re.compile("|".join(DEFENCE_KEYWORDS), re.IGNORECASE)
_IRRELEVANT_PATTERN = re.compile("|".join(IRRELEVANT_KEYWORDS), re.IGNORECASE)
_HISTORICAL_CONFLICT_PATTERN = re.compile("|".join(HISTORICAL_CONFLICT_KEYWORDS), re.IGNORECASE)
_HISTORICAL_HOOK_PATTERN = re.compile(
    "|".join(HISTORICAL_CONTEMPORARY_HOOK_KEYWORDS), re.IGNORECASE
)
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
_MILITARY_AIRCRAFT_OVERRIDE_PATTERN = re.compile(
    r"\b((?:raf|british|uk|eurofighter)\s+typhoon|eurofighter(?:\s+typhoon)?|"
    r"typhoon\s+(?:fighters?|jets?|aircraft|fleet|squadron))\b",
    re.IGNORECASE,
)
_PROCUREMENT_OVERRIDE_PATTERN = re.compile(
    r"\b(procurement|contract|arms sale|missile deal|weapons? contract|awarded .* contract)\b",
    re.IGNORECASE,
)
_OPERATIONAL_OVERRIDE_PATTERN = re.compile(
    r"\b(multinational exercise|joint exercise|nato exercise|carrier deployment|"
    r"troops deploy|scrambled jets|naval patrol|air patrol|border reinforcement)\b",
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

    # Insufficient text check: if title or combined text is essentially empty
    if not title.strip() or len(combined_text.strip()) < 10:
        return "pending"

    # Pure historical military retrospectives / case studies without a contemporary news hook
    # are excluded from timely daily intelligence briefings
    if _HISTORICAL_CONFLICT_PATTERN.search(combined_text):
        if not _HISTORICAL_HOOK_PATTERN.search(combined_text):
            return "irrelevant"

    # Check for non-geopolitical / irrelevant domains first (even for dedicated defence outlets)
    if _IRRELEVANT_PATTERN.search(combined_text):
        # Allow kinetic attacks, high-consequence legal cases, military aircraft operations,
        # procurement contracts, or operational exercises to override incidental hits,
        # but never override genuine sports, celebrity, or lifestyle reporting.
        has_kinetic = bool(_KINETIC_OVERRIDE_PATTERN.search(combined_text))
        has_strategic_legal = bool(_STRATEGIC_LEGAL_OVERRIDE_PATTERN.search(combined_text))
        has_military_aircraft = bool(_MILITARY_AIRCRAFT_OVERRIDE_PATTERN.search(combined_text))
        has_procurement = bool(_PROCUREMENT_OVERRIDE_PATTERN.search(combined_text))
        has_operational = bool(_OPERATIONAL_OVERRIDE_PATTERN.search(combined_text))
        is_sports = bool(_SPORTS_OR_LIFESTYLE_PATTERN.search(combined_text))
        if (
            has_kinetic
            or has_strategic_legal
            or has_military_aircraft
            or has_procurement
            or has_operational
        ) and not is_sports:
            is_defence = clean_source in DEDICATED_DEFENCE_SOURCES
            if is_defence or bool(_DEFENCE_PATTERN.search(combined_text)):
                return "relevant"
            return "pending"
        return "irrelevant"

    # Dedicated defence and military outlets are relevant by default for military coverage
    if clean_source in DEDICATED_DEFENCE_SOURCES:
        return "relevant"

    # Check for direct geopolitical and defence keywords
    if _DEFENCE_PATTERN.search(combined_text) or bool(
        _MILITARY_AIRCRAFT_OVERRIDE_PATTERN.search(combined_text)
    ):
        return "relevant"

    # General news without explicit keyword matches remains pending for later analysis
    return "pending"
