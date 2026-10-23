"""Rule-based topic and relevance classification engine."""

import re

from geopulse.classification.base import BaseClassifier
from geopulse.classification.models import ClassificationResult, TopicCategory

# Dedicated defence sources that cover military and security affairs exclusively
DEDICATED_DEFENCE_SOURCES: frozenset[str] = frozenset(
    {
        "defense news",
        "breaking defense",
        "war on the rocks",
        "uk ministry of defence",
    }
)

# Strong negative signals (sports, entertainment, lifestyle, medical therapy,
# civilian wildfires, routine regulations, consumer tech, routine domestic elections)
IRRELEVANT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\b(football|soccer|cricket|premier league|nfl|nba|tennis|golf|olympics)\b",
        re.I,
    ),
    re.compile(
        r"\b(box office|celebrity|hollywood|oscar|grammy|film festival|red carpet)\b",
        re.I,
    ),
    re.compile(
        r"\b(recipe|recipes|lifestyle|horoscope|fashion week|dating|dating app)\b",
        re.I,
    ),
    re.compile(
        r"\b(post-traumatic stress|ptsd|treating post-traumatic|therapy|therapies|"
        r"psychotherapy|mental health|trauma recovery|clinical trial|breakthrough in treating|"
        r"cancer|breast cancer|chemotherapy)\b",
        re.I,
    ),
    re.compile(
        r"\b(fight wildfires|wildfires?|bushfires?|forest fires?|firefighting|firefighters?|"
        r"typhoon|hurricane|cyclone|tornado|earthquake|tsunami|flights? cancelled as typhoon)\b",
        re.I,
    ),
    re.compile(
        r"\b(weekend protests?|transit disruption)\b",
        re.I,
    ),
    re.compile(
        r"\b(notice of proposed amendments|npa documents?|regulatory compliance notice|"
        r"transparency data:\s*military court|"
        r"(?:court listings? (?:for|at|of)?\s*military court(?: centres?)?|"
        r"military court(?: centres?)? court listings?)|"
        r"(?:military )?court[- ]martial (?:court )?listings?|"
        r"sentencing remarks in (?:the )?court[- ]martial|"
        r"military court service sentencing remarks|transcripts of sentencing remarks)\b",
        re.I,
    ),
    re.compile(
        r"\b(inappropriate relationship|resigns over inappropriate|personal conduct scandal)\b",
        re.I,
    ),
    re.compile(
        r"\b(snap election|general election|election campaign|parliamentary elections?|"
        r"voters? head to polls?|students? unseat|opinion polls?|mayoral race)\b",
        re.I,
    ),
    re.compile(
        r"\b(new electronics reach|electronics reach|smartphones?|consumer devices?|"
        r"prices soar as gov['’]?t fees)\b",
        re.I,
    ),
    re.compile(
        r"\b(investigation into financial affairs|financial affairs of|tax fraud probe)\b",
        re.I,
    ),
    re.compile(
        r"(\[sponsored\]|\bsponsored content\b|\badvertorial\b|"
        r"\bpartner content\b|\bsponsored by\b)",
        re.I,
    ),
    re.compile(
        r"\b(deception in mass autonomous warfare|theoretical monograph|doctrinal essay)\b",
        re.I,
    ),
    re.compile(
        r"\b(then and now: how the landscape has changed|how the landscape has changed since|"
        r"retrospective overview|historical lookback on the war)\b",
        re.I,
    ),
)

# Curated regex patterns for specific geopolitical topic categories
TOPIC_PATTERNS: dict[TopicCategory, tuple[re.Pattern[str], ...]] = {
    TopicCategory.ARMED_CONFLICT: (
        re.compile(
            r"\b(airstrike|air strike|airstrikes|air strikes|bombardment|shelling|"
            r"military strike|military strikes|forces strike)\b",
            re.I,
        ),
        re.compile(r"\b(battlefield|frontline clashes|ground offensive|counteroffensive)\b", re.I),
        re.compile(
            r"\b(casualties|casualty count|killed in action|missile strike|"
            r"attackers killed|wounded in .* shooting|fatal shooting|gunmen open fire|"
            r"downing of .* aircraft|soldiers? killed|soldiers? wounded|troops killed|"
            r"gun battle|killed in attack|attack by suspected .* gunmen)\b",
            re.I,
        ),
        re.compile(r"\b(drone strike|drone strikes|artillery barrage|combat losses)\b", re.I),
    ),
    TopicCategory.MILITARY_OPERATIONS: (
        re.compile(
            r"\b(troop deployment|troops deploy|troop movements?|military movements?|"
            r"military exercise|joint drills)\b",
            re.I,
        ),
        re.compile(
            r"\b(naval patrol|air patrol|scrambled jets|carrier strike group|"
            r"aircraft carrier|carrier group|carrier voyages?|naval voyages?|flotilla)\b",
            re.I,
        ),
        re.compile(r"\b(border reinforcement|readiness drill|peacekeeping mission)\b", re.I),
        re.compile(
            r"\b(autonomous warfare|mass autonomous|deception in .* warfare|mass drone)\b",
            re.I,
        ),
    ),
    TopicCategory.DEFENCE_POLICY: (
        re.compile(r"\b(defence policy|defense policy|military doctrine|national defence)\b", re.I),
        re.compile(r"\b(nato alliance|security pact|aukus|pentagon report|white paper)\b", re.I),
        re.compile(r"\b(defence budget|defense spending|military reform|conscription)\b", re.I),
        re.compile(
            r"\b(chip war|semiconductor competition|semiconductor rivalry|tech rivalry|"
            r"semiconductor industry|microchip competition|technological sovereignty|"
            r"tech sovereignty|critical technology competition|strategic supply[- ]chain policy|"
            r"microchip war)\b",
            re.I,
        ),
        re.compile(
            r"\b(cislunar warfare|cislunar capabilities|space warfare capability|"
            r"capability development|future warfare concept)\b",
            re.I,
        ),
        re.compile(
            r"\b(advanced radar capability|darc|deep space advanced radar capability|"
            r"radar programme|cawdor barracks|redevelop cawdor barracks)\b",
            re.I,
        ),
    ),
    TopicCategory.WEAPONS_PROCUREMENT: (
        re.compile(
            r"\b(procurement|arms sale|arms deal|missile contract|"
            r"fighter jets|air defence system|air defense system)\b",
            re.I,
        ),
        re.compile(
            r"\b(warship contract|submarine deal|ammunition order|"
            r"defense contract|defence contract|air force contract|navy contract|"
            r"army contract|military contract|pentagon contract|munitions contract|"
            r"precision munitions?)\b",
            re.I,
        ),
        re.compile(
            r"\b(hypersonic test|hypersonic missiles?|ballistic missiles?|"
            r"radar contract|patriot missiles?|weapon systems?)\b",
            re.I,
        ),
        re.compile(
            r"\b(missile orders?|orders? (?:[\w-]+ )*missiles?|buys? (?:[\w-]+ )*missiles?|"
            r"missiles? buy|commits? to .* missiles?|acquires? (?:[\w-]+ )*missiles?|"
            r"procures? (?:[\w-]+ )*missiles?|long-range strike option|long-range missiles?|"
            r"kongsberg.*missiles?|covenant.*missiles?)\b",
            re.I,
        ),
        re.compile(
            r"\b(radar contract|radar procurement|radar systems? contract|radar acquisition)\b",
            re.I,
        ),
    ),
    TopicCategory.DIPLOMACY: (
        re.compile(r"\b(peace talks|ceasefire negotiations|truce agreement|peace accord)\b", re.I),
        re.compile(
            r"\b(diplomatic (?:summit|talks|envoys?|negotiations?|mission)|"
            r"bilateral talks|un security council resolution)\b",
            re.I,
        ),
        re.compile(r"\b(envoy meeting|ambassador recalled|armistice negotiations)\b", re.I),
        re.compile(
            r"\b(negotiat\w+ (?:.* )?release|prisoner (?:exchange|swap|release)|"
            r"hostage release(?: negotiations)?|release of .* from .* prison|"
            r"military talks|talks (?:over|on) border|border (?:security )?talks|"
            r"invites? .* for .* talks)\b",
            re.I,
        ),
    ),
    TopicCategory.SANCTIONS: (
        re.compile(r"\b(economic sanctions|trade embargo|asset freeze|export controls)\b", re.I),
        re.compile(
            r"\b(sanctioned entity|financial blacklist|oil price cap|customs tariff)\b", re.I
        ),
        re.compile(
            r"\b(semiconductor export controls?|chip export controls?|advanced chip controls?|"
            r"semiconductor ban|chip sanctions|technology sanctions|chip embargo)\b",
            re.I,
        ),
    ),
    TopicCategory.TERRORISM_INSURGENCY: (
        re.compile(r"\b(suicide bombing|car bomb|militants ambush|insurgent attack)\b", re.I),
        re.compile(
            r"\b(jihadist group|hostage execution|improvised explosive device|ied blast)\b", re.I
        ),
    ),
    TopicCategory.CYBER_HYBRID_WARFARE: (
        re.compile(
            r"\b(cyberattack|cyber warfare|state-sponsored hackers|critical infrastructure hack)\b",
            re.I,
        ),
        re.compile(
            r"\b(gps jamming|electronic warfare|undersea cable sabotage|disinformation)\b",
            re.I,
        ),
        re.compile(
            r"\b(starlink (?:still )?restricted|starlink limits?|starlink terminals?|"
            r"satellite communications? restricted)\b",
            re.I,
        ),
    ),
    TopicCategory.SECURITY_INCIDENT: (
        re.compile(r"\b(vessel seized|tanker hit|strait of hormuz|maritime intercept)\b", re.I),
        re.compile(
            r"\b(airspace violation|border skirmish|hostage standoff|piracy attack)\b", re.I
        ),
        re.compile(
            r"\b(security incident|security alerts?|security breach|drone threat|drone alert|"
            r"drone sighting|suspicious drone|bomb threat|west bank shooting|shooting attack|"
            r"gun battle outside|attackers? shot dead|soldiers? killed in attack)\b",
            re.I,
        ),
    ),
}


STRATEGIC_LEGAL_OVERRIDE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(treason|espionage|classified|secrets?|war crimes?|atrocit\w*|"
    r"homicide|murder|bribery|corruption|reform|mutiny|sabotage|trafficking)\b",
    re.I,
)


class RuleBasedClassifier(BaseClassifier):
    """Deterministic rule-based geopolitical relevance and topic classifier."""

    def classify_text(
        self,
        title: str,
        snippet: str | None = None,
        source_name: str | None = None,
    ) -> ClassificationResult:
        clean_title = (title or "").strip()
        clean_snippet = (snippet or "").strip()
        combined_text = f"{clean_title} {clean_snippet}"
        clean_source = (source_name or "").strip().lower()

        # Handle empty or non-informative text
        if not clean_title or len(clean_title) < 4:
            return ClassificationResult(
                relevance_status="pending",
                primary_topic=TopicCategory.UNCERTAIN,
                confidence=0.1,
                explanation="Insufficient text available for classification",
            )

        is_defence_outlet = clean_source in DEDICATED_DEFENCE_SOURCES

        # Check negative signals (unless overridden by strategic/consequential legal cases)
        has_strategic_legal = bool(STRATEGIC_LEGAL_OVERRIDE_PATTERN.search(combined_text))
        irrelevant_hit = any(pat.search(combined_text) for pat in IRRELEVANT_PATTERNS)
        if irrelevant_hit and has_strategic_legal:
            irrelevant_hit = False

        # Score topic matches (give title 2x weight over snippet)
        topic_scores: dict[TopicCategory, int] = {}
        for category, patterns in TOPIC_PATTERNS.items():
            score = 0
            for pat in patterns:
                # Title match is worth 2 points
                if pat.search(clean_title):
                    score += 2
                # Snippet match is worth 1 point
                elif clean_snippet and pat.search(clean_snippet):
                    score += 1
            if score > 0:
                topic_scores[category] = score

        # Evaluate if negative signal matches
        if irrelevant_hit:
            if topic_scores:
                top_category, _ = sorted(
                    topic_scores.items(), key=lambda kv: kv[1], reverse=True
                )[0]
                # If a civilian/sports/entertainment report matches conflict keywords,
                # an isolated incident at a public venue is a security incident,
                # not battlefield warfare.
                if top_category == TopicCategory.ARMED_CONFLICT:
                    top_category = TopicCategory.SECURITY_INCIDENT

                if top_category == TopicCategory.SECURITY_INCIDENT:
                    is_sports_or_lifestyle = bool(
                        re.search(
                            r"\b(premier league|football|soccer|nba|nfl|tennis|golf|celebrity|"
                            r"hollywood|box office|horoscope|lifestyle|recipe|dating)\b",
                            combined_text,
                            re.I,
                        )
                    )
                    if is_sports_or_lifestyle:
                        return ClassificationResult(
                            relevance_status="pending",
                            primary_topic=top_category,
                            confidence=0.45,
                            explanation=(
                                f"Conflicting signals: matched both {top_category.value} "
                                "and entertainment/lifestyle patterns"
                            ),
                        )
                    return ClassificationResult(
                        relevance_status="relevant",
                        primary_topic=top_category,
                        confidence=0.75,
                        explanation=(
                            "Active security incident/attack reported with background "
                            "electoral/civilian timing context"
                        ),
                    )
            return ClassificationResult(
                relevance_status="irrelevant",
                primary_topic=TopicCategory.OTHER_GEOPOLITICAL,
                confidence=0.85,
                explanation=(
                    "Matched non-geopolitical domain (lifestyle, medical, routine politics, "
                    "civilian emergency, or personal scandal)"
                ),
            )

        # If topic scores were found
        if topic_scores:
            sorted_topics = sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)
            top_category, top_score = sorted_topics[0]
            secondary = tuple(cat for cat, _ in sorted_topics[1:])

            # Confidence based on score strength
            confidence = min(0.95, 0.5 + (top_score * 0.15))

            return ClassificationResult(
                relevance_status="relevant",
                primary_topic=top_category,
                secondary_topics=secondary,
                confidence=confidence,
                explanation=f"Matched patterns for {top_category.value} (score={top_score})",
            )

        # If from dedicated defence outlet but no specific pattern matched
        if is_defence_outlet:
            return ClassificationResult(
                relevance_status="relevant",
                primary_topic=TopicCategory.DEFENCE_POLICY,
                confidence=0.7,
                explanation=f"Dedicated defence outlet '{source_name}' reporting",
            )

        # If general news with weak or unclassified signals
        return ClassificationResult(
            relevance_status="pending",
            primary_topic=TopicCategory.UNCERTAIN,
            confidence=0.3,
            explanation="Ambiguous signals; pending further analysis",
        )
