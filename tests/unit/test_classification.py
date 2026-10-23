"""Unit tests for Phase 3 Part A: Geopolitical topic and relevance classification."""

from geopulse.classification import (
    BaseClassifier,
    ClassificationResult,
    RuleBasedClassifier,
    TopicCategory,
    classify_text,
)
from geopulse.ingestion.models import ArticleRecord


def test_clear_relevant_armed_conflict():
    res = classify_text(
        title="Russian forces launch overnight missile strike and artillery barrage on Kharkiv",
        snippet=(
            "Multiple explosions were reported in residential districts with casualties confirmed."
        ),
        source_name="BBC World News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.ARMED_CONFLICT
    assert res.confidence >= 0.65


def test_clear_relevant_weapons_procurement():
    res = classify_text(
        title="Poland signs $4bn procurement contract for Patriot missile batteries",
        snippet="The arms deal expands NATO air defence capability along the eastern flank.",
        source_name="Breaking Defense",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.WEAPONS_PROCUREMENT
    assert res.confidence >= 0.70


def test_clear_relevant_diplomacy():
    res = classify_text(
        title="Diplomatic envoys convene in Geneva for ceasefire negotiations",
        snippet="Bilateral talks aim to reach a durable peace accord between belligerents.",
        source_name="Al Jazeera English",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.DIPLOMACY
    assert res.confidence >= 0.65


def test_clear_relevant_sanctions():
    res = classify_text(
        title="European Union imposes new economic sanctions and asset freeze",
        snippet="The package includes export controls on dual-use technology and an oil price cap.",
        source_name="BBC World News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.SANCTIONS
    assert res.confidence >= 0.65


def test_clear_relevant_cyber_hybrid_warfare():
    res = classify_text(
        title="State-sponsored hackers launch coordinated cyberattack against power grid",
        snippet="Cyber warfare specialists detected malware disrupting critical infrastructure.",
        source_name="Defense News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.CYBER_HYBRID_WARFARE
    assert res.confidence >= 0.65


def test_clear_relevant_security_incident():
    res = classify_text(
        title="Commercial oil tanker hit in maritime intercept near Strait of Hormuz",
        snippet="A cargo vessel was boarded by armed commandos in international waters.",
        source_name="BBC World News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.SECURITY_INCIDENT
    assert res.confidence >= 0.65


def test_clear_irrelevant_sports():
    res = classify_text(
        title="Premier League: Arsenal defeats Manchester City 2-1 in thrilling football derby",
        snippet="The striker scored in stoppage time as title race tightens.",
        source_name="BBC World News",
    )
    assert res.relevance_status == "irrelevant"
    assert res.primary_topic == TopicCategory.OTHER_GEOPOLITICAL
    assert res.confidence >= 0.80


def test_clear_irrelevant_entertainment():
    res = classify_text(
        title="Hollywood box office breaks summer record as celebrities attend film festival",
        snippet="The red carpet event featured Oscar-winning actors and directors.",
        source_name="BBC World News",
    )
    assert res.relevance_status == "irrelevant"
    assert res.primary_topic == TopicCategory.OTHER_GEOPOLITICAL
    assert res.confidence >= 0.80


def test_ambiguous_case_remains_pending():
    res = classify_text(
        title="Mayor announces new municipal budget and infrastructure plan",
        snippet="Local transit routes will receive additional funding next year.",
        source_name="BBC World News",
    )
    # General news without defence/geopolitical signals should remain pending, not discarded
    assert res.relevance_status == "pending"
    assert res.primary_topic == TopicCategory.UNCERTAIN
    assert res.confidence <= 0.40


def test_conflicting_keywords_remain_pending():
    res = classify_text(
        title="Premier League match postponed after drone strike near stadium",
        snippet=(
            "Football authorities confirmed safety precautions following missile strike alerts."
        ),
        source_name="BBC World News",
    )
    # Contains both sports signal ("Premier League") and incident signal ("drone strike")
    # Public civilian event disrupted by an isolated incident maps to SECURITY_INCIDENT
    assert res.relevance_status == "pending"
    assert res.primary_topic == TopicCategory.SECURITY_INCIDENT
    assert "Conflicting signals" in (res.explanation or "")


def test_sports_event_disrupted_by_security_incident_classified_as_security_incident():
    res = classify_text(
        title="Premier League match postponed after drone strike near stadium",
        snippet=(
            "The football clash between Arsenal and Chelsea was delayed following "
            "an isolated security incident in the city centre."
        ),
        source_name="BBC World News",
    )
    # Isolated security incident at a sporting event must be classified as SECURITY_INCIDENT
    assert res.relevance_status == "pending"
    assert res.primary_topic == TopicCategory.SECURITY_INCIDENT
    assert res.confidence == 0.45


def test_unrelated_sports_coverage_with_metaphorical_terms_remains_irrelevant():
    res = classify_text(
        title="Premier League: Arsenal battles Chelsea in midfield dogfight",
        snippet="The striker fired a rocket shot into the net to claim a dramatic derby victory.",
        source_name="BBC World News",
    )
    # Metaphorical language in pure sports coverage must remain strictly irrelevant
    assert res.relevance_status == "irrelevant"
    assert res.primary_topic == TopicCategory.OTHER_GEOPOLITICAL


def test_missing_description_handles_cleanly():
    res = classify_text(
        title="US approves $2B arms deal for Patriot missiles",
        snippet=None,
        source_name="Defense News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.WEAPONS_PROCUREMENT
    assert res.confidence >= 0.60


def test_empty_or_too_short_text():
    res = classify_text(
        title="",
        snippet="",
        source_name=None,
    )
    assert res.relevance_status == "pending"
    assert res.primary_topic == TopicCategory.UNCERTAIN
    assert res.confidence <= 0.15


def test_non_english_unusual_text_retained_as_pending():
    res = classify_text(
        title="Военные учения начались на полигоне в Сибири",
        snippet="Подразделения отрабатывают нормативы боевой готовности.",
        source_name="General International Outlet",
    )
    # Non-English text without known English keywords must not be discarded
    assert res.relevance_status == "pending"
    assert res.primary_topic == TopicCategory.UNCERTAIN


def test_dedicated_defence_outlet_default_relevance():
    res = classify_text(
        title="Strategic modernization: assessing future force readiness",
        snippet=(
            "A review of personnel rotation and technological changes across military commands."
        ),
        source_name="War on the Rocks",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.DEFENCE_POLICY
    assert res.confidence >= 0.65


def test_relevance_distinct_from_armed_conflict():
    # Diplomacy is relevant but not armed conflict
    res_diplomacy = classify_text(
        title="Presidential summit concludes with historic peace accord and armistice negotiations",
        snippet="Both sides pledged to withdraw heavy equipment from the border zone.",
        source_name="BBC World News",
    )
    assert res_diplomacy.relevance_status == "relevant"
    assert res_diplomacy.primary_topic == TopicCategory.DIPLOMACY
    assert res_diplomacy.primary_topic != TopicCategory.ARMED_CONFLICT

    # Procurement is relevant but not armed conflict
    res_procurement = classify_text(
        title="Navy issues submarine deal tender and warship contract",
        snippet="Domestic shipyards compete for multi-billion naval expansion.",
        source_name="UK Ministry of Defence",
    )
    assert res_procurement.relevance_status == "relevant"
    assert res_procurement.primary_topic == TopicCategory.WEAPONS_PROCUREMENT
    assert res_procurement.primary_topic != TopicCategory.ARMED_CONFLICT


def test_classify_article_record():
    classifier = RuleBasedClassifier()
    art = ArticleRecord(
        source_name="Breaking Defense",
        title="Air Force tests new hypersonic missile in Pacific proving grounds",
        url="https://breakingdefense.com/test-hypersonic",
        canonical_url="https://breakingdefense.com/test-hypersonic",
        content_hash="test-hash-123",
        snippet="Flight telemetry demonstrated sustained Mach 5 speeds.",
    )
    res = classifier.classify_article(art)
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.WEAPONS_PROCUREMENT


def test_classifier_extensibility_interface():
    class MockOllamaClassifier(BaseClassifier):
        def classify_text(
            self,
            title: str,
            snippet: str | None = None,
            source_name: str | None = None,
        ) -> ClassificationResult:
            return ClassificationResult(
                relevance_status="relevant",
                primary_topic=TopicCategory.ARMED_CONFLICT,
                confidence=0.99,
                explanation="Mocked Ollama local model classification",
            )

    classifier = MockOllamaClassifier()
    res = classifier.classify_text("Sample Title")
    assert res.primary_topic == TopicCategory.ARMED_CONFLICT
    assert res.confidence == 0.99


def test_semiconductor_distinctions_defence_policy_and_sanctions():
    # Broad semiconductor rivalry and strategic supply chain -> DEFENCE_POLICY
    res_policy = classify_text(
        title="The Endless Chip War: Semiconductor Competition and National Security",
        snippet=(
            "Analysis of critical technology competition, semiconductor supply chain security, "
            "and technological sovereignty."
        ),
        source_name="War on the Rocks",
    )
    assert res_policy.relevance_status == "relevant"
    assert res_policy.primary_topic == TopicCategory.DEFENCE_POLICY

    # Test with the exact actual RSS headline and snippet from War on the Rocks
    actual_chip_snippet = (
        "This exclusive Cogs of War interview is with Rakesh Kumar, the author of Chip Age: "
        "How Semiconductors Became the World’s Most Powerful Resource. Rakesh has worked "
        "in the chip industry for more than two decades."
    )
    res_actual_chip = classify_text(
        title="The Endless Chip War",
        snippet=actual_chip_snippet,
        source_name="War on the Rocks",
    )
    assert res_actual_chip.relevance_status == "relevant"
    assert res_actual_chip.primary_topic == TopicCategory.DEFENCE_POLICY

    # Dedicated export controls and chip sanctions -> SANCTIONS
    res_sanctions = classify_text(
        title="Commerce Department expands semiconductor export controls and chip sanctions",
        snippet="New restrictions prohibit advanced lithography sales and target foundry networks.",
        source_name="Defense News",
    )
    assert res_sanctions.relevance_status == "relevant"
    assert res_sanctions.primary_topic == TopicCategory.SANCTIONS

    # Network intrusions, hacking, and electronic warfare -> CYBER_HYBRID_WARFARE
    res_cyber = classify_text(
        title="State-sponsored hackers launch coordinated cyberattack against power grid",
        snippet="Cyber warfare operations disrupted communications and network operations.",
        source_name="BBC World News",
    )
    assert res_cyber.relevance_status == "relevant"
    assert res_cyber.primary_topic == TopicCategory.CYBER_HYBRID_WARFARE


def test_capability_development_vs_procurement_accuracy():
    # Long-range capability development / warfare domain announcement -> DEFENCE_POLICY
    res_cislunar = classify_text(
        title="Space Force shifts into high gear on cislunar warfare, with RFIs coming soon",
        snippet=(
            "Maj. Gen. Stephen Purdy said the service wants to field cislunar capabilities "
            "by the end of fiscal 2029."
        ),
        source_name="Breaking Defense",
    )
    assert res_cislunar.relevance_status == "relevant"
    assert res_cislunar.primary_topic == TopicCategory.DEFENCE_POLICY

    # Direct military contract award to an aerospace contractor -> WEAPONS_PROCUREMENT
    res_contract = classify_text(
        title="Natilus secures Air Force contract to develop blended wing body aircraft",
        snippet="Natilus has secured a contract from the US Air Force to develop aircraft.",
        source_name="Breaking Defense",
    )
    assert res_contract.relevance_status == "relevant"
    assert res_contract.primary_topic == TopicCategory.WEAPONS_PROCUREMENT




def test_procurement_and_diplomacy_topics_accurately_classified():
    res_proc = classify_text(
        title="Denmark orders long-range missiles for new strike brigade",
        snippet="Defence Ministry signs contract for precision munitions delivery.",
        source_name="Breaking Defense",
    )
    assert res_proc.relevance_status == "relevant"
    assert res_proc.primary_topic == TopicCategory.WEAPONS_PROCUREMENT

    res_diplo = classify_text(
        title="Diplomatic envoys negotiate prisoner release and border security talks",
        snippet="Representatives meet in neutral territory to agree terms for veteran release.",
        source_name="BBC World News",
    )
    assert res_diplo.relevance_status == "relevant"
    assert res_diplo.primary_topic == TopicCategory.DIPLOMACY


def test_darc_programme_vs_radar_contract_distinction():
    # Multinational capability programme / site redevelopment proposal -> DEFENCE_POLICY
    res_darc = classify_text(
        title="Deep Space Advanced Radar Capability (DARC)",
        snippet=(
            "The Ministry of Defence (MOD) is proposing to redevelop Cawdor Barracks as the UK "
            "host site for the DARC radar programme."
        ),
        source_name="UK Ministry of Defence",
    )
    assert res_darc.relevance_status == "relevant"
    assert res_darc.primary_topic == TopicCategory.DEFENCE_POLICY

    # Signed radar equipment contract -> WEAPONS_PROCUREMENT
    res_radar_contract = classify_text(
        title="UK awards £100m radar systems contract for air surveillance",
        snippet="Defence Equipment & Support signs radar procurement deal with contractor.",
        source_name="Defense News",
    )
    assert res_radar_contract.relevance_status == "relevant"
    assert res_radar_contract.primary_topic == TopicCategory.WEAPONS_PROCUREMENT


def test_retrospective_war_overview_classified_as_irrelevant_for_daily_digest():
    res = classify_text(
        title="US-Iran war, then and now: How the landscape has changed since February",
        snippet=(
            "The positions of Washington and Tehran have changed dramatically since US-Israeli "
            "strikes began the war, analysts say."
        ),
        source_name="Al Jazeera English",
    )
    assert res.relevance_status == "irrelevant"
    assert res.primary_topic == TopicCategory.OTHER_GEOPOLITICAL


def test_lethal_security_incident_with_election_timing_remains_relevant():
    res = classify_text(
        title="Two Israeli soldiers killed in attack by suspected Palestinian gunmen in West Bank",
        snippet=(
            "The two attackers were shot dead after a gun battle outside Hebron, four days before "
            "Israel's general election."
        ),
        source_name="BBC World News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.SECURITY_INCIDENT


def test_naval_carrier_voyages_classified_as_military_operations():
    res = classify_text(
        title=(
            "China’s first aircraft carrier forms ‘evolving pattern’ of voyages in "
            "disputed Asian seas"
        ),
        snippet=(
            "The carrier group’s passages not far from Taiwan, Japan and the Philippines have "
            "raised alerts."
        ),
        source_name="Defense News",
    )
    assert res.relevance_status == "relevant"
    assert res.primary_topic == TopicCategory.MILITARY_OPERATIONS


