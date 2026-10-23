"""Unit tests for keyword and source-based relevance classification."""

from geopulse.ingestion.relevance import classify_relevance


def test_dedicated_defence_sources_are_always_relevant():
    assert classify_relevance("Navy Procurement Update", source_name="Defense News") == "relevant"
    assert classify_relevance("Budget review", source_name="Breaking Defense") == "relevant"
    assert classify_relevance("Strategic doctrine", source_name="War on the Rocks") == "relevant"
    assert (
        classify_relevance("Operations briefing", source_name="UK Ministry of Defence")
        == "relevant"
    )


def test_general_sources_match_military_keywords():
    assert (
        classify_relevance("Airstrikes reported near border post", source_name="BBC World News")
        == "relevant"
    )
    assert (
        classify_relevance("Missile barrage intercepts drones", source_name="Al Jazeera English")
        == "relevant"
    )
    assert (
        classify_relevance("Diplomats seek ceasefire in conflict", source_name="BBC World News")
        == "relevant"
    )


def test_general_sources_match_irrelevant_keywords():
    assert (
        classify_relevance("Premier League match recap and goals", source_name="BBC World News")
        == "irrelevant"
    )
    assert (
        classify_relevance("Hollywood box office hits new record", source_name="Al Jazeera English")
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "New electronics reach local markets, prices soar as gov't fees rise",
            source_name="Iran International",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Students unseat veterans in heated parliamentary election",
            source_name="Balkan Insight",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Police open investigation into financial affairs of regional company",
            source_name="Cyprus Mail",
        )
        == "irrelevant"
    )


def test_dedicated_defence_sources_with_irrelevant_topics_filtered():
    """Irrelevant content in dedicated defence sources must be filtered out."""
    assert (
        classify_relevance(
            "New clinic offers equine therapy for treating post-traumatic stress",
            source_name="Defense News",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Air Force assists local teams to fight wildfires across state",
            source_name="Breaking Defense",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Agency issues Notice of Proposed Amendments for administrative review",
            source_name="Defense News",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Commander resigns over inappropriate relationship",
            source_name="Breaking Defense",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Winning the information war through ideas and perception",
            snippet="[Sponsored] As social media and AI amplify disinformation campaigns...",
            source_name="Breaking Defense",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Deception in Mass Autonomous Warfare",
            snippet="Historical case study and theoretical monograph on military deception...",
            source_name="War on the Rocks",
        )
        == "irrelevant"
    )


def test_timely_strategic_analysis_remains_relevant():
    assert (
        classify_relevance(
            "The Endless Chip War",
            snippet="Exclusive Cogs of War interview on semiconductor competition...",
            source_name="War on the Rocks",
        )
        == "relevant"
    )



def test_ambiguous_topics_remain_pending():
    assert (
        classify_relevance("Central bank announces new interest rate", source_name="BBC World News")
        == "pending"
    )


def test_retrospective_overview_without_timely_hook_is_irrelevant():
    assert (
        classify_relevance(
            "US-Iran war, then and now: How the landscape has changed since February",
            snippet="The positions of Washington and Tehran have changed dramatically...",
            source_name="Al Jazeera English",
        )
        == "irrelevant"
    )


def test_kinetic_casualty_incident_with_election_timing_remains_relevant():
    assert (
        classify_relevance(
            "Two Israeli soldiers killed in attack by suspected Palestinian gunmen in West Bank",
            snippet=(
                "The two attackers were shot dead after a gun battle outside Hebron, "
                "four days before Israel's general election."
            ),
            source_name="BBC World News",
        )
        == "relevant"
    )


def test_sports_doping_and_athletics_filtered_as_irrelevant():
    assert (
        classify_relevance(
            "CAS clears Japan’s Olympic silver medallist Ikeda of doping, overturns ban",
            snippet="Race walker returns to competition immediately after appeal.",
            source_name="Al Jazeera English",
        )
        == "irrelevant"
    )


def test_routine_administrative_court_listings_filtered_as_irrelevant():
    """Routine court dockets and sentencing remarks must not clutter the intelligence briefing."""
    assert (
        classify_relevance(
            "Transparency data: Military Court Service sentencing remarks 2026",
            snippet="Transcripts of sentencing remarks from Service Court hearings.",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Transparency data: Military court centres court listings",
            snippet="Published court listings for the military court centres.",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "UK Ministry of Defence Publishes Court Listings for Military Court Centres",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "UK Ministry of Defence: Sentencing Remarks in Court Martial",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )


def test_consequential_military_justice_and_reforms_preserved_as_relevant():
    """Strategic legal cases (treason, espionage, war crimes, reforms) must NOT be filtered out."""
    assert (
        classify_relevance(
            "Court-martial sentences soldier for treason and leaking classified submarine secrets",
            snippet="Serviceman convicted of passing defence data to foreign adversary.",
            source_name="Defense News",
        )
        == "relevant"
    )
    assert (
        classify_relevance(
            "UK announces sweeping reform of military court service and court-martial procedures",
            snippet="Defence Secretary overhauls legal structure across armed forces.",
            source_name="UK Ministry of Defence",
        )
        == "relevant"
    )
    assert (
        classify_relevance(
            "Commander faces court-martial trial over war crimes in village shelling",
            snippet="Officer charged with murder and unlawful combat directives.",
            source_name="BBC World News",
        )
        == "relevant"
    )
    assert (
        classify_relevance(
            "Officer faces military court hearing over espionage and bribery scandal",
            source_name="Breaking Defense",
        )
        == "relevant"
    )

