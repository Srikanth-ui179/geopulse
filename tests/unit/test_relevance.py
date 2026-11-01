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
            "The Endless Chip War: Semiconductor Competition and National Security",
            snippet=(
                "Analysis of critical technology competition, semiconductor supply chain security, "
                "and technological sovereignty."
            ),
            source_name="War on the Rocks",
        )
        == "relevant"
    )


def test_broad_opinion_book_interview_is_irrelevant():
    """Broad author promotional interviews and commercial book discussions are excluded."""
    assert (
        classify_relevance(
            "The Endless Chip War",
            snippet=(
                "This exclusive Cogs of War interview is with Rakesh Kumar, the author of Chip "
                "Age: How Semiconductors Became the World’s Most Powerful Resource. Rakesh has "
                "worked in the chip industry for more than two decades."
            ),
            source_name="War on the Rocks",
        )
        == "irrelevant"
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


def test_military_aircraft_typhoon_not_filtered_as_weather_disaster():
    """RAF Typhoon fighters in military exercises must not be filtered out as weather disaster."""
    assert (
        classify_relevance(
            "UK-France deepen nuclear defence ties in first joint exercise",
            snippet=(
                "UK deepens nuclear defence ties with France in historic first joint air "
                "exercise as RAF Typhoon fighters join French Rafale jets in a French "
                "nuclear exercise."
            ),
            source_name="UK Ministry of Defence",
        )
        == "relevant"
    )
    assert (
        classify_relevance(
            "RAF Typhoon fighters scramble to intercept Russian bombers over North Sea",
            snippet="Quick Reaction Alert aircraft deployed from RAF Lossiemouth.",
            source_name="BBC World News",
        )
        == "relevant"
    )


def test_genuine_typhoon_weather_disaster_remains_filtered():
    """Genuine civilian typhoon and storm disasters must remain filtered as irrelevant."""
    assert (
        classify_relevance(
            "Typhoon Gaemi causes deadly landslides and widespread flooding across islands",
            snippet=(
                "Disaster response teams deploy emergency relief after super typhoon "
                "makes landfall."
            ),
            source_name="Al Jazeera English",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Flights cancelled as Typhoon approaches southern coastal provinces",
            snippet="Meteorological agency issues storm warning.",
            source_name="BBC World News",
        )
        == "irrelevant"
    )


def test_contractor_executive_appointments_filtered_as_irrelevant():
    """Routine corporate C-suite / HR appointments must be filtered as irrelevant."""
    assert (
        classify_relevance(
            "GDIT announces Amy Johnson as new president",
            snippet=(
                "Johnson replaces longtime president Amy Gilliland, who was let go from the "
                "company earlier this month after almost a decade of leading the business unit."
            ),
            source_name="Breaking Defense",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Lockheed Martin names new chief executive officer amid executive leadership changes",
            snippet="Board of directors announces leadership transition.",
            source_name="Defense News",
        )
        == "irrelevant"
    )


def test_substantive_contractor_defence_procurement_preserved_as_relevant():
    """Substantive contractor procurement contracts and arms deals must remain relevant."""
    assert (
        classify_relevance(
            "Pentagon awards GDIT $1.2B enterprise IT modernization contract",
            snippet=(
                "General Dynamics Information Technology wins cloud computing procurement deal."
            ),
            source_name="Breaking Defense",
        )
        == "relevant"
    )


def test_routine_administrative_hr_relationship_guidance_filtered_as_irrelevant():
    """Service personnel relationship admin guidance must be excluded from daily digest."""
    assert (
        classify_relevance(
            "Established Long-Term Relationship status for service personnel",
            snippet=(
                "The process for recording and approving a defence-recognised "
                "Established Long-Term Relationship (LTR(E))."
            ),
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )


def test_routine_environmental_estate_ash_dieback_filtered_as_irrelevant():
    """Environmental estate disease notices must be excluded from intelligence briefing."""
    assert (
        classify_relevance(
            "Guidance: Ash dieback on the Ministry of Defence Estate",
            snippet=(
                "Ash dieback affects ash trees across the country and has infected many woodlands "
                "across the Ministry of Defence (MOD) training estate."
            ),
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )


def test_routine_civilian_range_firing_times_filtered_as_irrelevant():
    """Civilian range clearance times notices must be excluded from intelligence briefing."""
    assert (
        classify_relevance(
            "Guidance: Otterburn firing times",
            snippet=(
                "Dates and times when Otterburn and Redesdale ranges and Bellshiel demolition "
                "range are being used for live firing."
            ),
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )


def test_official_government_defense_procurement_and_operations_remain_relevant():
    """Substantive official government defense announcements must remain relevant."""
    assert (
        classify_relevance(
            "UK Ministry of Defence awards £400m contract for air defence missiles",
            snippet="New surface-to-air missile systems to be deployed across armed forces.",
            source_name="UK Ministry of Defence",
        )
        == "relevant"
    )
    assert (
        classify_relevance(
            "UK Carrier Strike Group departs for major multinational exercise",
            snippet="HMS Queen Elizabeth deploys with allied naval task group.",
            source_name="UK Ministry of Defence",
        )
        == "relevant"
    )


def test_evergreen_low_flying_guidance_pages_are_filtered_as_irrelevant():
    """Routine evergreen low-flying aircraft guidance and complaints pages must be filtered."""
    assert (
        classify_relevance(
            "Military low flying: questions and answers",
            snippet=(
                "Information and guidance on why and where military low flying training occurs."
            ),
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Guidance: Low flying military aircraft",
            snippet=(
                "Public information on low flying training and how to register flight complaints."
            ),
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Complaints about military low flying",
            snippet="How to submit enquiries and complaints about low flying aircraft in the UK.",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )


def test_holiday_halloween_roundtables_and_discussion_pieces_are_filtered_as_irrelevant():
    """Seasonal Halloween discussion articles and holiday reading lists must be filtered."""
    assert (
        classify_relevance(
            "Halloween discussion: The spooky season of national security",
            snippet="Roundtable discussion on defense policy ghosts and seasonal themes.",
            source_name="War on the Rocks",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Holiday reading list: Our favorite strategic defense books of the year",
            snippet="Editors share informal book recommendations and seasonal reading.",
            source_name="War on the Rocks",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Weekly roundup and daily debrief newsletter",
            snippet="A summary of top stories and podcast links from the past week.",
            source_name="Defense News",
        )
        == "irrelevant"
    )


def test_consequential_kinetic_reporting_on_holiday_not_suppressed():
    """Genuine kinetic operations occurring on a holiday must remain relevant."""
    assert (
        classify_relevance(
            "Airstrike kills 15 militants in overnight raid on Halloween morning",
            snippet="Forces strike militant weapons cache in counteroffensive operation.",
            source_name="BBC World News",
        )
        == "relevant"
    )


def test_jsp_and_defence_safety_management_guidance_filtered_as_irrelevant():
    """Verify administrative guidance, safety frameworks, and exercise timetables are filtered."""
    assert (
        classify_relevance(
            "Guidance: Defence Safety Management System (JSP 815)",
            snippet=(
                "Joint Service Publication 815 provides the Defence Safety Management System "
                "framework, enabling Defence organisations to develop tailored safety management."
            ),
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Guidance: Military low flying: MOD sponsored air exercises",
            snippet="MOD sponsored low flying exercises within the UK low flying system.",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
    assert (
        classify_relevance(
            "Armed forces pensions",
            snippet="Information related to the Armed Forces Pension Schemes.",
            source_name="UK Ministry of Defence",
        )
        == "irrelevant"
    )
