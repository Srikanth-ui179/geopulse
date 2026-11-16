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
        r"\b(football(?:er|ers)?|soccer|cricket|premier league|nfl|nba|tennis|golf|olympics|"
        r"pitch|coaching\s+footballers?|athlete|athletes|sports?)\b",
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
        r"\b(personal recovery|rehabilitation|coma|surviving\s+(?:an?\s+)?(?:air\s*)?strike|"
        r"recovering\s+from\s+injuries|back\s+on\s+the\s+pitch|sports?\s+recovery|human-interest|"
        r"refuses\s+to\s+give\s+up\s+after\s+injuries)\b",
        re.I,
    ),
    re.compile(
        r"\b(fight wildfires|wildfires?|bushfires?|forest fires?|firefighting|firefighters?|"
        r"(?<!\braf\s)(?<!\bbritish\s)(?<!\buk\s)(?<!\beurofighter\s)typhoons?(?!\s+(?:fighters?|jets?|aircraft|fleet|squadron|pilot|pilots))|"
        r"hurricane|cyclone|tornado|earthquake|tsunami|flights? cancelled as typhoon)\b",
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
        r"\b(?:established long-term relationship|relationship status for service personnel|"
        r"service family accommodation|civilian accommodation allowance|forces help to buy|"
        r"relocation allowance|pension scheme updates?|service personnel welfare benefits|"
        r"recording and approving a defence-recognised)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:ash dieback|estate tree safety|invasive non-native species|woodland management plan|"
        r"forestry on the .*?estate|deer management|conservation on the .*?estate|"
        r"ash dieback on the (?:ministry of defence|mod) estate|"
        r"woodlands across the (?:ministry of defence|mod) training estate)\b",
        re.I,
    ),
    re.compile(
        r"\b(?:firing times|range firing times|firing notice|firing schedule|"
        r"range clearance times|byelaws? for .*? range|public access to .*? range|"
        r"dates and times when .*? ranges? (?:are|is) being used for live firing)\b",
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
        r"(?:\b(?:free\s+)?(?:ebook|e-book|whitepaper|webinar|marketing\s+newsletter)\b.*?"
        r"\b(?:download|register|sign\s+up|join\s+our|get\s+(?:the\s+)?ground\s+truth)\b|"
        r"\b(?:download\s+(?:the|your|our|this)\s+(?:free\s+)?(?:ebook|whitepaper|guide|report)|"
        r"register\s+(?:now|today)\s+for\s+(?:the\s+)?(?:webinar|roundtable|summit)|"
        r"join\s+(?:us\s+for\s+)?(?:the\s+)?(?:webinar|roundtable)|"
        r"missed\s+[\w\s'’-]+\?\s*get\s+(?:the\s+)?ground\s+truth\s+from\s+.*?free\s+ebook)\b|"
        r"/(?:free-ebook|download-ebook|webinar|whitepaper)/)",
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
    re.compile(
        r"\b(?:(?:announces?|names?|appoints?)\s+[\w\s'’-]+?\s+"
        r"(?:as\s+(?:the\s+)?(?:new\s+)?(?:president|ceo|cfo|coo|chief executive|"
        r"managing director|board member))|"
        r"(?:names?|appoints?|hires?)\s+(?:new\s+)?"
        r"(?:president|ceo|cfo|coo|chief executive|managing director)|"
        r"executive leadership (?:changes?|appointments?|transition))\b",
        re.I,
    ),
    re.compile(
        r"\b(?:digital (?:veteran|id|identity|service|entitlement)\s+cards?|"
        r"veterans?\s+(?:id|cards?|card\s+scheme)|apply\s+for\s+(?:new\s+)?digital\s+(?:veteran\s+)?card|"
        r"veterans?\s+card\s+applications?|"
        r"(?:reaches|reached|hits|hit|surpasses|surpassed|milestone of)\s+[\d,]+\s+"
        r"(?:downloads?|applications?|users?|subscribers?|cards?)|"
        r"[\d,]+\s+(?:downloads?|applications?|veterans?\s+apply|users?\s+register)|"
        r"digital\s+service\s+milestone|portal\s+milestone|service\s+card\s+applications?)\b",
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
            r"downing of .* aircraft|"
            r"(?:aircraft|plane|jet|helicopter|drone)\s+(?:downed|shot down)|"
            r"anti[- ]aircraft\s+(?:artillery|fire|missiles?)|"
            r"soldiers? killed|soldiers? wounded|troops killed|"
            r"gun battle|killed in attack|attack by suspected .* gunmen)\b",
            re.I,
        ),
        re.compile(
            r"\b(drone strike|drone strikes|artillery barrage|combat losses|"
            r"(?:drone|missile|rocket)\s+attacks?)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:(?:russian|israeli|ukrainian|us|coalition|missile|rocket|drone|artillery|deadly|overnight|precision|retaliatory|air)\s+)?"
            r"(?:strikes?|bombings?|bombardments?|shellings?|attacks?)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:war in (?:ukraine|gaza|sudan|yemen|middle east|lebanon|iran)|"
            r"(?:iran|gaza|ukraine|russia|lebanon|yemen|sudan)\s+war|"
            r"gaza war|ukraine war|russia-ukraine war|israel-hamas war|israel-hezbollah conflict|"
            r"armed conflict|escalation in (?:middle east|gaza|ukraine)|middle east war)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:bombing|bombed|bombs?|artillery fire|missiles? fired|rockets? fired|"
            r"intercepted missiles?|cross-border fire|frontline fighting|incursions?|invaded?)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:houthi (?:attacks?|strikes?|missiles?|drones?)|"
            r"red sea (?:attacks?|strikes?|shipping)|targeted vessel|"
            r"merchant ship attacked|maritime strike)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:killed in (?:strikes?|fighting|attack|bombing|shelling)|deadly strike|"
            r"killed by (?:shelling|strikes?|missiles?)|bomb damage)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:intercept(?:s|ed|ing)?\s+(?:[\w-]+\s+)?(?:missiles?|rockets?|drones?)|"
            r"cruise\s+missiles?|ballistic\s+missiles?|missile\s+barrage|overnight\s+barrage)\b",
            re.I,
        ),
    ),
    TopicCategory.MILITARY_OPERATIONS: (
        re.compile(
            r"\b(troop deployment|troops deploy|troop movements?|military movements?|"
            r"military exercise|joint drills|usfk|united states forces korea)\b",
            re.I,
        ),
        re.compile(
            r"\b(military demonstration|military drill|live[- ]fire (?:[\w-]+\s+)*exercise|"
            r"combat drill|readiness drill|field exercise|ateş serbest|war games|"
            r"military maneuvers?)\b",
            re.I,
        ),
        re.compile(
            r"\b(space operations?|military satellites?|satellite maneuvers?|orbital maneuvers?|"
            r"tails? (?:a )?(?:[\w-]+ )*satellite|shadowing (?:a )?(?:[\w-]+ )*satellite|"
            r"space warfare|space wars?|orbital proximity operations?|rendezvous and proximity)\b",
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
        re.compile(
            r"\b(?:(?:rushes?|deploys?|deployed|mobiliz(?:e|ed|ation)|sends?)\s+"
            r"(?:thousands of\s+)?(?:troops|forces|soldiers|reservists))\b",
            re.I,
        ),
        re.compile(
            r"\b(?:taiwan strait|south china sea|blockade exercise|encirclement drills?|"
            r"air defence identification zone|adiz violations?|military patrol)\b",
            re.I,
        ),
    ),
    TopicCategory.DEFENCE_POLICY: (
        re.compile(
            r"\b(defence policy|defense policy|strategic analysis|strategic assessment|"
            r"(?:military|strategic|counter-drone|naval|air|warfare)\s+doctrine|"
            r"national defence)\b",
            re.I,
        ),
        re.compile(r"\b(nato alliance|security pact|aukus|pentagon report|white paper)\b", re.I),
        re.compile(r"\b(defence budget|defense spending|military reform|conscription)\b", re.I),
        re.compile(
            r"\b(military reform|institutional (?:change|reform)|command restructuring|"
            r"defence transformation|defense transformation|military transformation|"
            r"institutional transformation|warfare command|restructuring .* command|"
            r"prepare for modern war|preparing for modern warfare)\b",
            re.I,
        ),
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
        re.compile(
            r"\b(defence ministers?['’]? statement|defense ministers?['’]? statement|"
            r"joint defence statement|joint defense statement|e5 defence|e5 defense)\b",
            re.I,
        ),
        re.compile(
            r"\b(oversight of (?:the )?(?:department of defense|dod|military|pentagon)|"
            r"congressional oversight|defense authorization|ndaa|"
            r"executive authority over (?:the )?(?:military|dod|defense)|oversight role)\b",
            re.I,
        ),
        re.compile(
            r"\b(department of defense|defense department|pentagon|dod)\b",
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
            r"\b(interceptor missiles?|air[- ]defense missiles?|air[- ]defence missiles?|"
            r"drone interceptors?|counter[- ]drone (?:systems?|missiles?)|"
            r"anti[- ]aircraft missiles?|surface[- ]to[- ]air missiles?|sam systems?|"
            r"unveils? (?:[\w-]+ )*missiles?|unveils? (?:[\w-]+ )*interceptor|"
            r"offering (?:[\w-]+ )*missiles?|one-way attack (?:drone|unmanned aerial vehicle|uav)|"
            r"attack drones?|suicide drones?|kamikaze drones?)\b",
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
        re.compile(
            r"\b(offshore patrol vessels?|patrol vessels?|naval systems?|"
            r"patrol boats?|naval vessels?)\b",
            re.I,
        ),
    ),
    TopicCategory.DIPLOMACY: (
        re.compile(
            r"\b(peace talks|peace deal|peace accord|peace agreement|peace treaty|peace summit|"
            r"ceasefire|truce|armistice|cessation of hostilities)\b",
            re.I,
        ),
        re.compile(
            r"\b(diplomatic (?:summit|talks|envoys?|negotiations?|mission|dialogue)|"
            r"bilateral talks|trilateral talks|un security council resolution)\b",
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
        re.compile(
            r"\b(diplomatic recognition|"
            r"recogni[sz]e (?:the )?(?:taliban|government|regime|state|sovereignty)|"
            r"diplomatic ties|diplomatic relations|accredit(?:ed)? (?:ambassador|envoy)|"
            r"accept(?:s|ed)? credentials|present(?:s|ed)? credentials|"
            r"establish(?:ing)? (?:diplomatic )?relations|reopen(?:s|ed|ing)? embassy|"
            r"normali[sz]ation of relations|normali[sz]e ties|"
            r"taliban\s+(?:diplomacy|envoys?|ambassador|embassy|relations|recognition|talks))\b",
            re.I,
        ),
        re.compile(
            r"\b(?:ceasefire talks|ceasefire negotiations|gaza ceasefire|ukraine peace talks|"
            r"hostage deal|hostage negotiations|security guarantees|bilateral security pact|"
            r"diplomatic summit|envoy meets?|un security council|"
            r"foreign policy|geopolitical alliance|diplomatic isolation|"
            r"isolate (?:[\w'-]+\s+)*in washington|foreign relations|bilateral relations)\b",
            re.I,
        ),
    ),
    TopicCategory.SANCTIONS: (
        re.compile(
            r"\b((?:economic |trade |financial |energy )?sanctions?|eases? sanctions|"
            r"lifts? sanctions|sanctions relief|sanctions waiver|trade embargo|"
            r"asset freeze|export controls?|eases? (?:trade\s+|import\s+|export\s+)?restrictions|"
            r"lifts? restrictions|(?:import|export|trade)\s+restrictions)\b",
            re.I,
        ),
        re.compile(
            r"\b(sanctioned entity|financial blacklist|oil price cap|customs tariff|"
            r"diesel deal|petroleum sanctions|energy crunch)\b",
            re.I,
        ),
        re.compile(
            r"\b(semiconductor export controls?|chip export controls?|advanced chip controls?|"
            r"semiconductor ban|chip sanctions|technology sanctions|chip embargo)\b",
            re.I,
        ),
        re.compile(
            r"\b(?:sanctions? against|new sanctions|tightens? sanctions|sanctions package|"
            r"oil embargo|arms embargo|trade restrictions|"
            r"designated under sanctions|asset freezes?)\b",
            re.I,
        ),
    ),
    TopicCategory.TERRORISM_INSURGENCY: (
        re.compile(r"\b(suicide bombing|car bomb|militants ambush|insurgent attack)\b", re.I),
        re.compile(
            r"\b(jihadist group|hostage execution|improvised explosive device|ied blast)\b", re.I
        ),
        re.compile(
            r"\b(terror(?:ist|ism)?\s+(?:plot|attack|cell|threat|suspects?|network|financing)|"
            r"suspected\s+terror(?:\s+plot)?|foiled\s+(?:terror\s+)?plot|counter-?terrorism)\b",
            re.I,
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
        re.compile(
            r"\b(?:border clash|border skirmish|cross-border incident|drone shot down|"
            r"hijacking attempt|maritime incident|explosive device|"
            r"consulate attack|embassy breach)\b",
            re.I,
        ),
    ),
    TopicCategory.INTELLIGENCE_ESPIONAGE: (
        re.compile(
            r"\b(espionage|treason|classified documents?|national defense information|"
            r"military secrets?|selling secrets?|leaked? classified|counterintelligence|"
            r"foreign agents?|fara violation|spy ring|passing intelligence|intelligence probe|"
            r"surveillance operation|clandestine operation|indicted for espionage|"
            r"arrested for espionage|intelligence leak|intelligence investigation)\b",
            re.I,
        ),
        re.compile(
            r"\b(cia|mi6|mossad|fsb|gru|mss|gchq|nsa|counter-espionage|wiretapping|"
            r"intelligence assessment|signals intelligence|human intelligence|humint|sigint)\b",
            re.I,
        ),
    ),
    TopicCategory.MILITARY_TECHNOLOGY: (
        re.compile(
            r"\b(military technology|defence technology|defense technology|military ai|"
            r"artificial intelligence in defense|autonomous warfare|c4isr|quantum computing|"
            r"directed energy|laser weapon|railgun|electronic countermeasures|"
            r"darpa|defence innovation|defense innovation|technet|technet indo-pacific|afcea|"
            r"technology symposium|unmanned systems? testing|combat cloud)\b",
            re.I,
        ),
    ),
}


STRATEGIC_LEGAL_OVERRIDE_PATTERN: re.Pattern[str] = re.compile(
    r"\b(treason|espionage|classified|secrets?|war crimes?|atrocit\w*|"
    r"homicide|murder|bribery|corruption|reform|mutiny|sabotage|trafficking)\b",
    re.I,
)


HISTORICAL_CONFLICT_PATTERN = re.compile(
    r"\b(?:battle of\s+[\w\s'-]+?|bail[eé]n|waterloo|trafalgar|gettysburg|antietam|"
    r"verdun|somme|passchendaele|gallipoli|stalingrad|normandy landing|"
    r"crimean war|boer war|peloponnesian war|punic war|crusades|ancient warfare|"
    r"roman (?:legion|army|empire)|medieval (?:warfare|knights|siege)|"
    r"napoleon(?:ic)?(?:\s+wars?|\s+era)?|"
    r"historical case study|historical perspective on the battle|"
    r"lessons from the battle of|historical military strategy of|"
    r"in\s+(?:1[4-9]\d\d|1[0-3]\d\d)|"
    r"on the morning of [a-z]+ \d{1,2},\s*(?:1[4-9]\d\d|1[0-3]\d\d))\b",
    re.I,
)

HISTORICAL_HOOK_PATTERN = re.compile(
    r"\b(?:dna (?:testing|analysis|match|identification)|"
    r"remains? (?:identified|laid to rest|unearthed|discovered|recovered|found)|"
    r"laid to rest|unearthed|archaeolog\w*|ceremony|memorial service|commemorat\w*|"
    r"newly declassified|unsealed (?:records|files)|inquest|inquiry (?:finds|reveals|concludes)|"
    r"new evidence uncovered|posthumous|apology|pardon|museum exhibit)\b",
    re.I,
)


class RuleBasedClassifier(BaseClassifier):
    """Deterministic rule-based geopolitical relevance and topic classifier."""

    def classify_text(
        self,
        title: str,
        snippet: str | None = None,
        source_name: str | None = None,
        url: str | None = None,
    ) -> ClassificationResult:
        clean_title = (title or "").strip()
        clean_snippet = (snippet or "").strip()
        clean_url = (url or "").strip()
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

        # Check for historical military retrospectives without contemporary news hook
        if HISTORICAL_CONFLICT_PATTERN.search(combined_text):
            if not HISTORICAL_HOOK_PATTERN.search(combined_text):
                return ClassificationResult(
                    relevance_status="irrelevant",
                    primary_topic=TopicCategory.OTHER_GEOPOLITICAL,
                    confidence=0.90,
                    explanation=(
                        "Historical military retrospective or case study "
                        "without a contemporary news hook"
                    ),
                )

        has_strategic_legal = bool(STRATEGIC_LEGAL_OVERRIDE_PATTERN.search(combined_text))
        # Check for routine historical commemorations (e.g. WWI/WWII soldier burials)
        # These are preserved as relevant but categorized as OTHER_GEOPOLITICAL to prevent
        # displacing modern operational defence developments and active procurement contracts.
        _HIST_COMMEMORATION_PAT = re.compile(
            r"\b(?:laid to rest|buried with (?:full )?military honours?|reinterment|re-interred|"
            r"remains? (?:identified|buried|interred|discovered|unearthed)|memorial service|"
            r"cemetery|cwgc|war graves? commission)\b",
            re.I,
        )
        _HIST_CONTEXT_PAT = re.compile(
            r"\b(?:first world war|second world war|wwi|wwii|great war|"
            r"somme|passchendaele|ypres|verdun|gallipoli|flanders|over a century after|"
            r"identified after \d+ years|century after the war)\b",
            re.I,
        )
        if HISTORICAL_CONFLICT_PATTERN.search(combined_text) or _HIST_CONTEXT_PAT.search(
            combined_text
        ):
            if _HIST_COMMEMORATION_PAT.search(combined_text) and not has_strategic_legal:
                return ClassificationResult(
                    relevance_status="relevant",
                    primary_topic=TopicCategory.OTHER_GEOPOLITICAL,
                    confidence=0.80,
                    explanation="Historical military commemoration or reinterment ceremony",
                )

        is_defence_outlet = clean_source in DEDICATED_DEFENCE_SOURCES

        # Check negative signals (unless overridden by strategic legal cases,
        # military aircraft, procurement contracts, or operational exercises)
        has_strategic_legal = bool(STRATEGIC_LEGAL_OVERRIDE_PATTERN.search(combined_text))
        has_military_aircraft = bool(
            re.search(
                r"\b((?:raf|british|uk|eurofighter)\s+typhoon|eurofighter(?:\s+typhoon)?|"
                r"typhoon\s+(?:fighters?|jets?|aircraft|fleet|squadron))\b",
                combined_text,
                re.I,
            )
        )
        has_procurement_contract = bool(
            re.search(
                r"\b(procurement|contract|arms sale|missile deal|weapons? contract|"
                r"awarded .* contract)\b",
                combined_text,
                re.I,
            )
        )
        has_operational = bool(
            re.search(
                r"\b(multinational exercise|joint exercise|nato exercise|carrier deployment|"
                r"troops deploy|scrambled jets|naval patrol|air patrol|border reinforcement)\b",
                combined_text,
                re.I,
            )
        )
        is_promotional = bool(
            re.search(
                r"(?:\b(?:free\s+)?(?:ebook|e-book|whitepaper|webinar|marketing\s+newsletter)\b.*?"
                r"\b(?:download|register|sign\s+up|join\s+our|get\s+(?:the\s+)?ground\s+truth)\b|"
                r"\b(?:download\s+(?:the|your|our|this)\s+(?:free\s+)?(?:ebook|whitepaper|guide|report)|"
                r"register\s+(?:now|today)\s+for\s+(?:the\s+)?(?:webinar|roundtable|summit)|"
                r"join\s+(?:us\s+for\s+)?(?:the\s+)?(?:webinar|roundtable)|"
                r"missed\s+[\w\s'’-]+\?\s*get\s+(?:the\s+)?ground\s+truth\s+from\s+.*?free\s+ebook)\b|"
                r"/(?:free-ebook|download-ebook|webinar|whitepaper)/)",
                combined_text,
                re.I,
            )
        )
        irrelevant_hit = any(pat.search(combined_text) for pat in IRRELEVANT_PATTERNS)
        if (
            irrelevant_hit
            and not is_promotional
            and (
                has_strategic_legal
                or has_military_aircraft
                or has_procurement_contract
                or has_operational
            )
        ):
            is_sports = bool(
                re.search(
                    r"\b(premier league|football(?:er|ers)?|soccer|nba|nfl|tennis|golf|"
                    r"celebrity|hollywood|box office|horoscope|lifestyle|recipe|dating|pitch|"
                    r"athlete|athletes|personal recovery|rehabilitation|coma|"
                    r"back\s+on\s+the\s+pitch|refuses\s+to\s+give\s+up\s+after\s+injuries)\b",
                    combined_text,
                    re.I,
                )
            )
            if not is_sports:
                irrelevant_hit = False

        # Liveblog title normalization:
        # If the article is a rolling live update (e.g. "Iran war live: Qatar says...")
        # extract the substantive sub-headline after "live:" so the topic reflects
        # the actual development rather than the generic war liveblog wrapper.
        topic_title = clean_title
        liveblog_match = re.search(r"^(?:.*?war\s+)?live:\s*(.+)$", clean_title, re.I)
        if liveblog_match:
            sub_title = liveblog_match.group(1).strip()
            if len(sub_title) > 10:
                topic_title = sub_title

        # Score topic matches (give title 2x weight over snippet)
        topic_scores: dict[TopicCategory, int] = {}
        for category, patterns in TOPIC_PATTERNS.items():
            score = 0
            for pat in patterns:
                # Title match is worth 2 points
                if pat.search(topic_title):
                    score += 2
                # Snippet match is worth 1 point
                elif clean_snippet and pat.search(clean_snippet):
                    score += 1
            if score > 0:
                topic_scores[category] = score

        # Autonomous warfare doctrine & institutional reform preference:
        # When autonomous warfare terms occur alongside institutional reform, doctrine,
        # command restructuring, or defence transformation, prefer defence_policy over
        # military_operations. Do not alter genuine kinetic combat or drone strikes.
        has_autonomous_warfare = bool(
            re.search(
                r"\b(autonomous warfare|mass autonomous|deception in .* warfare|mass drone)\b",
                combined_text,
                re.I,
            )
        )
        has_institutional_reform = bool(
            re.search(
                r"\b(institutional (?:change|reform)|military reform|doctrine|"
                r"command restructuring|defence transformation|defense transformation|"
                r"military transformation|institutional transformation|warfare command|"
                r"prepare for modern war|preparing for modern warfare)\b",
                combined_text,
                re.I,
            )
        )
        has_kinetic_or_strike = bool(
            re.search(
                r"\b(airstrike|air strike|drone strike|bombing|shelling|casualties|"
                r"killed in action|combat losses|ground offensive|counteroffensive|gun battle)\b",
                combined_text,
                re.I,
            )
        )
        if has_autonomous_warfare and has_institutional_reform and not has_kinetic_or_strike:
            current_mp = topic_scores.get(TopicCategory.MILITARY_OPERATIONS, 0)
            current_dp = topic_scores.get(TopicCategory.DEFENCE_POLICY, 0)
            topic_scores[TopicCategory.DEFENCE_POLICY] = max(current_dp, current_mp + 1, 2)
            if (
                TopicCategory.MILITARY_OPERATIONS in topic_scores
                and topic_scores[TopicCategory.MILITARY_OPERATIONS]
                >= topic_scores[TopicCategory.DEFENCE_POLICY]
            ):
                topic_scores[TopicCategory.MILITARY_OPERATIONS] = (
                    topic_scores[TopicCategory.DEFENCE_POLICY] - 1
                )

        # Strategic Commentary & Analytical Essays vs Armed Conflict:
        # If an article is strategic commentary, analysis, or an essay discussing societal
        # exhaustion, war fatigue, or long-term strategic choices (e.g. regarding Ukraine) without
        # breaking kinetic strikes in the title, prefer DEFENCE_POLICY rather than ARMED_CONFLICT.
        is_opinion_url = bool(
            clean_url
            and re.search(
                r"/(?:opinions?|commentary|analysis|ideas|perspectives?)/", clean_url, re.I
            )
        )
        is_strategic_commentary = is_opinion_url or bool(
            re.search(
                r"\b(strategic analysis|commentary|essay|opinion|perspective|"
                r"societal exhaustion|war fatigue|lessons learned|strategic dilemma|"
                r"path forward for|future of the war|analytical assessment|"
                r"military strategy|strategic doctrine|operational doctrine|"
                r"counter-drone strategy|defense strategy|defence strategy)\b",
                combined_text,
                re.I,
            )
        )
        has_kinetic_title = bool(
            re.search(
                r"\b(airstrike|air strike|airstrikes|air strikes|bombing|shelling|"
                r"missile strike|frontline clashes|ground offensive|counteroffensive|"
                r"soldiers? killed|troops killed|combat losses|gun battle|attackers? killed|"
                r"drone strike|drone strikes|(?:drone|missile|rocket)\s+attacks?|"
                r"artillery barrage|military strike|military strikes|"
                r"downing of|shot down|combat operations?|naval clash|clashes?|incursion)\b",
                clean_title,
                re.I,
            )
        )

        # Political/Electoral or Macroeconomic Effects vs Armed Conflict:
        # If an article focuses on political, electoral, or economic fallout (e.g. elections,
        # midterms, voting, party control, inflation, cost of living) and does not describe
        # active kinetic strikes in the title, background mentions of war must not misclassify
        # the article as ARMED_CONFLICT.
        is_political_or_electoral = bool(
            re.search(
                r"\b(midterms?|elections?|voters?|democrats?|republicans?|congressional|"
                r"presidential (?:race|campaign|election)|political fallout|cabinet reshuffle|"
                r"parliamentary vote|coalition government|win\s+the\s+midterms|electoral)\b",
                clean_title,
                re.I,
            )
        )
        has_economic_fallout = bool(
            re.search(
                r"\b(inflation|energy (?:prices|markets?|security|reserves?)|"
                r"gas (?:prices|reserves?|storage)|cost of living|"
                r"economic (?:fallout|impact|uncertainty|growth)|"
                r"market (?:turmoil|uncertainty|volatility)|recession risks?|gdp growth)\b",
                clean_title,
                re.I,
            )
        )
        if (is_political_or_electoral or has_economic_fallout) and not has_kinetic_title:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
                topic_scores[TopicCategory.OTHER_GEOPOLITICAL] = max(
                    topic_scores.get(TopicCategory.OTHER_GEOPOLITICAL, 0),
                    ac_score,
                    2,
                )

        if is_strategic_commentary and not has_kinetic_title and not is_political_or_electoral:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
                topic_scores[TopicCategory.DEFENCE_POLICY] = max(
                    topic_scores.get(TopicCategory.DEFENCE_POLICY, 0),
                    ac_score,
                )

        # Sanctions and economic/energy agreements vs Armed Conflict:
        # If an article focuses on sanctions, trade agreements, energy deals, or financial controls,
        # and has no kinetic strikes or armed conflict events in the title, do not let incidental
        # mentions of "battlefield efforts" or "war funding" in the snippet misclassify it as
        # ARMED_CONFLICT.
        has_sanctions = topic_scores.get(TopicCategory.SANCTIONS, 0) >= 2 or bool(
            re.search(
                r"\b(sanctions?|trade deal|diesel deal|energy deal|tariff|export controls?)\b",
                clean_title,
                re.I,
            )
        )
        if has_sanctions and not has_kinetic_title:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
                topic_scores[TopicCategory.SANCTIONS] = max(
                    topic_scores.get(TopicCategory.SANCTIONS, 0),
                    ac_score,
                )

        # Weapons Procurement vs Armed Conflict:
        # If an article is explicitly about buying, procuring, or committing to contracts
        # and has no kinetic strikes/casualties in title, prefer WEAPONS_PROCUREMENT.
        has_procurement_title = bool(
            re.search(
                r"\b(procurement|contract|arms sale|arms deal|missile deal|weapons? contract|"
                r"buys?|to buy|commits? to .* missiles?|orders? .* missiles?|acquires?)\b",
                clean_title,
                re.I,
            )
        )
        if has_procurement_title and not has_kinetic_title:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
                topic_scores[TopicCategory.WEAPONS_PROCUREMENT] = max(
                    topic_scores.get(TopicCategory.WEAPONS_PROCUREMENT, 0),
                    ac_score + 1,
                )

        # Intelligence & Espionage vs Armed Conflict & Cyber-Hybrid Warfare:
        # Espionage investigations, treason, classified document leaks, and spy scandals
        # on military bases or domestic settings must not be misclassified as ARMED_CONFLICT
        # or CYBER_HYBRID_WARFARE (even if background text mentions hybrid threats).
        has_espionage = bool(
            re.search(
                r"\b(espionage|treason|classified\s+documents?|national\s+defense\s+information|"
                r"military\s+secrets?|selling\s+secrets?|spy\s+ring|counterintelligence|"
                r"foreign\s+agent|fara\s+violation|passing\s+intelligence|intelligence\s+probe|"
                r"indicted\s+for\s+espionage|arrested\s+for\s+espionage|espionage\s+investigation|"
                r"espionage\s+allegations)\b",
                combined_text,
                re.I,
            )
        )
        has_espionage_title = bool(
            re.search(
                r"\b(espionage|treason|spying|spy\b|classified\s+documents?|national\s+defense\s+information|"
                r"military\s+secrets?|selling\s+secrets?|counterintelligence)\b",
                clean_title,
                re.I,
            )
        )

        # Public vigilance campaigns & force protection advisories vs Armed Conflict:
        # Military awareness campaigns, security advisories, and vigilance programs
        # (e.g. EUCOM "See Something, Do Something") must not automatically qualify
        # as ARMED_CONFLICT. Preserve them as defence policy / cyber-hybrid warfare
        # only when there is no specific espionage arrest/investigation.
        is_vigilance_or_advisory = bool(
            re.search(
                r"\b(vigilance campaign|public awareness campaign|security advisory|"
                r"security advisories|force protection (?:campaign|initiative|advisory|mindset)|"
                r"see something,? do something|digital discipline|watch for suspicious behavior|"
                r"cover their digital tracks|unusual behavior or unattended items)\b",
                combined_text,
                re.I,
            )
        )
        if is_vigilance_or_advisory and not has_kinetic_title and not has_espionage:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
                preferred_cat = (
                    TopicCategory.CYBER_HYBRID_WARFARE
                    if bool(
                        re.search(
                            r"\b(hybrid|cyber|digital|drones?\s+incursions?)\b",
                            combined_text,
                            re.I,
                        )
                    )
                    else TopicCategory.DEFENCE_POLICY
                )
                topic_scores[preferred_cat] = max(
                    topic_scores.get(preferred_cat, 0),
                    ac_score + 1,
                    3,
                )

        if (has_espionage_title or has_espionage) and not has_kinetic_title:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
            else:
                ac_score = 0
            cyber_score = topic_scores.get(TopicCategory.CYBER_HYBRID_WARFARE, 0)
            topic_scores[TopicCategory.INTELLIGENCE_ESPIONAGE] = max(
                topic_scores.get(TopicCategory.INTELLIGENCE_ESPIONAGE, 0),
                ac_score + 1,
                cyber_score + 2 if has_espionage_title else cyber_score + 1,
                4 if has_espionage_title else 3,
            )

        # Terrorism & Counter-Terrorism vs Armed Conflict:
        # Terror plots, threats, and investigations prominently reported in the title must not
        # be misclassified as ARMED_CONFLICT due to incidental background references to ongoing
        # wars (e.g. "war in Iran") or distant strike operations.
        has_terrorism_title = bool(
            re.search(
                r"\b(terror(?:ist|ism)?\s+(?:plot|attack|cell|threat|suspects?|network|financing)|"
                r"suspected\s+terror(?:\s+plot)?|foiled\s+(?:terror\s+)?plot|counter-?terrorism|"
                r"indicted\s+in\s+.*plot|arrested\s+on\s+terror)\b",
                clean_title,
                re.I,
            )
        )
        if has_terrorism_title:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                ac_score = topic_scores.pop(TopicCategory.ARMED_CONFLICT)
            else:
                ac_score = 0
            topic_scores[TopicCategory.TERRORISM_INSURGENCY] = max(
                topic_scores.get(TopicCategory.TERRORISM_INSURGENCY, 0),
                ac_score + 1,
                3,
            )

        # TechNet Indo-Pacific & Industry Conferences vs Procurement / Armed Conflict:
        # Industry symposia, conference keynotes, and exhibitions (e.g. TechNet Indo-Pacific)
        # without signed procurement contracts must be categorized as MILITARY_TECHNOLOGY or
        # DEFENCE_POLICY.
        has_technet_or_symposium = bool(
            re.search(
                r"\b(technet|technet\s+indo-pacific|afcea|industry\s+symposium|"
                r"defence\s+conference|defense\s+conference|conference\s+panel|"
                r"symposium\s+highlights?|conference\s+opens?)\b",
                combined_text,
                re.I,
            )
        )
        if has_technet_or_symposium and not has_procurement_title and not has_kinetic_title:
            if TopicCategory.ARMED_CONFLICT in topic_scores:
                topic_scores.pop(TopicCategory.ARMED_CONFLICT)
            if TopicCategory.WEAPONS_PROCUREMENT in topic_scores:
                wp_score = topic_scores.pop(TopicCategory.WEAPONS_PROCUREMENT)
                topic_scores[TopicCategory.MILITARY_TECHNOLOGY] = max(
                    topic_scores.get(TopicCategory.MILITARY_TECHNOLOGY, 0),
                    wp_score,
                    2,
                )
            elif (
                TopicCategory.MILITARY_TECHNOLOGY not in topic_scores
                and TopicCategory.DEFENCE_POLICY not in topic_scores
            ):
                topic_scores[TopicCategory.MILITARY_TECHNOLOGY] = 2

        # Evaluate if negative signal matches
        if irrelevant_hit:
            is_retrospective_or_essay = bool(
                re.search(
                    r"\b(then and now: how the landscape has changed|"
                    r"how the landscape has changed since|"
                    r"retrospective overview|historical lookback on the war|"
                    r"deception in mass autonomous warfare|theoretical monograph|"
                    r"doctrinal essay)\b",
                    combined_text,
                    re.I,
                )
            )
            if is_retrospective_or_essay:
                return ClassificationResult(
                    relevance_status="irrelevant",
                    primary_topic=TopicCategory.OTHER_GEOPOLITICAL,
                    confidence=0.85,
                    explanation=(
                        "Retrospective war overview or theoretical essay "
                        "without an active breaking news hook"
                    ),
                )

            if topic_scores:
                top_category, _ = sorted(topic_scores.items(), key=lambda kv: kv[1], reverse=True)[
                    0
                ]
                # If a civilian/sports/entertainment report matches conflict keywords,
                # an isolated incident at a public venue is a security incident,
                # not battlefield warfare.
                if top_category == TopicCategory.ARMED_CONFLICT:
                    top_category = TopicCategory.SECURITY_INCIDENT

                if top_category == TopicCategory.SECURITY_INCIDENT:
                    is_sports_or_lifestyle = bool(
                        re.search(
                            r"\b(premier league|football(?:er|ers)?|soccer|nba|nfl|tennis|golf|"
                            r"celebrity|hollywood|box office|horoscope|lifestyle|recipe|dating|"
                            r"pitch|coaching\s+footballers?|athlete|athletes|"
                            r"personal recovery|rehabilitation|coma|back\s+on\s+the\s+pitch|"
                            r"refuses\s+to\s+give\s+up\s+after\s+injuries)\b",
                            combined_text,
                            re.I,
                        )
                    )
                    if is_sports_or_lifestyle:
                        is_personal_rehab = bool(
                            re.search(
                                r"\b(personal recovery|rehabilitation|coma|back\s+on\s+the\s+pitch|"
                                r"refuses\s+to\s+give\s+up\s+after\s+injuries|recovering\s+from\s+injuries|"
                                r"surviving\s+(?:an?\s+)?(?:air\s*)?strike|coaching\s+youngsters|"
                                r"coaching\s+footballers?)\b",
                                combined_text,
                                re.I,
                            )
                        )
                        if is_personal_rehab:
                            return ClassificationResult(
                                relevance_status="irrelevant",
                                primary_topic=TopicCategory.OTHER_GEOPOLITICAL,
                                confidence=0.85,
                                explanation=(
                                    "Personal recovery, sports, or human-interest story "
                                    "mentioning conflict as background"
                                ),
                            )
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
