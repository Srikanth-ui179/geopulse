"""Regression fixtures and verification for geopolitical news relevance classification."""

import pytest

from geopulse.classification.models import TopicCategory
from geopulse.classification.rules import RuleBasedClassifier

FIXTURES = [
    # Ukraine armed conflict & strikes
    (
        "Russian missile strikes kill civilians across Ukraine",
        "Overnight bombardment hits energy infrastructure in Kyiv and Kharkiv.",
        "BBC News",
        "relevant",
        TopicCategory.ARMED_CONFLICT,
    ),
    # Gaza armed conflict & strikes
    (
        "Israeli airstrikes kill dozens in northern Gaza",
        "Artillery shelling and drone strikes continue in Jabalia refugee camp.",
        "Al Jazeera English",
        "relevant",
        TopicCategory.ARMED_CONFLICT,
    ),
    # Red Sea maritime conflict
    (
        "Houthi rebels target container ship in Red Sea with anti-ship missiles",
        "UKMTO confirms vessel damaged after drone strike off the coast of Yemen.",
        "BBC News",
        "relevant",
        TopicCategory.ARMED_CONFLICT,
    ),
    # Taiwan military operations
    (
        "China stages joint military drills surrounding Taiwan",
        "PLA forces conduct live-fire exercises in the Taiwan Strait simulating blockade.",
        "Al Jazeera English",
        "relevant",
        TopicCategory.MILITARY_OPERATIONS,
    ),
    # Sanctions package
    (
        "EU prepares 14th sanctions package against Russia",
        "Brussels targets oil tankers and shadow fleet carrying Russian crude.",
        "BBC News",
        "relevant",
        TopicCategory.SANCTIONS,
    ),
    # Diplomacy & ceasefire
    (
        "Gaza ceasefire talks resume in Doha with international mediators",
        "Mediators from Qatar, Egypt and US meet to negotiate hostage release and truce.",
        "Al Jazeera English",
        "relevant",
        TopicCategory.DIPLOMACY,
    ),
    # Foreign policy & diplomatic isolation
    (
        "Could the US midterms further isolate Israel in Washington?",
        "Analysis on US-Israel relations and foreign policy implications.",
        "Al Jazeera English",
        "relevant",
        TopicCategory.DIPLOMACY,
    ),
    # Military aid & defence policy
    (
        "US approves new military aid package for Ukraine",
        "Pentagon announces 425 million dollars in additional weapons and ammunition.",
        "BBC News",
        "relevant",
        TopicCategory.DEFENCE_POLICY,
    ),
    # Negative controls - sports, entertainment, local lifestyle
    (
        "Manchester United manager sacked after Premier League defeat",
        "Football club parts ways with coach following 2-1 loss.",
        "BBC News",
        "irrelevant",
        TopicCategory.OTHER_GEOPOLITICAL,
    ),
    (
        "Met Gala 2026: The best red carpet looks",
        "Celebrities arrive in high fashion at annual museum fundraiser.",
        "BBC News",
        "irrelevant",
        TopicCategory.OTHER_GEOPOLITICAL,
    ),
    (
        "Local bakery wins state award for sourdough recipe",
        "Family-owned shop in Portland recognized for artisanal bread.",
        "Al Jazeera English",
        "irrelevant",
        TopicCategory.OTHER_GEOPOLITICAL,
    ),
]


@pytest.fixture
def classifier() -> RuleBasedClassifier:
    return RuleBasedClassifier()


@pytest.mark.parametrize(
    ("title", "snippet", "source", "expected_relevance", "expected_topic"),
    FIXTURES,
)
def test_geopolitical_relevance_fixtures(
    classifier: RuleBasedClassifier,
    title: str,
    snippet: str,
    source: str,
    expected_relevance: str,
    expected_topic: TopicCategory,
) -> None:
    res = classifier.classify_text(title=title, snippet=snippet, source_name=source)
    assert res.relevance_status == expected_relevance, (
        f"Expected {expected_relevance} for '{title}', got {res.relevance_status}"
    )
    assert res.primary_topic == expected_topic, (
        f"Expected topic {expected_topic} for '{title}', got {res.primary_topic}"
    )
