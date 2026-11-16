"""Unit and integration tests for local NLI verification engine."""

import json
from pathlib import Path

import pytest

from geopulse.summarization.evidence_matcher import (
    SemanticSupportVerdict,
    verify_claim_semantic_support,
)
from geopulse.summarization.nli_verifier import (
    LocalNLIVerifier,
    NLIResult,
    split_into_atomic_claims,
)


def test_split_into_atomic_claims():
    """Verify compound claims decompose into atomic propositions."""
    # 1. Simple atomic claim
    single = "Ministers signed a naval agreement in Sofia."
    assert split_into_atomic_claims(single) == [single]

    # 2. Semicolon-delimited compound claim
    compound_semi = (
        "The ministers condemned attacks on commercial shipping; "
        "they agreed to enhance intelligence sharing."
    )
    parts_semi = split_into_atomic_claims(compound_semi)
    assert len(parts_semi) == 2
    assert "condemned attacks" in parts_semi[0]
    assert "intelligence sharing" in parts_semi[1]

    # 3. Coordinated 'and [verb]' compound claim
    compound_and = (
        "Ministers condemned strikes on shipping and agreed to enhance intelligence sharing."
    )
    parts_and = split_into_atomic_claims(compound_and)
    assert len(parts_and) == 2
    assert "condemned strikes" in parts_and[0]
    assert "intelligence sharing" in parts_and[1]

    # 4. Dependent clause with 'while'
    compound_while = (
        "State Department approved a Patriot sale, while Congress must review notification."
    )
    parts_while = split_into_atomic_claims(compound_while)
    assert len(parts_while) == 2
    assert "approved a Patriot" in parts_while[0]
    assert "Congress must review" in parts_while[1]


def test_nli_verifier_availability():
    """Verify LocalNLIVerifier detects presence of local model weights."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)
    assert verifier.is_available() is True

    # Non-existent model path
    bad_verifier = LocalNLIVerifier(model_dir=Path(".geopulse/models/non_existent"), quantized=True)
    assert bad_verifier.is_available() is False


def test_nli_verifier_single_pair_inference():
    """Verify single-pair NLI inference classifies entailment, contradiction, and neutral."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    # 1. Entailment
    res_ent = verifier.classify_pair(
        premise=(
            "Bulgarian, Romanian and Turkish defence ministers met in Sofia to enhance "
            "intelligence sharing."
        ),
        hypothesis=(
            "Defence ministers from Bulgaria, Romania and Turkey agreed in Sofia to enhance "
            "intelligence sharing."
        ),
    )
    assert isinstance(res_ent, NLIResult)
    assert res_ent.classification == "entailment"
    assert res_ent.is_supported is True
    assert res_ent.probabilities[1] >= 0.60

    # 2. Contradiction
    res_contra = verifier.classify_pair(
        premise=(
            "Bulgarian, Romanian and Turkish defence ministers met in Sofia to sign agreement."
        ),
        hypothesis=(
            "Ministers from Bulgaria, Romania and Turkey met in Constanța to sign agreement."
        ),
    )
    assert res_contra.classification == "contradiction"
    assert res_contra.is_supported is False
    assert res_contra.probabilities[0] >= 0.50

    # 3. Neutral
    res_neut = verifier.classify_pair(
        premise=(
            "Bulgarian, Romanian and Turkish defence ministers met in Sofia to sign agreement."
        ),
        hypothesis="Commercial shipping faced attacks from Houthi naval drones in the Red Sea.",
    )
    assert res_neut.is_supported is False
    assert res_neut.classification in {"neutral", "contradiction"}


def test_nli_batch_inference_consistency():
    """Verify batch inference yields identical predictions and probabilities to single calls."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    pairs = [
        (
            "The State Department approved a potential Foreign Military Sale to the Netherlands.",
            "The U.S. approved a potential arms sale to the Netherlands.",
        ),
        (
            "More than 570 merchant vessels were struck and at least 40 seafarers killed.",
            "Over 1,200 merchant vessels were struck in the attacks.",
        ),
    ]

    single_0 = verifier.classify_pair(pairs[0][0], pairs[0][1])
    single_1 = verifier.classify_pair(pairs[1][0], pairs[1][1])

    batch = verifier.classify_batch(pairs)
    assert len(batch) == 2

    assert batch[0].classification == single_0.classification
    assert batch[1].classification == single_1.classification
    assert abs(batch[0].probabilities[1] - single_0.probabilities[1]) < 0.25
    assert abs(batch[1].probabilities[1] - single_1.probabilities[1]) < 0.25


def test_nli_fails_closed_when_model_missing():
    """Verify that when the NLI model is missing, verification fails closed."""
    verifier = LocalNLIVerifier(model_dir=Path(".geopulse/models/missing_model"), quantized=True)
    with pytest.raises(FileNotFoundError):
        verifier.classify_pair("Some premise", "Some hypothesis")

    # verify_claim_semantic_support wrapper must fail closed and return is_supported=False
    verdict = verify_claim_semantic_support(
        claim_text="Ministers met in Sofia",
        evidence_span="Ministers met in Sofia",
        nli_verifier=verifier,
    )
    assert verdict.is_supported is False
    assert verdict.verdict == "unsupported"
    assert "failed closed" in verdict.reason


def test_deterministic_safeguard_quantity_mismatch_quarantines():
    """Verify numerical mismatch is rejected even if model outputs plausible score."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    premise = (
        "More than 570 merchant vessels were struck and at least 40 seafarers killed since 2026."
    )
    hypothesis = (
        "More than 800 merchant vessels were struck and at least 120 seafarers killed since 2026."
    )

    res = verifier.verify_claim(claim_text=hypothesis, evidence_span=premise)
    assert res.is_supported is False
    assert res.deterministic_passed is False
    assert "Unsupported numerical figures" in str(res.reason)


def test_deterministic_safeguard_procurement_stage_confusion():
    """Verify approved FMS sale notification recast as physical delivery is rejected."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    premise = (
        "The State Department approved a potential Foreign Military Sale to the Netherlands "
        "of Patriot missile systems for $2.8 billion, which now awaits congressional review."
    )
    hypothesis = (
        "The Netherlands purchased and took physical delivery of Patriot missile batteries."
    )

    res = verifier.verify_claim(claim_text=hypothesis, evidence_span=premise)
    assert res.is_supported is False
    assert "Procurement stage mismatch" in str(res.reason)


def test_deterministic_safeguard_foreign_entity_quarantines():
    """Verify introduction of unmentioned foreign actor fails closed."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    premise = "A French Navy frigate and a Canadian patrol vessel transited the Taiwan Strait."
    hypothesis = "A Russian frigate and an Australian patrol vessel transited the Taiwan Strait."

    res = verifier.verify_claim(claim_text=hypothesis, evidence_span=premise)
    assert res.is_supported is False
    assert res.deterministic_passed is False
    assert "Foreign entity" in str(res.reason)


def test_deterministic_safeguard_demonym_mapping_succeeds():
    """Verify demonym mapping (Turkish -> Turkey) resolves cleanly without false rejection."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    premise = "Bulgarian, Romanian and Turkish defence ministers met in Sofia on October 28."
    hypothesis = (
        "Ministers from Bulgaria, Romania and Turkey agreed on maritime intelligence in Sofia."
    )

    res = verifier.verify_claim(claim_text=hypothesis, evidence_span=premise)
    assert res.deterministic_passed is True


def test_verify_claim_semantic_support_integration():
    """Verify verify_claim_semantic_support integrates NLI and records diagnostics."""
    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    claim = (
        "The U.S. State Department approved a potential $2.8 billion Patriot sale to Netherlands."
    )
    evidence = (
        "The State Department approved a potential Foreign Military Sale to the Netherlands "
        "of Patriot missile systems and related equipment for an estimated cost of $2.8 billion."
    )

    # 1. With NLI enabled
    verdict_nli = verify_claim_semantic_support(
        claim_text=claim,
        evidence_span=evidence,
        nli_verifier=verifier,
    )
    assert isinstance(verdict_nli, SemanticSupportVerdict)
    assert verdict_nli.is_supported is True
    assert verdict_nli.nli_classification == "entailment"
    assert verdict_nli.nli_confidence is not None and verdict_nli.nli_confidence >= 0.60

    # 2. Without NLI (fallback deterministic verifier)
    verdict_det = verify_claim_semantic_support(
        claim_text=claim,
        evidence_span=evidence,
        nli_verifier=None,
    )
    assert verdict_det.is_supported is True
    assert verdict_det.nli_classification is None
    assert verdict_det.nli_confidence is None


def test_benchmark_corpus_zero_false_acceptance_rate():
    """Demonstrate that on the 30-item geopolitical benchmark corpus, FAR is strictly 0.0%."""
    import json

    with open("tests/fixtures/nli_benchmark_corpus.json", encoding="utf-8") as f:
        corpus = json.load(f)

    model_dir = Path(".geopulse/models/nli-deberta-v3-small")
    verifier = LocalNLIVerifier(model_dir=model_dir, quantized=True)

    unsupported_items = [item for item in corpus if not item["ground_truth_supported"]]
    assert len(unsupported_items) >= 15

    false_accepts = []
    for item in unsupported_items:
        res = verifier.verify_claim(
            claim_text=item["hypothesis"],
            evidence_span=item["premise"],
        )
        if res.is_supported:
            false_accepts.append(
                {
                    "id": item["id"],
                    "story": item["story"],
                    "category": item["category"],
                    "confidence": res.confidence,
                    "reason": res.reason,
                }
            )

    # The most important metric is preventing unsupported claims from being accepted
    assert len(false_accepts) == 0, f"False accepts detected: {false_accepts}"


def test_production_feature_flag_modes(monkeypatch):
    """Verify the three production feature flag operating modes:
    1. Enabled mode uses local ONNX NLI and records diagnostic metadata.
    2. Disabled mode falls back cleanly to deterministic rules with null NLI metadata.
    3. Engine failure fails closed without silent downgrade or permissive bypass.
    """
    from geopulse.config import Settings
    from geopulse.summarization.service import EventSummarizer

    claim = (
        "Defence ministers from Bulgaria, Romania and Turkey agreed in Sofia "
        "to enhance intelligence sharing."
    )
    evidence = (
        "Bulgarian, Romanian and Turkish defence ministers met in Sofia "
        "on October 28 and agreed to enhance intelligence sharing."
    )

    # Mode 1: Enabled Mode
    settings_enabled = Settings(nli_verification_enabled=True)
    summarizer_enabled = EventSummarizer(settings=settings_enabled)
    assert summarizer_enabled.nli_verifier is not None
    assert isinstance(summarizer_enabled.nli_verifier, LocalNLIVerifier)

    verdict_enabled = verify_claim_semantic_support(
        claim_text=claim,
        evidence_span=evidence,
        nli_verifier=summarizer_enabled.nli_verifier,
    )
    assert verdict_enabled.is_supported is True
    assert verdict_enabled.nli_classification == "entailment"
    assert verdict_enabled.nli_confidence is not None and verdict_enabled.nli_confidence >= 0.60

    # Mode 2: Disabled Mode (Clean deterministic fallback)
    settings_disabled = Settings(nli_verification_enabled=False)
    summarizer_disabled = EventSummarizer(settings=settings_disabled)
    assert summarizer_disabled.nli_verifier is None

    verdict_disabled = verify_claim_semantic_support(
        claim_text=claim,
        evidence_span=evidence,
        nli_verifier=summarizer_disabled.nli_verifier,
    )
    assert verdict_disabled.is_supported is True
    assert verdict_disabled.nli_classification is None
    assert verdict_disabled.nli_confidence is None

    # Mode 3: Engine Failure Mode (Fails closed without silent downgrade)
    corrupted_verifier = LocalNLIVerifier(model_dir=Path("/nonexistent/corrupted/model/path"))
    verdict_failure = verify_claim_semantic_support(
        claim_text=claim,
        evidence_span=evidence,
        nli_verifier=corrupted_verifier,
    )
    assert verdict_failure.is_supported is False
    assert verdict_failure.verdict == "unsupported"
    assert "failed closed" in verdict_failure.reason


def test_expanded_geopolitical_corpus_locked_test_set_zero_far():
    """Verify that on the locked held-out test split (Split B, 60 pairs across 15 distinct events),
    the full pipeline achieves strictly 0.0% False Acceptance Rate (FAR).
    """
    corpus_file = Path("tests/fixtures/geopolitical_nli_corpus.json")
    assert corpus_file.is_file(), "geopolitical_nli_corpus.json must exist"

    with open(corpus_file, encoding="utf-8") as f:
        corpus = json.load(f)

    test_pairs = [p for p in corpus if p["split"] == "test"]
    assert len(test_pairs) == 60, f"Expected exactly 60 test pairs, got {len(test_pairs)}"

    # Check distribution: 16 entailment, 29 contradiction, 15 neutral
    non_supported = [p for p in test_pairs if not p["ground_truth_supported"]]
    supported = [p for p in test_pairs if p["ground_truth_supported"]]
    assert len(non_supported) == 44
    assert len(supported) == 16

    verifier = LocalNLIVerifier(quantized=True)

    false_accepts = []
    for item in non_supported:
        res = verifier.verify_claim(claim_text=item["hypothesis"], evidence_span=item["premise"])
        if res.is_supported:
            false_accepts.append(
                {
                    "id": item["id"],
                    "gold_label": item["ground_truth_label"],
                    "predicted": res.classification,
                    "confidence": res.confidence,
                    "reason": res.reason,
                }
            )

    assert len(false_accepts) == 0, f"False acceptances in locked test set: {false_accepts}"

    # Also verify that supported items pass (FRR = 0%)
    false_rejects = []
    for item in supported:
        res = verifier.verify_claim(claim_text=item["hypothesis"], evidence_span=item["premise"])
        if not res.is_supported:
            false_rejects.append(
                {
                    "id": item["id"],
                    "gold_label": item["ground_truth_label"],
                    "reason": res.reason,
                }
            )

    assert len(false_rejects) == 0, f"False rejections in locked test set: {false_rejects}"
