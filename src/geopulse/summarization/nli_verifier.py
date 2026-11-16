"""Local Natural Language Inference (NLI) claim verification engine.

Provides ONNX Runtime-powered inference using compact DeBERTa-v3 models
on CPU, integrated with deterministic safeguards for entities, quantities,
dates, locations, negation, and procurement stages.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

logger = logging.getLogger(__name__)

# Demonym and country alias mappings for geopolitical entity grounding
DEMONYM_MAP: dict[str, str] = {
    "bulgarian": "bulgaria",
    "romanian": "romania",
    "turkish": "turkey",
    "norwegian": "norway",
    "german": "germany",
    "japanese": "japan",
    "ukrainian": "ukraine",
    "canadian": "canada",
    "russian": "russia",
    "chinese": "china",
    "french": "france",
    "british": "britain",
    "american": "united states",
    "finnish": "finland",
    "estonian": "estonia",
    "australian": "australia",
    "dutch": "netherlands",
    "polish": "poland",
    "swedish": "sweden",
    "korean": "korea",
}

COUNTRY_ALIASES: dict[str, str] = {
    "us": "united states",
    "u.s.": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
    "prc": "china",
    "pla": "china",
    "eu": "european union",
    "nato": "nato",
    "pacom": "indo-pacific command",
    "indopacom": "indo-pacific command",
}

# Stopwords and common nouns excluded from proper noun alignment
IGNORED_ENTITIES: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "in",
        "on",
        "at",
        "for",
        "to",
        "of",
        "and",
        "or",
        "by",
        "with",
        "from",
        "under",
        "according",
        "during",
        "while",
        "after",
        "before",
        "over",
        "between",
        "ministers",
        "defense",
        "defence",
        "foreign",
        "military",
        "officials",
        "government",
        "governments",
        "authorities",
        "leaders",
        "representatives",
        "contractors",
        "suppliers",
        "partners",
        "allies",
        "nations",
        "forces",
        "industry",
        "agreement",
        "statement",
        "meeting",
        "decision",
        "policy",
        "framework",
        "program",
        "programme",
        "system",
        "network",
        "operation",
        "drill",
        "exercise",
        "security",
        "intelligence",
        "naval",
        "commercial",
        "merchant",
        "cargo",
        "vessel",
        "vessels",
        "ship",
        "ships",
        "pact",
        "state",
        "department",
        "ministry",
        "command",
        "agency",
        "council",
        "service",
    }
)

NLIClassification = Literal["entailment", "contradiction", "neutral"]


@dataclass(frozen=True)
class AtomicClaim:
    """An individual atomic proposition extracted from a compound claim."""

    text: str
    is_supported: bool
    classification: NLIClassification
    confidence: float
    failure_reason: str | None = None


@dataclass(frozen=True)
class NLIResult:
    """Full diagnostic result of NLI verification for a premise-hypothesis pair."""

    premise: str
    hypothesis: str
    classification: NLIClassification
    probabilities: tuple[float, float, float]  # (contradiction, entailment, neutral)
    is_supported: bool
    confidence: float
    reason: str | None = None
    atomic_results: tuple[AtomicClaim, ...] = ()
    deterministic_passed: bool = True


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for logits array stably."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def _parse_num_unit(raw_str: str) -> tuple[float | None, str]:
    """Extract numeric value and normalized scale unit from a number string."""
    clean = raw_str.strip().lstrip("$£€").strip()
    m = re.match(r"^([\d.,]+)(?:\s*(billion|million|thousand|percent|%|m|bn))?$", clean, re.I)
    if not m:
        return None, ""
    val_str = m.group(1).replace(",", "")
    try:
        val = float(val_str)
    except ValueError:
        return None, ""
    raw_unit = (m.group(2) or "").lower()
    if raw_unit in {"bn", "billion"}:
        norm_unit = "billion"
    elif raw_unit in {"m", "million"}:
        norm_unit = "million"
    elif raw_unit in {"%", "percent"}:
        norm_unit = "percent"
    else:
        norm_unit = raw_unit
    return val, norm_unit


def split_into_atomic_claims(claim_text: str) -> list[str]:
    """Split compound claims into atomic propositions.

    Handles coordinated clauses joined by 'and', 'while', 'as well as', 'after which',
    or semicolons, provided each clause carries substantive predicate content.
    """
    clean = claim_text.strip()
    if not clean:
        return []

    # Semicolon splitting
    if ";" in clean:
        parts = [p.strip() for p in clean.split(";") if len(p.strip()) >= 15]
        if len(parts) >= 2:
            return parts

    # Coordinated clause splitting: "and [verb phrase]" or ", while [clause]"
    # Match ", while " or ", but " or " and "
    clauses = re.split(r",\s*(?:while|whereas|after which|as well as)\s+", clean, flags=re.I)
    if len(clauses) >= 2 and all(len(c.strip()) >= 20 for c in clauses):
        return [c.strip() for c in clauses]

    # Check for " [subject] [verb] ... and [verb] ... "
    # e.g. "Ministers condemned attacks and agreed to share intelligence."
    match_and = re.match(
        r"^(?P<lead>.*?)\s+and\s+(?P<tail>(?:agreed|signed|announced|called|demanded|ordered|approved|deployed|transferred|warned|reaffirmed|stated|noted)\b.*)$",
        clean,
        flags=re.I,
    )
    if match_and:
        lead = match_and.group("lead").strip()
        tail = match_and.group("tail").strip()
        if len(lead) >= 20 and len(tail) >= 20:
            # Reconstruct subject for tail if tail begins directly with verb
            lead_subj = re.match(r"^((?:The\s+)?[A-Z][a-z]+(?:\s+[A-Za-z]+){0,4})\b", lead)
            if lead_subj:
                subj = lead_subj.group(1).strip()
                if not re.match(r"^(?:he|she|it|they|ministers|officials|forces)\b", tail, re.I):
                    tail = f"{subj} {tail}"
            return [lead, tail]

    return [clean]


class LocalNLIVerifier:
    """Local ONNX Runtime inference engine for NLI claim verification on CPU."""

    def __init__(
        self,
        model_dir: str | Path | None = None,
        quantized: bool = True,
        entailment_threshold: float = 0.60,
        contradiction_threshold: float = 0.50,
        threads: int = 4,
    ) -> None:
        if model_dir is None:
            model_dir = Path(".geopulse/models/nli-deberta-v3-small")
        self.model_dir = Path(model_dir)
        self.quantized = quantized
        self.entailment_threshold = entailment_threshold
        self.contradiction_threshold = contradiction_threshold
        self.threads = max(1, threads)

        self._session: Any = None
        self._tokenizer: Any = None
        self._id2label: dict[str, str] = {
            "0": "contradiction",
            "1": "entailment",
            "2": "neutral",
        }
        self._is_loaded: bool = False

    def is_available(self) -> bool:
        """Check whether local model weights and tokenizer exist on disk."""
        if not self.model_dir.exists():
            return False
        model_file = "model_quantized.onnx" if self.quantized else "model.onnx"
        model_path = self.model_dir / "onnx" / model_file
        tok_path = self.model_dir / "tokenizer.json"
        return model_path.is_file() and tok_path.is_file()

    def _ensure_loaded(self) -> None:
        """Lazily load the ONNX Runtime session and tokenizer."""
        if self._is_loaded:
            return

        model_file = "model_quantized.onnx" if self.quantized else "model.onnx"
        model_path = self.model_dir / "onnx" / model_file
        tok_path = self.model_dir / "tokenizer.json"
        cfg_path = self.model_dir / "config.json"

        if not model_path.is_file() or not tok_path.is_file():
            msg = (
                f"NLI model files not found in {self.model_dir}. Expected {model_file} and "
                f"tokenizer.json. Verification fails closed."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        try:
            import onnxruntime as ort
            from tokenizers import Tokenizer
        except ImportError as exc:
            msg = f"onnxruntime or tokenizers package is missing: {exc}. Verification fails closed."
            logger.error(msg)
            raise ImportError(msg) from exc

        # Load Tokenizer
        self._tokenizer = Tokenizer.from_file(str(tok_path))
        self._tokenizer.enable_truncation(max_length=512)
        self._tokenizer.enable_padding(length=128)

        # Load Session
        sess_opts = ort.SessionOptions()
        sess_opts.intra_op_num_threads = self.threads
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._session = ort.InferenceSession(
            str(model_path), sess_opts, providers=["CPUExecutionProvider"]
        )

        # Load label mapping
        if cfg_path.is_file():
            try:
                with open(cfg_path, encoding="utf-8") as fp:
                    cfg = json.load(fp)
                if "id2label" in cfg:
                    self._id2label = {str(k): str(v) for k, v in cfg["id2label"].items()}
            except Exception as exc:
                logger.warning("Could not read config.json id2label: %s", exc)

        self._is_loaded = True
        logger.info("LocalNLIVerifier successfully loaded from %s", self.model_dir)

    def classify_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Run NLI classification on a single premise-hypothesis pair."""
        results = self.classify_batch([(premise, hypothesis)])
        return results[0]

    def classify_batch(self, pairs: list[tuple[str, str]]) -> list[NLIResult]:
        """Run batched NLI classification across multiple pairs for high CPU throughput."""
        if not pairs:
            return []

        self._ensure_loaded()

        # Batch tokenization
        encodings = self._tokenizer.encode_batch(pairs)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)

        feed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        # Check model inputs
        input_names = [inp.name for inp in self._session.get_inputs()]
        if "token_type_ids" in input_names:
            feed["token_type_ids"] = np.array([e.type_ids for e in encodings], dtype=np.int64)

        raw_logits = self._session.run(None, feed)[0]
        probs = softmax(raw_logits)

        results: list[NLIResult] = []
        for idx, (prem, hyp) in enumerate(pairs):
            prob_dist = probs[idx]
            # Order: 0: contradiction, 1: entailment, 2: neutral
            p_contra = float(prob_dist[0])
            p_ent = float(prob_dist[1])
            p_neut = float(prob_dist[2])

            # Calibrate classification
            if p_contra >= self.contradiction_threshold:
                classification: NLIClassification = "contradiction"
                is_supported = False
                confidence = p_contra
                reason = f"NLI Contradiction detected with confidence {p_contra:.3f}"
            elif p_ent >= self.entailment_threshold:
                classification = "entailment"
                is_supported = True
                confidence = p_ent
                reason = f"NLI Entailment supported with confidence {p_ent:.3f}"
            else:
                classification = "neutral"
                is_supported = False
                confidence = p_neut
                reason = (
                    f"NLI Insufficient evidence (Neutral): P(ent)={p_ent:.3f}, P(neut)={p_neut:.3f}"
                )

            results.append(
                NLIResult(
                    premise=prem,
                    hypothesis=hyp,
                    classification=classification,
                    probabilities=(p_contra, p_ent, p_neut),
                    is_supported=is_supported,
                    confidence=confidence,
                    reason=reason,
                )
            )

        return results

    def verify_claim(
        self,
        claim_text: str,
        evidence_span: str,
        full_article_text: str | None = None,
    ) -> NLIResult:
        """Perform comprehensive claim verification combining NLI with deterministic safeguards.

        1. Splits compound claims into atomic propositions.
        2. Evaluates deterministic checks for entities, quantities, dates, negation, and stages.
        3. Executes batch NLI inference across atomic propositions against the evidence span.
        4. Fails closed if any atomic proposition contradicts or lacks entailment.
        """
        # 1. Deterministic Safeguards Pre-Check
        det_passed, det_failure = self._check_deterministic_safeguards(
            claim_text=claim_text,
            evidence_span=evidence_span,
            full_article_text=full_article_text,
        )

        # 2. Atomic Claim Decomposition
        atomic_claims = split_into_atomic_claims(claim_text)
        pairs = [(evidence_span, ac) for ac in atomic_claims]

        # 3. Model Inference
        batch_nli = self.classify_batch(pairs)

        atomic_results: list[AtomicClaim] = []
        overall_supported = det_passed
        overall_classification: NLIClassification = "entailment"
        max_contra = 0.0
        min_ent = 1.0
        reasons: list[str] = []

        if not det_passed and det_failure:
            reasons.append(f"Deterministic Safeguard: {det_failure}")

        for ac_text, nli_res in zip(atomic_claims, batch_nli):
            p_contra, p_ent, p_neut = nli_res.probabilities
            max_contra = max(max_contra, p_contra)
            min_ent = min(min_ent, p_ent)

            ac_supported = nli_res.is_supported and det_passed
            ac_fail = nli_res.reason if not nli_res.is_supported else det_failure

            if nli_res.classification == "contradiction":
                overall_classification = "contradiction"
                overall_supported = False
            elif nli_res.classification == "neutral" and overall_classification != "contradiction":
                overall_classification = "neutral"
                overall_supported = False

            atomic_results.append(
                AtomicClaim(
                    text=ac_text,
                    is_supported=ac_supported,
                    classification=nli_res.classification,
                    confidence=nli_res.confidence,
                    failure_reason=ac_fail,
                )
            )

        if not overall_supported:
            for ar in atomic_results:
                if not ar.is_supported and ar.failure_reason:
                    reasons.append(f"Atomic '{ar.text[:40]}...': {ar.failure_reason}")

        final_reason = (
            "; ".join(reasons) if reasons else "Fully verified by NLI and deterministic safeguards"
        )
        final_confidence = min_ent if overall_supported else max_contra

        return NLIResult(
            premise=evidence_span,
            hypothesis=claim_text,
            classification=overall_classification,
            probabilities=batch_nli[0].probabilities if batch_nli else (0.0, 0.0, 1.0),
            is_supported=overall_supported,
            confidence=final_confidence,
            reason=final_reason,
            atomic_results=tuple(atomic_results),
            deterministic_passed=det_passed,
        )

    def _check_deterministic_safeguards(
        self,
        claim_text: str,
        evidence_span: str,
        full_article_text: str | None = None,
    ) -> tuple[bool, str | None]:
        """Apply deterministic safeguards for entities, quantities, dates, and stages."""
        c_lower = claim_text.lower()
        e_lower = evidence_span.lower()
        full_lower = full_article_text.lower() if full_article_text else e_lower

        # 1. Procurement and Deployment Stage Confusion
        # Approved potential sale vs Completed physical delivery
        is_fms_approval = any(
            w in e_lower
            for w in [
                "approved a potential",
                "state department approved",
                "dsca",
                "notified congress",
            ]
        )
        claims_delivery = any(
            w in c_lower
            for w in [
                "took delivery",
                "physical delivery",
                "delivered to",
                "received delivery",
                "purchased and took",
            ]
        )
        if is_fms_approval and claims_delivery:
            return (
                False,
                "Procurement stage mismatch: approved sale recast as completed delivery",
            )

        # Prime contractor vs Subcontractor confusion
        if "prime contract" in c_lower and "subcontractor" in e_lower:
            # Check if actor claimed as prime is identified as subcontractor in evidence
            pat_sub = (
                r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s+(?:served as|as|was)\s+"
                r"(?:a\s+)?(?:shipyard\s+)?subcontractor\b"
            )
            sub_matches = re.findall(pat_sub, evidence_span)
            for sub in sub_matches:
                if sub.lower() in c_lower and "awarded the" in c_lower:
                    return (
                        False,
                        f"Contract role reversal: subcontractor {sub} claimed as prime awardee",
                    )

        # 2. Number & Quantity Compatibility
        # Find currency, large numbers, and quantities (normalizing plurals e.g. '16s' -> '16')
        num_pat = (
            r"\b(?:\$|£|€)?\d+(?:[.,]\d+)?s?(?:\s*(?:billion|million|thousand|percent|%|m|bn))?\b"
        )

        def _extract_clean_nums(text: str) -> set[str]:
            raw_list = re.findall(num_pat, text)
            clean_set: set[str] = set()
            for item in raw_list:
                cleaned = re.sub(r"(\d+)s\b", r"\1", item)
                clean_set.add(cleaned)
            return clean_set

        c_nums = _extract_clean_nums(c_lower)
        e_nums = _extract_clean_nums(e_lower)
        full_nums = _extract_clean_nums(full_lower)

        # Distinct quantities in claim that are completely absent from source
        unsupported_nums = c_nums - (e_nums | full_nums)
        if unsupported_nums:
            # Check for valid lower-bound assertions (e.g. 'more than 300' supported by '310')
            lower_bound_pat = (
                r"(?:more than|over|at least|in excess of|exceeding)\s+"
                r"((?:\$|£|€)?\d+(?:[.,]\d+)?(?:\s*(?:billion|million|thousand|percent|%|m|bn))?)"
            )
            lb_matches = re.findall(lower_bound_pat, c_lower)

            source_num_objs: list[tuple[float, str, bool]] = []
            for s_raw in e_nums | full_nums:
                v, u = _parse_num_unit(s_raw)
                if v is not None:
                    is_year = 1900 <= v <= 2099 and u == "" and "." not in s_raw
                    source_num_objs.append((v, u, is_year))

            substantive_unsupported: list[str] = []
            for raw in unsupported_nums:
                # Filter out minor harmless numbers (e.g. '2' in 'two ministers')
                if re.match(r"^[1-3]$", raw.strip()):
                    continue
                v_c, u_c = _parse_num_unit(raw)
                if v_c is None:
                    substantive_unsupported.append(raw)
                    continue

                is_lb = any(raw in lb_m for lb_m in lb_matches)
                if is_lb:
                    # Check if source contains M >= v_c with matching unit
                    entailed = False
                    for v_s, u_s, is_year in source_num_objs:
                        if is_year and v_c < 1000:
                            continue
                        if u_s == u_c and v_s >= v_c:
                            entailed = True
                            break
                    if entailed:
                        continue
                substantive_unsupported.append(raw)

            if substantive_unsupported:
                return (
                    False,
                    f"Unsupported numerical figures in claim: {substantive_unsupported}",
                )

        # 3. Negation Polarity
        neg_words = {"not", "no", "never", "refused", "denied", "without"}
        c_neg = any(w in c_lower.split() for w in neg_words)
        e_neg = any(w in e_lower.split() for w in neg_words)
        if c_neg != e_neg:
            active_verbs = {"enter", "signed", "closed", "deploy", "agreed", "deliver"}
            if any(v in c_lower for v in active_verbs) and any(v in e_lower for v in active_verbs):
                return False, "Negation / Polarity conflict between premise and claim"

        # 4. Entity & Proper Noun Grounding (with demonym resolution)
        proper_nouns = set(re.findall(r"\b[A-Z][a-z]{3,}\b", claim_text))
        substantive_entities = {
            pn.lower() for pn in proper_nouns if pn.lower() not in IGNORED_ENTITIES
        }

        # Check if each substantive entity in claim appears in evidence or article
        for ent in substantive_entities:
            # Map alias or demonym
            canonical_ent = COUNTRY_ALIASES.get(ent, DEMONYM_MAP.get(ent, ent))
            # Check presence
            found = (
                ent in full_lower
                or canonical_ent in full_lower
                or any(
                    dem
                    for dem, country in DEMONYM_MAP.items()
                    if country == ent and dem in full_lower
                )
            )
            if not found:
                # If foreign entity is a country or known geopolitical actor, fail closed
                known_foreign = {
                    "russia",
                    "china",
                    "nato",
                    "australia",
                    "france",
                    "britain",
                    "germany",
                    "japan",
                    "turkey",
                    "bulgaria",
                    "romania",
                    "estonia",
                    "finland",
                    "taiwan",
                    "iran",
                    "israel",
                    "houthi",
                    "al jazeera",
                    "reuters",
                }
                if ent in known_foreign or canonical_ent in known_foreign:
                    return (
                        False,
                        f"Foreign entity '{ent}' asserted in claim but absent from source evidence",
                    )

        return True, None
