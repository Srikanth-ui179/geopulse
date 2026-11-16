"""Verification tests confirming that hardcoded story overrides are absent.

Audit Decision 2 mandated the complete purging of story-specific hardcoded
claim substitutions, overwrites, and hardcoded headlines ("ritek", "levanger",
"570 vessels", "offset authority", "bulgaria/romania/turkey" headline injection).
"""

from pathlib import Path


def test_service_py_contains_no_hardcoded_story_overrides():
    """Verify service.py contains no hardcoded claim text or headline overrides."""
    service_path = Path("src/geopulse/summarization/service.py")
    assert service_path.exists(), "src/geopulse/summarization/service.py must exist"

    content = service_path.read_text(encoding="utf-8").lower()

    # Story-specific claim substitutions that were purged
    forbidden_claim_snippets = [
        "assembly of 37 of norway's 54 ordered leopard",
        "annual capacity of up to 36 tanks",
        "broader bilateral hansa defence cooperation framework",
        "polar satellite constellation from andøya",
        "more than 570 vessels have been hit in the black sea",
        "following an oct. 28 meeting in sofia",
        "offset authority, requiring foreign contractors",
        "embedded maven smart system for planning and targeting joint strikes",
        "bulgaria, romania and turkey agree on black sea maritime intelligence sharing",
        "bulgaria, romania and turkey sign black sea naval agreement",
    ]

    for snippet in forbidden_claim_snippets:
        assert snippet not in content, (
            f"Found forbidden hardcoded story override in service.py: '{snippet}'"
        )


def test_prompts_py_contains_no_hardcoded_story_placeholders():
    """Verify that prompts.py uses generalized guidance rather than story-specific examples."""
    prompts_path = Path("src/geopulse/summarization/prompts.py")
    assert prompts_path.exists(), "src/geopulse/summarization/prompts.py must exist"

    content = prompts_path.read_text(encoding="utf-8").lower()

    # Story-specific phrases that were generalized
    forbidden_prompt_snippets = [
        "oct. 28 meeting in sofia",
        "leopard 2a8 tanks began on oct. 15 at levanger",
        "up to 36 tanks annually",
        "offset authority (april 2027)",
        "more than 570 vessels were hit and at least 40 seafarers killed",
        "polar satellite constellation from andøya",
        "japanese partner forces using the embedded maven smart system",
    ]

    for snippet in forbidden_prompt_snippets:
        assert snippet not in content, (
            f"Found forbidden hardcoded story phrase in prompts.py: '{snippet}'"
        )
