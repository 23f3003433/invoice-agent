
import json

ALLOWED_ACTIONS = {
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
}


def validate_proposals(batch, ai_result):
    """
    Validate AI-generated proposals against the invoice batch.

    ai_result must be a dictionary containing:
    {
        "batchId": "...",
        "proposals": [...]
    }

    This function validates structure and package coverage.
    Evidence and business correctness must also be checked
    against the actual invoice documents.
    """

    if not isinstance(batch, dict):
        raise ValueError("Invoice batch must be an object")

    if not isinstance(ai_result, dict):
        raise ValueError("AI result must be a JSON object")

    batch_id = batch.get("batchId")

    if ai_result.get("batchId") != batch_id:
        raise ValueError("Batch ID mismatch")

    packages = batch.get("packages")

    if not isinstance(packages, list) or not packages:
        raise ValueError("Batch must contain invoice packages")

    proposals = ai_result.get("proposals")

    if not isinstance(proposals, list):
        raise ValueError("Proposals must be a list")

    expected_ids = [p.get("packageId") for p in packages]

    if (
        any(not isinstance(x, str) or not x for x in expected_ids)
        or len(expected_ids) != len(set(expected_ids))
    ):
        raise ValueError("Package IDs must be unique and nonempty")

    if len(proposals) != len(packages):
        raise ValueError("Exactly one proposal is required per package")

    seen_packages = set()
    seen_actions = set()

    for proposal in proposals:
        if not isinstance(proposal, dict):
            raise ValueError("Each proposal must be an object")

        package_id = proposal.get("packageId")
        action_id = proposal.get("actionId")
        action = proposal.get("action")
        facts = proposal.get("facts")
        evidence = proposal.get("evidenceRefs")
        rationale = proposal.get("rationale")

        if package_id not in expected_ids:
            raise ValueError("Unknown package ID")

        if package_id in seen_packages:
            raise ValueError("Duplicate package proposal")

        seen_packages.add(package_id)

        if (
            not isinstance(action_id, str)
            or len(action_id) < 12
            or action_id in seen_actions
        ):
            raise ValueError("Invalid or duplicate action ID")

        seen_actions.add(action_id)

        if action not in ALLOWED_ACTIONS:
            raise ValueError("Unsupported invoice action")

        if not isinstance(facts, dict):
            raise ValueError("Proposal facts must be an object")

        for key in ("vendorName", "invoiceNumber", "currency"):
            if not isinstance(facts.get(key), str) or not facts[key]:
                raise ValueError(f"Missing or invalid fact: {key}")

        amount = facts.get("amountMinor")

        if type(amount) is not int or amount < 0:
            raise ValueError("amountMinor must be a nonnegative integer")

        if (
            not isinstance(evidence, list)
            or len(evidence) != 3
            or any(not isinstance(ref, str) or not ref for ref in evidence)
            or len(set(evidence)) != 3
        ):
            raise ValueError(
                "Exactly three unique evidence references are required"
            )

        if (
            not isinstance(rationale, str)
            or not 60 <= len(rationale) <= 1500
        ):
            raise ValueError("Rationale must contain 60–1500 characters")

        if action not in rationale:
            raise ValueError("Rationale must name the proposed action")

        for ref in evidence:
            if ref not in rationale:
                raise ValueError(
                    "Rationale must cite every evidence reference"
                )

    return ai_result


def parse_ai_json(text):
    """Parse a model response that should contain a JSON object."""

    try:
        result = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("AI response is not valid JSON") from exc

    if not isinstance(result, dict):
        raise ValueError("AI response must be a JSON object")

    return result
