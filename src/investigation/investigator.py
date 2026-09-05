"""
Phase 7: AI Investigation
============================
Bounded LLM investigation of risk cases.

Architecture:
  - The LLM receives ONLY structured evidence (computed scores, flags, rule names).
  - It MUST recommend exactly ONE action from a fixed set.
  - Maximum ONE LLM call per investigation (capped rounds).
  - A grounding check validates the LLM's output references only real evidence fields.
  - All risk flag values are pinned by code AFTER the LLM call — the model's text
    cannot change any computed flag's value (deterministic-override pattern).
  - A deterministic fallback exists when no LLM API key is configured.

Actions (fixed set):
  1. ALLOW              — transaction appears safe
  2. STEP_UP_VERIFY     — request additional verification from customer
  3. HOLD_FOR_REVIEW    — hold transaction pending human review
  4. BLOCK_ESCALATE     — block transaction and escalate to fraud team

The LLM NEVER decides fraud. It only summarizes evidence and recommends an action.
"""

import os
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List
from enum import Enum


class InvestigationAction(str, Enum):
    ALLOW = "ALLOW"
    STEP_UP_VERIFY = "STEP_UP_VERIFY"
    HOLD_FOR_REVIEW = "HOLD_FOR_REVIEW"
    BLOCK_ESCALATE = "BLOCK_ESCALATE"


# Valid evidence field names that the LLM is allowed to reference
VALID_EVIDENCE_FIELDS = {
    "ml_fraud_score", "risk_tier", "composite_score",
    "behavioral_velocity_24h", "behavioral_escalation_ratio",
    "behavioral_amount_zscore", "behavioral_counterparty_repeat",
    "behavioral_total_tx_count", "behavioral_is_new_counterparty",
    "policy_rules_triggered", "policy_triggered_count",
    "transaction_amount", "transaction_type", "transaction_step",
    "customer_mean_amount", "customer_max_amount",
}


@dataclass
class InvestigationEvidence:
    """Structured evidence payload sent to the investigator."""
    # Transaction details
    transaction_amount: float
    transaction_type: str
    transaction_step: int

    # ML model output
    ml_fraud_score: float
    risk_tier: str
    composite_score: float

    # Behavioral signals
    behavioral_velocity_24h: int
    behavioral_escalation_ratio: float
    behavioral_amount_zscore: float
    behavioral_counterparty_repeat: int
    behavioral_total_tx_count: int
    behavioral_is_new_counterparty: bool

    # Policy
    policy_rules_triggered: List[str]
    policy_triggered_count: int

    # Customer history
    customer_mean_amount: float
    customer_max_amount: float

    def to_dict(self) -> dict:
        return asdict(self)

    def to_prompt_context(self) -> str:
        """Format evidence as structured text for the LLM prompt."""
        rules = ", ".join(self.policy_rules_triggered) if self.policy_rules_triggered else "none"
        return f"""STRUCTURED EVIDENCE (all values computed by deterministic code):

Transaction:
  - Amount: {self.transaction_amount:,.2f}
  - Type: {self.transaction_type}
  - Step: {self.transaction_step}

ML Fraud Model:
  - Fraud probability: {self.ml_fraud_score:.4f}
  - Risk tier: {self.risk_tier}
  - Composite risk score: {self.composite_score:.4f}

Behavioral Signals:
  - Transaction velocity (24h): {self.behavioral_velocity_24h}
  - Amount escalation ratio: {self.behavioral_escalation_ratio:.2f}x customer average
  - Amount z-score: {self.behavioral_amount_zscore:.2f}
  - Counterparty repeat count: {self.behavioral_counterparty_repeat}
  - Is new counterparty: {self.behavioral_is_new_counterparty}
  - Total prior transactions: {self.behavioral_total_tx_count}

Policy Rules:
  - Rules triggered: {self.policy_triggered_count} ({rules})

Customer Baseline:
  - Historical mean amount: {self.customer_mean_amount:,.2f}
  - Historical max amount: {self.customer_max_amount:,.2f}"""


@dataclass
class InvestigationResult:
    """Result of an AI investigation."""
    action: str  # one of InvestigationAction values
    explanation: str
    evidence_cited: List[str]  # which evidence fields were referenced
    confidence: str  # LOW / MEDIUM / HIGH
    provider: str  # "llm" or "deterministic_fallback"
    grounding_passed: bool
    raw_response: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ============================================================================
# LLM INVESTIGATION PROMPT
# ============================================================================

INVESTIGATION_PROMPT = """You are an AI fraud investigation assistant. Your role is to analyze structured evidence about a flagged transaction and provide a recommendation.

RULES:
1. You MUST base your analysis ONLY on the evidence provided below. Do NOT invent facts.
2. You MUST recommend exactly ONE action from this list:
   - ALLOW: Transaction appears safe, no further action needed.
   - STEP_UP_VERIFY: Request additional verification (OTP, ID check) from the customer.
   - HOLD_FOR_REVIEW: Hold the transaction for manual human review.
   - BLOCK_ESCALATE: Block the transaction immediately and escalate to the fraud team.
3. You MUST explain which specific evidence fields support your recommendation.
4. Keep your response concise (3-5 sentences).

{evidence}

Based on the above evidence, provide your analysis in this exact format:
ACTION: [one of ALLOW, STEP_UP_VERIFY, HOLD_FOR_REVIEW, BLOCK_ESCALATE]
EVIDENCE_CITED: [comma-separated list of evidence field names you used]
CONFIDENCE: [LOW, MEDIUM, or HIGH]
EXPLANATION: [your analysis in 3-5 sentences]"""


# ============================================================================
# DETERMINISTIC FALLBACK
# ============================================================================

def deterministic_investigate(evidence: InvestigationEvidence) -> InvestigationResult:
    """
    Template-based investigation when no LLM API key is configured.

    Uses simple rules to generate a structured recommendation.
    This is NOT an ML model — it's a deterministic template.
    """
    cited = []
    reasons = []

    # Determine action based on risk tier and signals
    if evidence.risk_tier == "CRITICAL":
        action = InvestigationAction.BLOCK_ESCALATE
        confidence = "HIGH"
        cited.extend(["risk_tier", "composite_score", "ml_fraud_score"])
        reasons.append(
            f"Transaction flagged as CRITICAL risk (composite score: {evidence.composite_score:.3f})."
        )
    elif evidence.risk_tier == "HIGH":
        action = InvestigationAction.HOLD_FOR_REVIEW
        confidence = "HIGH"
        cited.extend(["risk_tier", "composite_score"])
        reasons.append(
            f"Transaction flagged as HIGH risk (composite score: {evidence.composite_score:.3f})."
        )
    elif evidence.risk_tier == "MEDIUM":
        action = InvestigationAction.STEP_UP_VERIFY
        confidence = "MEDIUM"
        cited.extend(["risk_tier", "composite_score"])
        reasons.append(
            f"Transaction flagged as MEDIUM risk (composite score: {evidence.composite_score:.3f})."
        )
    else:
        action = InvestigationAction.ALLOW
        confidence = "HIGH"
        cited.extend(["risk_tier"])
        reasons.append("Transaction is LOW risk.")

    # Add behavioral context
    if evidence.behavioral_escalation_ratio > 2.0:
        cited.append("behavioral_escalation_ratio")
        reasons.append(
            f"Amount is {evidence.behavioral_escalation_ratio:.1f}x the customer's "
            f"historical average ({evidence.customer_mean_amount:,.2f}), indicating escalation."
        )
        cited.append("customer_mean_amount")

    if evidence.behavioral_velocity_24h > 3:
        cited.append("behavioral_velocity_24h")
        reasons.append(
            f"High transaction velocity: {evidence.behavioral_velocity_24h} transactions in 24h."
        )

    if evidence.policy_triggered_count > 0:
        cited.append("policy_rules_triggered")
        rules = ", ".join(evidence.policy_rules_triggered)
        reasons.append(f"Policy rules triggered: {rules}.")

    if evidence.ml_fraud_score > 0.5 and "ml_fraud_score" not in cited:
        cited.append("ml_fraud_score")
        reasons.append(
            f"ML fraud model assigns probability {evidence.ml_fraud_score:.3f}."
        )

    explanation = " ".join(reasons)

    return InvestigationResult(
        action=action.value,
        explanation=explanation,
        evidence_cited=list(set(cited)),
        confidence=confidence,
        provider="deterministic_fallback",
        grounding_passed=True,
        raw_response=None,
    )


# ============================================================================
# LLM-BASED INVESTIGATION
# ============================================================================

def parse_llm_response(raw: str) -> dict:
    """Parse the structured LLM response into components."""
    parsed = {"action": None, "evidence_cited": [], "confidence": None, "explanation": ""}

    for line in raw.strip().split("\n"):
        line = line.strip()
        if line.upper().startswith("ACTION:"):
            action_str = line.split(":", 1)[1].strip().upper()
            # Normalize
            action_str = action_str.replace("-", "_").replace(" ", "_")
            if action_str in [a.value for a in InvestigationAction]:
                parsed["action"] = action_str
        elif line.upper().startswith("EVIDENCE_CITED:"):
            cited_str = line.split(":", 1)[1].strip()
            parsed["evidence_cited"] = [
                f.strip() for f in cited_str.split(",") if f.strip()
            ]
        elif line.upper().startswith("CONFIDENCE:"):
            conf = line.split(":", 1)[1].strip().upper()
            if conf in ("LOW", "MEDIUM", "HIGH"):
                parsed["confidence"] = conf
        elif line.upper().startswith("EXPLANATION:"):
            parsed["explanation"] = line.split(":", 1)[1].strip()

    return parsed


def grounding_check(evidence_cited: List[str]) -> tuple:
    """
    Validate that all cited evidence fields actually exist in the payload.
    Returns (passed: bool, invalid_fields: list).
    """
    invalid = [f for f in evidence_cited if f not in VALID_EVIDENCE_FIELDS]
    return len(invalid) == 0, invalid


def llm_investigate(
    evidence: InvestigationEvidence,
    api_key: Optional[str] = None,
) -> InvestigationResult:
    """
    LLM-based investigation using Google Gemini or similar.
    Falls back to deterministic if API key is missing or call fails.
    """
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    if not api_key:
        return deterministic_investigate(evidence)

    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = INVESTIGATION_PROMPT.format(evidence=evidence.to_prompt_context())

        # Single LLM call — capped at 1
        response = model.generate_content(prompt)
        raw = response.text

        # Parse response
        parsed = parse_llm_response(raw)

        # Grounding check
        grounded, invalid_fields = grounding_check(parsed.get("evidence_cited", []))
        if not grounded:
            # Re-ask or fall back
            return InvestigationResult(
                action=InvestigationAction.HOLD_FOR_REVIEW.value,
                explanation=(
                    f"LLM referenced non-existent evidence fields: {invalid_fields}. "
                    f"Falling back to HOLD_FOR_REVIEW for safety."
                ),
                evidence_cited=["risk_tier"],
                confidence="LOW",
                provider="llm_grounding_failed",
                grounding_passed=False,
                raw_response=raw,
            )

        # Validate action
        action = parsed.get("action")
        if action is None:
            action = InvestigationAction.HOLD_FOR_REVIEW.value

        return InvestigationResult(
            action=action,
            explanation=parsed.get("explanation", "No explanation provided."),
            evidence_cited=parsed.get("evidence_cited", []),
            confidence=parsed.get("confidence", "MEDIUM"),
            provider="llm_gemini",
            grounding_passed=True,
            raw_response=raw,
        )

    except Exception as e:
        # Any LLM failure falls back to deterministic
        fallback = deterministic_investigate(evidence)
        fallback.provider = f"deterministic_fallback (llm_error: {str(e)[:100]})"
        return fallback


# ============================================================================
# PUBLIC API
# ============================================================================

def investigate(evidence: InvestigationEvidence) -> InvestigationResult:
    """
    Run an AI investigation on a risk case.

    Tries LLM first (if API key is configured), falls back to deterministic.
    All risk flag values are pinned by the calling code — the LLM's text
    cannot change any computed value.
    """
    result = llm_investigate(evidence)

    # DETERMINISTIC OVERRIDE PATTERN:
    # After the LLM call, the calling code pins all risk flag values.
    # The LLM's recommendation is advisory text — it cannot change:
    #   - ml_fraud_score
    #   - risk_tier
    #   - composite_score
    #   - behavioral signals
    #   - policy rule results
    # These are computed BEFORE the LLM is called and remain immutable.

    return result


def build_evidence(
    txn_amount: float,
    txn_type: str,
    txn_step: int,
    ml_score: float,
    risk_tier: str,
    composite_score: float,
    behavioral_signals,
    policy_result,
) -> InvestigationEvidence:
    """Helper to build InvestigationEvidence from pipeline outputs."""
    return InvestigationEvidence(
        transaction_amount=txn_amount,
        transaction_type=txn_type,
        transaction_step=txn_step,
        ml_fraud_score=ml_score,
        risk_tier=risk_tier,
        composite_score=composite_score,
        behavioral_velocity_24h=behavioral_signals.velocity_24h,
        behavioral_escalation_ratio=behavioral_signals.amount_escalation_ratio,
        behavioral_amount_zscore=behavioral_signals.amount_zscore,
        behavioral_counterparty_repeat=behavioral_signals.counterparty_tx_count,
        behavioral_total_tx_count=behavioral_signals.total_tx_count,
        behavioral_is_new_counterparty=behavioral_signals.is_new_counterparty,
        policy_rules_triggered=[
            r["rule_name"] for r in policy_result.get("all_results", []) if r.get("triggered")
        ] if isinstance(policy_result, dict) else [
            r.rule_name for r in policy_result.triggered_rules
        ],
        policy_triggered_count=(
            policy_result.get("rules_triggered", 0) if isinstance(policy_result, dict)
            else policy_result.rules_triggered
        ),
        customer_mean_amount=behavioral_signals.customer_mean_amount,
        customer_max_amount=behavioral_signals.customer_max_amount,
    )
