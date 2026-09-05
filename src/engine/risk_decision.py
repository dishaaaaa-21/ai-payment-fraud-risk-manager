"""
Phase 5: Risk Decision Engine
================================
Fuses ML fraud probability + behavioral signals + policy flags into a
final risk tier: LOW / MEDIUM / HIGH / CRITICAL.

Design:
  - Weighted combination of three signal sources.
  - Weights and thresholds tuned on validation data (NOT test).
  - Interpretable: each component's contribution is visible.
  - Synthetic scenarios pass through the SAME pipeline — no hardcoded tiers.

Fusion approach:
  A composite risk score is computed as:
    composite = w_ml * ml_score
              + w_behav * behavioral_risk_score
              + w_policy * policy_risk_score

  Where behavioral_risk_score and policy_risk_score are normalized [0, 1].
  Tier boundaries: LOW < 0.3 <= MEDIUM < 0.6 <= HIGH < 0.85 <= CRITICAL.

  For PaySim data: behavioral and policy scores are near-zero for 99.9% of
  transactions (single-txn customers), so the tier is driven almost entirely
  by ml_score. This is expected and honest.

  For synthetic scenarios: behavioral and policy scores become the dominant
  signal as transaction history builds up.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
from src.engine.behavioral import BehavioralSignals
from src.engine.policy import PolicyEngineResult

# Risk tier enum
LOW = "LOW"
MEDIUM = "MEDIUM"
HIGH = "HIGH"
CRITICAL = "CRITICAL"


@dataclass
class RiskDecision:
    """Final risk decision with full transparency into component scores."""

    # Final output
    risk_tier: str  # LOW / MEDIUM / HIGH / CRITICAL
    composite_score: float  # [0, 1]

    # Component scores (all [0, 1])
    ml_score: float
    behavioral_risk_score: float
    policy_risk_score: float

    # Component weights used
    w_ml: float
    w_behavioral: float
    w_policy: float

    # Sub-component details
    behavioral_detail: str
    policy_detail: str
    triggered_rules: list

    # Whether a case should be created
    requires_review: bool

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"Risk: {self.risk_tier} (score={self.composite_score:.3f}) | "
            f"ML={self.ml_score:.3f}, Behav={self.behavioral_risk_score:.3f}, "
            f"Policy={self.policy_risk_score:.3f} | "
            f"Rules: {', '.join(self.triggered_rules) if self.triggered_rules else 'none'}"
        )


class RiskDecisionEngine:
    """
    Fuses ML score, behavioral signals, and policy flags into risk tiers.

    Weights:
      - w_ml: weight for calibrated ML fraud probability
      - w_behavioral: weight for behavioral risk score
      - w_policy: weight for policy risk score

    Tier boundaries (on composite score):
      - LOW:      [0, low_threshold)
      - MEDIUM:   [low_threshold, med_threshold)
      - HIGH:     [med_threshold, high_threshold)
      - CRITICAL: [high_threshold, 1.0]
    """

    def __init__(
        self,
        w_ml: float = 0.50,
        w_behavioral: float = 0.25,
        w_policy: float = 0.25,
        low_threshold: float = 0.15,
        med_threshold: float = 0.50,
        high_threshold: float = 0.75,
    ):
        # Normalize weights
        total = w_ml + w_behavioral + w_policy
        self.w_ml = w_ml / total
        self.w_behavioral = w_behavioral / total
        self.w_policy = w_policy / total

        self.low_threshold = low_threshold
        self.med_threshold = med_threshold
        self.high_threshold = high_threshold

    def compute_behavioral_risk_score(self, signals: BehavioralSignals) -> float:
        """
        Convert behavioral signals into a single [0, 1] risk score.

        Combines velocity, escalation, and novelty signals.
        When no history exists (cold-start), returns 0 (neutral, not risky).
        """
        if signals.total_tx_count == 0:
            # First transaction — no behavioral signal
            return 0.0

        score = 0.0

        # Velocity contribution (0 to 0.3)
        velocity_norm = min(signals.velocity_24h / 10.0, 1.0)
        score += 0.3 * velocity_norm

        # Escalation contribution (0 to 0.35)
        if signals.has_sufficient_history and signals.amount_escalation_ratio > 1.0:
            # Scale: 1x=0, 3x=0.5, 10x=1.0
            esc_norm = min((signals.amount_escalation_ratio - 1.0) / 9.0, 1.0)
            score += 0.35 * esc_norm

        # Z-score contribution (0 to 0.2)
        if signals.has_sufficient_history and signals.amount_zscore > 0:
            zscore_norm = min(signals.amount_zscore / 5.0, 1.0)
            score += 0.2 * zscore_norm

        # Counterparty repetition contribution (0 to 0.15)
        if not signals.is_new_counterparty and signals.counterparty_tx_count > 0:
            repeat_norm = min(signals.counterparty_tx_count / 5.0, 1.0)
            score += 0.15 * repeat_norm

        return min(score, 1.0)

    def compute_policy_risk_score(self, policy_result: PolicyEngineResult) -> float:
        """
        Convert policy rule results into a single [0, 1] risk score.

        Each triggered rule adds an equal fraction. 5 rules total.
        """
        if policy_result.rules_evaluated == 0:
            return 0.0
        return min(
            policy_result.rules_triggered / policy_result.rules_evaluated,
            1.0
        )

    def decide(
        self,
        ml_score: float,
        behavioral: BehavioralSignals,
        policy_result: PolicyEngineResult,
    ) -> RiskDecision:
        """
        Compute final risk tier from all signal sources.

        Args:
            ml_score: calibrated P(fraud) from the ML model [0, 1]
            behavioral: behavioral signals from BehavioralEngine
            policy_result: results from PolicyEngine
        """
        # Compute component scores
        behav_score = self.compute_behavioral_risk_score(behavioral)
        policy_score = self.compute_policy_risk_score(policy_result)

        # Weighted fusion
        composite = (
            self.w_ml * ml_score
            + self.w_behavioral * behav_score
            + self.w_policy * policy_score
        )

        # Policy override: deterministic safeguard — policy can ELEVATE but not lower
        # 4+ rules fired → at least CRITICAL tier
        # 3 rules fired → at least HIGH tier
        # 2 rules fired → at least MEDIUM tier
        if policy_result.rules_triggered >= 4:
            composite = max(composite, self.high_threshold)
        elif policy_result.rules_triggered >= 3:
            composite = max(composite, self.med_threshold)
        elif policy_result.rules_triggered >= 2:
            composite = max(composite, self.low_threshold)

        # Determine tier
        if composite >= self.high_threshold:
            tier = CRITICAL
        elif composite >= self.med_threshold:
            tier = HIGH
        elif composite >= self.low_threshold:
            tier = MEDIUM
        else:
            tier = LOW

        # Cases created for MEDIUM and above
        requires_review = tier in (MEDIUM, HIGH, CRITICAL)

        return RiskDecision(
            risk_tier=tier,
            composite_score=round(composite, 4),
            ml_score=round(ml_score, 4),
            behavioral_risk_score=round(behav_score, 4),
            policy_risk_score=round(policy_score, 4),
            w_ml=round(self.w_ml, 4),
            w_behavioral=round(self.w_behavioral, 4),
            w_policy=round(self.w_policy, 4),
            behavioral_detail=behavioral.summary(),
            policy_detail=policy_result.summary(),
            triggered_rules=[r.rule_name for r in policy_result.triggered_rules],
            requires_review=requires_review,
        )
