"""
Phase 4: Policy Engine
========================
Deterministic, named policy rules for transaction risk assessment.

Each rule:
  - Has a human-readable name
  - Returns a triggered boolean
  - Provides an explanation with the evidence values used
  - Is configurable via thresholds
  - Is fully auditable

Rules:
  1. VELOCITY_THRESHOLD — too many transactions in a short window
  2. REPEATED_HIGH_VALUE — repeated high-value txns to same counterparty
  3. HIGH_VALUE_NEW_COUNTERPARTY — large amount to a first-time receiver
  4. BLOCKLISTED_ENTITY — counterparty on a blocklist
  5. ABNORMAL_ESCALATION — current amount far exceeds customer's average

PaySim caveat: REPEATED_HIGH_VALUE and VELOCITY_THRESHOLD rarely fire on PaySim
because 99.9% of customers have exactly 1 transaction. These rules are primarily
demonstrated through the synthetic escalation scenario.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Dict, Set, Optional
from src.engine.behavioral import BehavioralSignals


@dataclass
class PolicyResult:
    """Result of a single policy rule evaluation."""
    rule_name: str
    triggered: bool
    explanation: str
    evidence: Dict  # the actual values that were checked

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PolicyEngineResult:
    """Aggregate result from all policy rules."""
    rules_evaluated: int = 0
    rules_triggered: int = 0
    triggered_rules: List[PolicyResult] = field(default_factory=list)
    all_results: List[PolicyResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rules_evaluated": self.rules_evaluated,
            "rules_triggered": self.rules_triggered,
            "triggered_rule_names": [r.rule_name for r in self.triggered_rules],
            "all_results": [r.to_dict() for r in self.all_results],
        }

    def summary(self) -> str:
        if self.rules_triggered == 0:
            return "No policy rules triggered"
        names = [r.rule_name for r in self.triggered_rules]
        return f"{self.rules_triggered} rule(s) triggered: {', '.join(names)}"


class PolicyEngine:
    """
    Deterministic policy engine with configurable named rules.

    All thresholds can be overridden at construction time.
    Rules are evaluated independently — a transaction can trigger multiple rules.
    """

    def __init__(
        self,
        velocity_threshold: int = 5,
        velocity_window_steps: int = 3,
        high_value_amount: float = 50_000,
        repeated_high_value_min_count: int = 2,
        new_counterparty_high_value: float = 100_000,
        escalation_ratio_threshold: float = 3.0,
        blocklist: Optional[Set[str]] = None,
    ):
        self.velocity_threshold = velocity_threshold
        self.velocity_window_steps = velocity_window_steps
        self.high_value_amount = high_value_amount
        self.repeated_high_value_min_count = repeated_high_value_min_count
        self.new_counterparty_high_value = new_counterparty_high_value
        self.escalation_ratio_threshold = escalation_ratio_threshold
        # Blocklist: in production this would come from a database.
        # For PaySim, we use a synthetic/configurable list.
        self.blocklist: Set[str] = blocklist or set()

    def evaluate(
        self,
        amount: float,
        name_dest: str,
        tx_type: str,
        behavioral: BehavioralSignals,
    ) -> PolicyEngineResult:
        """
        Evaluate all policy rules for a transaction.

        Args:
            amount: transaction amount
            name_dest: destination account identifier
            tx_type: transaction type (CASH_OUT, TRANSFER, etc.)
            behavioral: pre-computed behavioral signals
        """
        result = PolicyEngineResult()
        rules = [
            self._rule_velocity_threshold(behavioral),
            self._rule_repeated_high_value(amount, behavioral),
            self._rule_high_value_new_counterparty(amount, behavioral),
            self._rule_blocklisted_entity(name_dest),
            self._rule_abnormal_escalation(amount, behavioral),
        ]

        result.all_results = rules
        result.rules_evaluated = len(rules)
        result.triggered_rules = [r for r in rules if r.triggered]
        result.rules_triggered = len(result.triggered_rules)

        return result

    def _rule_velocity_threshold(self, behavioral: BehavioralSignals) -> PolicyResult:
        """Rule 1: Customer transaction velocity exceeds threshold."""
        # Use the velocity window that matches our configured window
        if self.velocity_window_steps <= 1:
            velocity = behavioral.velocity_1h
        elif self.velocity_window_steps <= 3:
            velocity = behavioral.velocity_3h
        else:
            velocity = behavioral.velocity_24h

        triggered = velocity >= self.velocity_threshold
        return PolicyResult(
            rule_name="VELOCITY_THRESHOLD",
            triggered=triggered,
            explanation=(
                f"Customer made {velocity} transactions in the last "
                f"{self.velocity_window_steps} step(s). "
                f"Threshold: {self.velocity_threshold}."
                + (" EXCEEDED." if triggered else " Within normal range.")
            ),
            evidence={
                "velocity_count": velocity,
                "window_steps": self.velocity_window_steps,
                "threshold": self.velocity_threshold,
            },
        )

    def _rule_repeated_high_value(
        self, amount: float, behavioral: BehavioralSignals
    ) -> PolicyResult:
        """Rule 2: Repeated high-value payments to the same counterparty."""
        is_high_value = amount >= self.high_value_amount
        is_repeat = behavioral.counterparty_tx_count >= self.repeated_high_value_min_count
        triggered = is_high_value and is_repeat

        return PolicyResult(
            rule_name="REPEATED_HIGH_VALUE",
            triggered=triggered,
            explanation=(
                f"Amount: {amount:,.2f} (threshold: {self.high_value_amount:,.2f}). "
                f"Prior txns to this counterparty: {behavioral.counterparty_tx_count} "
                f"(min repeat: {self.repeated_high_value_min_count})."
                + (" HIGH-VALUE REPEAT DETECTED." if triggered else "")
            ),
            evidence={
                "amount": amount,
                "high_value_threshold": self.high_value_amount,
                "counterparty_prior_count": behavioral.counterparty_tx_count,
                "repeat_threshold": self.repeated_high_value_min_count,
                "is_high_value": is_high_value,
                "is_repeat": is_repeat,
            },
        )

    def _rule_high_value_new_counterparty(
        self, amount: float, behavioral: BehavioralSignals
    ) -> PolicyResult:
        """Rule 3: High-value transaction to a never-before-seen counterparty."""
        is_high = amount >= self.new_counterparty_high_value
        is_new = behavioral.is_new_counterparty
        triggered = is_high and is_new

        return PolicyResult(
            rule_name="HIGH_VALUE_NEW_COUNTERPARTY",
            triggered=triggered,
            explanation=(
                f"Amount: {amount:,.2f} (threshold: {self.new_counterparty_high_value:,.2f}). "
                f"New counterparty: {is_new}."
                + (" HIGH-VALUE TO NEW COUNTERPARTY." if triggered else "")
            ),
            evidence={
                "amount": amount,
                "threshold": self.new_counterparty_high_value,
                "is_new_counterparty": is_new,
            },
        )

    def _rule_blocklisted_entity(self, name_dest: str) -> PolicyResult:
        """Rule 4: Counterparty is on the blocklist."""
        triggered = name_dest in self.blocklist

        return PolicyResult(
            rule_name="BLOCKLISTED_ENTITY",
            triggered=triggered,
            explanation=(
                f"Destination: {name_dest}. "
                + (f"BLOCKLISTED ENTITY DETECTED." if triggered
                   else "Not on blocklist.")
            ),
            evidence={
                "destination": name_dest,
                "is_blocklisted": triggered,
                "blocklist_size": len(self.blocklist),
            },
        )

    def _rule_abnormal_escalation(
        self, amount: float, behavioral: BehavioralSignals
    ) -> PolicyResult:
        """Rule 5: Transaction amount far exceeds customer's historical average."""
        ratio = behavioral.amount_escalation_ratio
        has_history = behavioral.has_sufficient_history
        triggered = has_history and ratio >= self.escalation_ratio_threshold

        return PolicyResult(
            rule_name="ABNORMAL_ESCALATION",
            triggered=triggered,
            explanation=(
                f"Amount: {amount:,.2f}. "
                + (f"Customer avg: {behavioral.customer_mean_amount:,.2f}. "
                   f"Escalation ratio: {ratio:.2f}x "
                   f"(threshold: {self.escalation_ratio_threshold:.1f}x)."
                   if has_history
                   else f"Insufficient history ({behavioral.total_tx_count} txns). ")
                + (" ABNORMAL ESCALATION DETECTED." if triggered else "")
            ),
            evidence={
                "amount": amount,
                "customer_mean": behavioral.customer_mean_amount,
                "escalation_ratio": ratio,
                "threshold": self.escalation_ratio_threshold,
                "has_sufficient_history": has_history,
                "prior_tx_count": behavioral.total_tx_count,
            },
        )
