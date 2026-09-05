"""
Phase 6: Synthetic Escalation Scenario
=========================================
A scripted, deterministic customer timeline inspired by documented
task/commission scam typology.

IMPORTANT: This scenario is SYNTHETIC. It is:
  - Clearly labeled SYNTHETIC everywhere it appears
  - Never merged into real PaySim evaluation metrics
  - Used to demonstrate the behavioral engine, policy engine, and
    risk decision pipeline working together

The scenario:
  A victim ("C_VICTIM_SYNTH") receives small "commission" payments initially,
  building trust. Then is induced to make increasingly large payments to a
  scam entity ("M_SCAM_SYNTH").

  Step  Amount    Direction        Expected behavior
  100   1,200     OUT to scam      Low risk — small amount, first txn
  101   2,800     OUT to scam      Low risk — still small, repeat counterparty noted
  103   8,000     OUT to scam      Medium — escalation pattern emerging
  106   20,000    OUT to scam      High — significant escalation, multiple rules fire
  110   50,000    OUT to scam      Critical — extreme escalation, all signals active

  Risk tiers are NOT hardcoded. Each transaction flows through:
    feature builder -> ML score -> behavioral engine -> policy engine -> risk fusion
  The increasing risk EMERGES from the pipeline's real computation.
"""

import sys
import os
from dataclasses import dataclass
from typing import List

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.engine.behavioral import BehavioralEngine, BehavioralSignals
from src.engine.policy import PolicyEngine
from src.engine.risk_decision import RiskDecisionEngine, RiskDecision


# ============================================================================
# SYNTHETIC SCENARIO DEFINITION
# ============================================================================

SYNTHETIC_LABEL = "[SYNTHETIC]"
VICTIM_ID = "C_VICTIM_SYNTH"
SCAM_ENTITY_ID = "M_SCAM_SYNTH"

# The scam entity is blocklisted (known bad actor)
SCAM_BLOCKLIST = {SCAM_ENTITY_ID}


@dataclass
class SyntheticTransaction:
    """A single transaction in the synthetic scenario."""
    step: int
    amount: float
    name_orig: str
    name_dest: str
    tx_type: str
    narrative: str  # human-readable description of what's happening


# Task/commission scam progression
SCENARIO_TRANSACTIONS = [
    SyntheticTransaction(
        step=100, amount=1_200, name_orig=VICTIM_ID, name_dest=SCAM_ENTITY_ID,
        tx_type="TRANSFER",
        narrative="Initial small transfer — victim sends first 'task deposit'. "
                  "Scam entity promises commission returns."
    ),
    SyntheticTransaction(
        step=101, amount=2_800, name_orig=VICTIM_ID, name_dest=SCAM_ENTITY_ID,
        tx_type="TRANSFER",
        narrative="Second transfer — slightly larger. Victim received a small "
                  "'commission' and now trusts the scheme."
    ),
    SyntheticTransaction(
        step=103, amount=8_000, name_orig=VICTIM_ID, name_dest=SCAM_ENTITY_ID,
        tx_type="TRANSFER",
        narrative="Escalation begins — victim invests more, expecting larger returns. "
                  "Scam entity claims 'higher tier' requires bigger deposit."
    ),
    SyntheticTransaction(
        step=106, amount=20_000, name_orig=VICTIM_ID, name_dest=SCAM_ENTITY_ID,
        tx_type="TRANSFER",
        narrative="Major escalation — victim is now deeply invested. Scam entity "
                  "claims a 'withdrawal fee' or 'tax' must be paid first."
    ),
    SyntheticTransaction(
        step=110, amount=50_000, name_orig=VICTIM_ID, name_dest=SCAM_ENTITY_ID,
        tx_type="TRANSFER",
        narrative="Critical escalation — victim makes largest payment. Classic "
                  "task/commission scam climax before the scammer disappears."
    ),
]


@dataclass
class ScenarioResult:
    """Result of processing one synthetic transaction through the full pipeline."""
    transaction: SyntheticTransaction
    behavioral_signals: BehavioralSignals
    policy_result: dict
    risk_decision: RiskDecision
    ml_score: float  # placeholder or real score

    def to_dict(self) -> dict:
        return {
            "label": SYNTHETIC_LABEL,
            "step": self.transaction.step,
            "amount": self.transaction.amount,
            "name_orig": self.transaction.name_orig,
            "name_dest": self.transaction.name_dest,
            "tx_type": self.transaction.tx_type,
            "narrative": self.transaction.narrative,
            "ml_score": self.ml_score,
            "behavioral_signals": self.behavioral_signals.to_dict(),
            "policy_result": self.policy_result,
            "risk_tier": self.risk_decision.risk_tier,
            "composite_score": self.risk_decision.composite_score,
            "risk_decision": self.risk_decision.to_dict(),
        }


def run_scenario(
    ml_scorer=None,
    behavioral_engine: BehavioralEngine = None,
    policy_engine: PolicyEngine = None,
    risk_engine: RiskDecisionEngine = None,
    verbose: bool = True,
) -> List[ScenarioResult]:
    """
    Run the synthetic escalation scenario through the full pipeline.

    Args:
        ml_scorer: callable(amount, oldbalanceOrg, ...) -> P(fraud).
                   If None, uses a simple heuristic based on amount.
        behavioral_engine: if None, creates a fresh one
        policy_engine: if None, creates one with scam blocklist
        risk_engine: if None, creates default

    Returns:
        List of ScenarioResult for each transaction.
    """
    # Initialize engines (fresh state for scenario)
    if behavioral_engine is None:
        behavioral_engine = BehavioralEngine()
    else:
        behavioral_engine.reset()

    if policy_engine is None:
        policy_engine = PolicyEngine(blocklist=SCAM_BLOCKLIST)

    if risk_engine is None:
        risk_engine = RiskDecisionEngine()

    results = []

    if verbose:
        print(f"\n{'='*70}")
        print(f"{SYNTHETIC_LABEL} ESCALATION SCENARIO — Task/Commission Scam")
        print(f"{'='*70}")
        print(f"  Victim: {VICTIM_ID}")
        print(f"  Scam Entity: {SCAM_ENTITY_ID} (blocklisted)")
        print(f"  Transactions: {len(SCENARIO_TRANSACTIONS)}")
        print()

    for i, txn in enumerate(SCENARIO_TRANSACTIONS, 1):
        if verbose:
            print(f"\n--- {SYNTHETIC_LABEL} Transaction {i}/{len(SCENARIO_TRANSACTIONS)} ---")
            print(f"  Step: {txn.step} | Amount: {txn.amount:,.2f} | "
                  f"{txn.name_orig} -> {txn.name_dest}")
            print(f"  Narrative: {txn.narrative}")

        # 1. Behavioral signals (computed from history BEFORE this txn)
        signals = behavioral_engine.score(
            step=txn.step,
            amount=txn.amount,
            name_orig=txn.name_orig,
            name_dest=txn.name_dest,
            tx_type=txn.tx_type,
        )

        # 2. ML score
        if ml_scorer is not None:
            ml_score = ml_scorer(txn)
        else:
            # Simple heuristic when no ML model is loaded
            # Scale: small amounts = low score, large = higher
            ml_score = min(txn.amount / 200_000, 0.5)

        # 3. Policy evaluation
        policy_result = policy_engine.evaluate(
            amount=txn.amount,
            name_dest=txn.name_dest,
            tx_type=txn.tx_type,
            behavioral=signals,
        )

        # 4. Risk decision (fusion)
        risk_decision = risk_engine.decide(
            ml_score=ml_score,
            behavioral=signals,
            policy_result=policy_result,
        )

        if verbose:
            print(f"  ML Score: {ml_score:.4f}")
            print(f"  Behavioral: {signals.summary()}")
            print(f"  Policy: {policy_result.summary()}")
            print(f"  >>> RISK: {risk_decision.risk_tier} "
                  f"(composite={risk_decision.composite_score:.3f})")

        result = ScenarioResult(
            transaction=txn,
            behavioral_signals=signals,
            policy_result=policy_result.to_dict(),
            risk_decision=risk_decision,
            ml_score=ml_score,
        )
        results.append(result)

        # 5. Record transaction in behavioral history AFTER scoring
        behavioral_engine.record(
            step=txn.step,
            amount=txn.amount,
            name_orig=txn.name_orig,
            name_dest=txn.name_dest,
            tx_type=txn.tx_type,
        )

    if verbose:
        print(f"\n{'='*70}")
        print(f"{SYNTHETIC_LABEL} SCENARIO SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Step':>6s} {'Amount':>10s} {'ML':>8s} {'Behav':>8s} "
              f"{'Policy':>8s} {'Composite':>10s} {'Tier':>10s} {'Rules'}")
        print(f"  {'-'*6} {'-'*10} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*10} {'-'*20}")

        for r in results:
            rules_str = ", ".join(r.risk_decision.triggered_rules) if r.risk_decision.triggered_rules else "none"
            print(f"  {r.transaction.step:>6d} {r.transaction.amount:>10,.0f} "
                  f"{r.risk_decision.ml_score:>8.4f} "
                  f"{r.risk_decision.behavioral_risk_score:>8.4f} "
                  f"{r.risk_decision.policy_risk_score:>8.4f} "
                  f"{r.risk_decision.composite_score:>10.4f} "
                  f"{r.risk_decision.risk_tier:>10s} {rules_str}")

    return results


if __name__ == "__main__":
    results = run_scenario(verbose=True)
