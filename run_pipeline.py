"""
End-to-End Pipeline Demonstration
====================================
Runs the complete pipeline: synthetic scenario through all layers.

This script demonstrates:
  1. Behavioral engine computing signals from transaction history
  2. Policy engine evaluating deterministic rules
  3. Risk decision engine fusing ML + behavioral + policy into tiers
  4. AI investigation generating bounded explanations
  5. Human review workflow (simulated approve/override)
  6. Append-only audit trail recording everything

The synthetic scenario is the primary demonstration because PaySim cannot
meaningfully exercise the behavioral engine (99.9% single-txn customers).
"""

import sys
import os
import json

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)

from src.engine.behavioral import BehavioralEngine
from src.engine.policy import PolicyEngine
from src.engine.risk_decision import RiskDecisionEngine
from src.scenarios.synthetic import (
    run_scenario, SCENARIO_TRANSACTIONS, SCAM_BLOCKLIST,
    SYNTHETIC_LABEL, VICTIM_ID, SCAM_ENTITY_ID,
)
from src.investigation.investigator import (
    investigate, build_evidence, InvestigationEvidence
)
from src.audit.case_store import CaseStore, AnalystAction
from src.audit.audit_trail import AuditTrail


def main():
    print("=" * 70)
    print("END-TO-END PIPELINE DEMONSTRATION")
    print("Phases 3-8: Behavioral + Policy + Risk + AI Investigation + Review + Audit")
    print("=" * 70)

    # Initialize all components
    behavioral_engine = BehavioralEngine()
    policy_engine = PolicyEngine(blocklist=SCAM_BLOCKLIST)
    risk_engine = RiskDecisionEngine()
    case_store = CaseStore()
    audit_trail = AuditTrail()
    audit_trail.clear()  # fresh log for demo

    # ========================================================================
    # Step 1: Run synthetic scenario through behavioral + policy + risk fusion
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 1: Running synthetic escalation scenario")
    print(f"{'='*70}")

    scenario_results = run_scenario(
        behavioral_engine=behavioral_engine,
        policy_engine=policy_engine,
        risk_engine=risk_engine,
        verbose=True,
    )

    # ========================================================================
    # Step 2: AI Investigation for cases requiring review
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"STEP 2: AI Investigation (bounded, grounded)")
    print(f"{'='*70}")

    for i, result in enumerate(scenario_results, 1):
        txn = result.transaction
        risk = result.risk_decision

        if not risk.requires_review:
            print(f"\n  Txn {i} (Step {txn.step}, {txn.amount:,.0f}): "
                  f"{risk.risk_tier} - AUTO ALLOWED (no case needed)")

            # Audit: record auto-allow
            audit_trail.append(
                event_type="RISK_DECISION_AUTO_ALLOW",
                transaction_step=txn.step,
                transaction_amount=txn.amount,
                transaction_type=txn.tx_type,
                name_orig=txn.name_orig,
                name_dest=txn.name_dest,
                ml_score=result.ml_score,
                risk_tier=risk.risk_tier,
                composite_score=risk.composite_score,
                behavioral_signals=result.behavioral_signals.to_dict(),
                policy_rules_triggered=risk.triggered_rules,
                is_synthetic=True,
            )
            continue

        print(f"\n  --- {SYNTHETIC_LABEL} Txn {i} (Step {txn.step}, "
              f"{txn.amount:,.0f}) -> {risk.risk_tier} ---")

        # Build evidence for investigation
        evidence = build_evidence(
            txn_amount=txn.amount,
            txn_type=txn.tx_type,
            txn_step=txn.step,
            ml_score=result.ml_score,
            risk_tier=risk.risk_tier,
            composite_score=risk.composite_score,
            behavioral_signals=result.behavioral_signals,
            policy_result=result.policy_result,
        )

        # Investigate
        investigation = investigate(evidence)

        print(f"  AI Provider: {investigation.provider}")
        print(f"  AI Action: {investigation.action}")
        print(f"  AI Confidence: {investigation.confidence}")
        print(f"  AI Explanation: {investigation.explanation}")
        print(f"  Evidence Cited: {investigation.evidence_cited}")
        print(f"  Grounding: {'PASSED' if investigation.grounding_passed else 'FAILED'}")

        # ====================================================================
        # Step 3: Create case + Human Review (simulated)
        # ====================================================================
        case = case_store.create_case(
            transaction_step=txn.step,
            transaction_amount=txn.amount,
            transaction_type=txn.tx_type,
            name_orig=txn.name_orig,
            name_dest=txn.name_dest,
            ml_score=result.ml_score,
            risk_tier=risk.risk_tier,
            composite_score=risk.composite_score,
            behavioral_signals=result.behavioral_signals.to_dict(),
            policy_rules_triggered=risk.triggered_rules,
            policy_result=result.policy_result,
            ai_recommendation=investigation.action,
            ai_explanation=investigation.explanation,
            ai_evidence_cited=investigation.evidence_cited,
            ai_provider=investigation.provider,
            is_synthetic=True,
        )

        print(f"\n  Case Created: {case.case_id}")

        # Audit: record case creation
        audit_trail.append(
            event_type="CASE_CREATED",
            transaction_step=txn.step,
            transaction_amount=txn.amount,
            transaction_type=txn.tx_type,
            name_orig=txn.name_orig,
            name_dest=txn.name_dest,
            ml_score=result.ml_score,
            risk_tier=risk.risk_tier,
            composite_score=risk.composite_score,
            behavioral_signals=result.behavioral_signals.to_dict(),
            policy_rules_triggered=risk.triggered_rules,
            ai_recommendation=investigation.action,
            ai_explanation=investigation.explanation,
            case_id=case.case_id,
            is_synthetic=True,
        )

        # Simulate human review
        # For demo: analyst approves MEDIUM/HIGH, overrides to BLOCK for CRITICAL
        if risk.risk_tier == "CRITICAL":
            action = AnalystAction.OVERRIDE_BLOCK.value
            notes = "Analyst: clear escalation scam pattern. Blocking immediately."
        elif risk.risk_tier == "HIGH":
            action = AnalystAction.APPROVE.value
            notes = "Analyst: AI recommendation to hold for review is appropriate."
        else:
            action = AnalystAction.APPROVE.value
            notes = "Analyst: verified, step-up verification sent to customer."

        case = case_store.review_case(case.case_id, action, notes)
        print(f"  Human Review: {action} | Notes: {notes}")
        print(f"  Case Status: {case.status}")

        # Audit: record human review
        audit_trail.append(
            event_type="HUMAN_REVIEW",
            transaction_step=txn.step,
            transaction_amount=txn.amount,
            transaction_type=txn.tx_type,
            name_orig=txn.name_orig,
            name_dest=txn.name_dest,
            ml_score=result.ml_score,
            risk_tier=risk.risk_tier,
            composite_score=risk.composite_score,
            behavioral_signals=result.behavioral_signals.to_dict(),
            policy_rules_triggered=risk.triggered_rules,
            ai_recommendation=investigation.action,
            ai_explanation=investigation.explanation,
            human_decision=action,
            human_notes=notes,
            case_id=case.case_id,
            is_synthetic=True,
        )

    # ========================================================================
    # Step 4: Summary
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"PIPELINE SUMMARY")
    print(f"{'='*70}")

    print(f"\n  Cases created: {len(case_store.cases)}")
    for case in case_store.list_cases():
        print(f"    {case.case_id}: {case.risk_tier} | AI: {case.ai_recommendation} "
              f"| Human: {case.analyst_action} | Status: {case.status}")

    print(f"\n  Audit trail entries: {audit_trail.count()}")
    print(f"  Audit log path: {audit_trail.log_path}")

    # Print example audit record
    records = audit_trail.read_all()
    if records:
        print(f"\n  --- Example Audit Record (first CASE_CREATED) ---")
        for r in records:
            if r["event_type"] == "CASE_CREATED":
                print(json.dumps(r, indent=2, default=str))
                break

    # Print example case
    all_cases = case_store.list_cases()
    if all_cases:
        print(f"\n  --- Example Case (highest risk) ---")
        critical_cases = [c for c in all_cases if c.risk_tier == "CRITICAL"]
        if critical_cases:
            example = critical_cases[0]
        else:
            example = all_cases[0]
        # Print selected fields
        print(f"    Case ID: {example.case_id}")
        print(f"    Risk Tier: {example.risk_tier}")
        print(f"    Amount: {example.transaction_amount:,.2f}")
        print(f"    ML Score: {example.ml_score:.4f}")
        print(f"    Composite: {example.composite_score:.4f}")
        print(f"    Policy Rules: {example.policy_rules_triggered}")
        print(f"    AI Rec: {example.ai_recommendation}")
        print(f"    AI Explanation: {example.ai_explanation[:200]}")
        print(f"    Human: {example.analyst_action}")
        print(f"    Status: {example.status}")
        print(f"    Synthetic: {example.is_synthetic}")


if __name__ == "__main__":
    main()
