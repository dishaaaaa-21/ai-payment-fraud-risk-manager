"""
Phase 8: Case Store — Human Review
=====================================
Manages risk cases that require human analyst review.

Cases are created for MEDIUM / HIGH / CRITICAL transactions.
Analysts can approve or override the AI recommendation.
All actions are recorded in the audit trail.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List
from enum import Enum


class CaseStatus(str, Enum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    APPROVED = "APPROVED"       # analyst agreed with AI recommendation
    OVERRIDDEN = "OVERRIDDEN"   # analyst changed the recommendation
    CLOSED = "CLOSED"


class AnalystAction(str, Enum):
    APPROVE = "APPROVE"         # agree with AI recommendation
    OVERRIDE_ALLOW = "OVERRIDE_ALLOW"         # analyst says: allow it
    OVERRIDE_STEP_UP = "OVERRIDE_STEP_UP"     # analyst says: step-up verify
    OVERRIDE_HOLD = "OVERRIDE_HOLD"           # analyst says: hold for further review
    OVERRIDE_BLOCK = "OVERRIDE_BLOCK"         # analyst says: block/escalate


@dataclass
class RiskCase:
    """A risk case requiring human review."""
    case_id: str
    created_at: str

    # Transaction details
    transaction_step: int
    transaction_amount: float
    transaction_type: str
    name_orig: str
    name_dest: str

    # Risk assessment
    ml_score: float
    risk_tier: str
    composite_score: float
    behavioral_signals: dict
    policy_rules_triggered: List[str]
    policy_result: dict

    # AI investigation
    ai_recommendation: str
    ai_explanation: str
    ai_evidence_cited: List[str]
    ai_provider: str

    # Human review
    status: str = CaseStatus.OPEN.value
    analyst_action: Optional[str] = None
    analyst_notes: Optional[str] = None
    reviewed_at: Optional[str] = None

    # Synthetic label
    is_synthetic: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


class CaseStore:
    """
    In-memory case store for risk cases.

    In production, this would be backed by a database.
    For the MVP, cases are stored in memory and can be exported.
    """

    def __init__(self):
        self.cases: Dict[str, RiskCase] = {}

    def create_case(
        self,
        transaction_step: int,
        transaction_amount: float,
        transaction_type: str,
        name_orig: str,
        name_dest: str,
        ml_score: float,
        risk_tier: str,
        composite_score: float,
        behavioral_signals: dict,
        policy_rules_triggered: list,
        policy_result: dict,
        ai_recommendation: str,
        ai_explanation: str,
        ai_evidence_cited: list,
        ai_provider: str,
        is_synthetic: bool = False,
    ) -> RiskCase:
        """Create a new risk case."""
        case_id = f"CASE-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now(timezone.utc).isoformat()

        case = RiskCase(
            case_id=case_id,
            created_at=now,
            transaction_step=transaction_step,
            transaction_amount=transaction_amount,
            transaction_type=transaction_type,
            name_orig=name_orig,
            name_dest=name_dest,
            ml_score=ml_score,
            risk_tier=risk_tier,
            composite_score=composite_score,
            behavioral_signals=behavioral_signals,
            policy_rules_triggered=policy_rules_triggered,
            policy_result=policy_result,
            ai_recommendation=ai_recommendation,
            ai_explanation=ai_explanation,
            ai_evidence_cited=ai_evidence_cited,
            ai_provider=ai_provider,
            is_synthetic=is_synthetic,
        )

        self.cases[case_id] = case
        return case

    def review_case(
        self,
        case_id: str,
        analyst_action: str,
        analyst_notes: str = "",
    ) -> RiskCase:
        """Record an analyst's review of a case."""
        if case_id not in self.cases:
            raise ValueError(f"Case {case_id} not found")

        case = self.cases[case_id]

        if case.status not in (CaseStatus.OPEN.value, CaseStatus.UNDER_REVIEW.value):
            raise ValueError(f"Case {case_id} is already {case.status}")

        case.analyst_action = analyst_action
        case.analyst_notes = analyst_notes
        case.reviewed_at = datetime.now(timezone.utc).isoformat()

        if analyst_action == AnalystAction.APPROVE.value:
            case.status = CaseStatus.APPROVED.value
        else:
            case.status = CaseStatus.OVERRIDDEN.value

        return case

    def get_case(self, case_id: str) -> Optional[RiskCase]:
        return self.cases.get(case_id)

    def list_cases(
        self,
        status: Optional[str] = None,
        risk_tier: Optional[str] = None,
    ) -> List[RiskCase]:
        """List cases, optionally filtered."""
        cases = list(self.cases.values())
        if status:
            cases = [c for c in cases if c.status == status]
        if risk_tier:
            cases = [c for c in cases if c.risk_tier == risk_tier]
        return sorted(cases, key=lambda c: c.created_at, reverse=True)

    def get_open_cases(self) -> List[RiskCase]:
        return self.list_cases(status=CaseStatus.OPEN.value)
