"""
Phase 8: Audit Trail — Append-Only Log
=========================================
Append-only audit trail recording every risk decision.

Each record contains:
  - Transaction data
  - ML model score
  - Behavioral signals
  - Policy rules triggered
  - Final risk tier
  - AI recommendation
  - Human decision (if reviewed)
  - Timestamp

Storage: JSONL format (one JSON object per line) — simple, inspectable,
append-only. In production, this would be an immutable log store.
"""

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, List


class AuditTrail:
    """
    Append-only audit log stored as JSONL.

    Each entry is a complete snapshot of the risk decision at a point in time.
    Entries are never modified or deleted — new entries are appended for updates.
    """

    def __init__(self, log_path: str = None):
        if log_path is None:
            project_root = os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            )
            log_path = os.path.join(project_root, "data", "audit_log.jsonl")

        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def append(
        self,
        event_type: str,
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
        ai_recommendation: str = None,
        ai_explanation: str = None,
        human_decision: str = None,
        human_notes: str = None,
        case_id: str = None,
        is_synthetic: bool = False,
        extra: dict = None,
    ):
        """
        Append an audit record.

        Args:
            event_type: RISK_DECISION, CASE_CREATED, HUMAN_REVIEW, etc.
            ... all risk decision components
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "case_id": case_id,
            "is_synthetic": is_synthetic,
            "transaction": {
                "step": transaction_step,
                "amount": transaction_amount,
                "type": transaction_type,
                "name_orig": name_orig,
                "name_dest": name_dest,
            },
            "ml_score": ml_score,
            "risk_tier": risk_tier,
            "composite_score": composite_score,
            "behavioral_signals": behavioral_signals,
            "policy_rules_triggered": policy_rules_triggered,
            "ai_recommendation": ai_recommendation,
            "ai_explanation": ai_explanation,
            "human_decision": human_decision,
            "human_notes": human_notes,
        }

        if extra:
            record["extra"] = extra

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_all(self) -> List[dict]:
        """Read all audit records (for display/export)."""
        records = []
        if os.path.exists(self.log_path):
            with open(self.log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
        return records

    def read_by_case(self, case_id: str) -> List[dict]:
        """Read all records for a specific case."""
        return [r for r in self.read_all() if r.get("case_id") == case_id]

    def count(self) -> int:
        """Count total audit records."""
        if not os.path.exists(self.log_path):
            return 0
        with open(self.log_path, "r") as f:
            return sum(1 for line in f if line.strip())

    def clear(self):
        """Clear the audit log (for testing only — never in production)."""
        if os.path.exists(self.log_path):
            os.remove(self.log_path)
