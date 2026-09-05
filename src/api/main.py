"""
Phase 9-10: FastAPI Backend — Risk Manager API
=================================================
Serves the frontend and provides all API endpoints.

Endpoints:
  POST /api/score              — Score a single transaction
  GET  /api/cases              — List risk cases
  POST /api/cases/{id}/review  — Human review of a case
  GET  /api/audit              — Audit trail records
  GET  /api/timeline/{cust_id} — Customer risk timeline
  GET  /api/metrics            — Model validation metrics
  POST /api/scenario/synthetic — Run the synthetic escalation scenario
"""

import os
import sys
import json
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_ROOT)

from src.engine.behavioral import BehavioralEngine
from src.engine.policy import PolicyEngine
from src.engine.risk_decision import RiskDecisionEngine
from src.scenarios.synthetic import run_scenario, SYNTHETIC_LABEL
from src.investigation.investigator import investigate, build_evidence
from src.audit.case_store import CaseStore, AnalystAction
from src.audit.audit_trail import AuditTrail

# ============================================================================
# APP STATE — initialized once at startup
# ============================================================================
app = FastAPI(title="AI Risk Manager", version="1.0.0")

# Shared state
behavioral_engine = BehavioralEngine()
policy_engine = PolicyEngine(blocklist=set())
risk_engine = RiskDecisionEngine()
case_store = CaseStore()
audit_trail = AuditTrail()

# Load validation metrics from Phase 1B results
METRICS_PATH = os.path.join(_PROJECT_ROOT, "models", "validation_results.json")
validation_metrics = {}
if os.path.exists(METRICS_PATH):
    with open(METRICS_PATH, "r") as f:
        validation_metrics = json.load(f)

# Store scenario results for timeline queries
scenario_timeline = {}  # customer_id -> list of results


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ScoreRequest(BaseModel):
    step: int
    amount: float
    name_orig: str
    name_dest: str
    tx_type: str = "TRANSFER"
    oldbalanceOrg: float = 0.0
    oldbalanceDest: float = 0.0


class ReviewRequest(BaseModel):
    action: str  # APPROVE, OVERRIDE_ALLOW, OVERRIDE_STEP_UP, OVERRIDE_HOLD, OVERRIDE_BLOCK
    notes: str = ""


# ============================================================================
# STATIC FILES — Frontend
# ============================================================================

frontend_dir = os.path.join(_PROJECT_ROOT, "frontend")
static_dir = os.path.join(frontend_dir, "static")
templates_dir = os.path.join(frontend_dir, "templates")

if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_frontend():
    index_path = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Frontend not built yet. Use /docs for API."}


# ============================================================================
# API: POST /api/score — Score a single transaction
# ============================================================================

@app.post("/api/score")
async def score_transaction(req: ScoreRequest):
    """Score a single transaction through the full risk pipeline."""
    # 1. Behavioral signals
    signals = behavioral_engine.score(
        step=req.step, amount=req.amount,
        name_orig=req.name_orig, name_dest=req.name_dest,
        tx_type=req.tx_type,
    )

    # 2. ML score (heuristic placeholder — real model needs full feature vector)
    ml_score = min(req.amount / 200_000, 0.5)

    # 3. Policy evaluation
    policy_result = policy_engine.evaluate(
        amount=req.amount, name_dest=req.name_dest,
        tx_type=req.tx_type, behavioral=signals,
    )

    # 4. Risk fusion
    risk_decision = risk_engine.decide(
        ml_score=ml_score, behavioral=signals,
        policy_result=policy_result,
    )

    # 5. Investigation (if needed)
    investigation_result = None
    if risk_decision.requires_review:
        evidence = build_evidence(
            txn_amount=req.amount, txn_type=req.tx_type, txn_step=req.step,
            ml_score=ml_score, risk_tier=risk_decision.risk_tier,
            composite_score=risk_decision.composite_score,
            behavioral_signals=signals, policy_result=policy_result,
        )
        inv = investigate(evidence)
        investigation_result = inv.to_dict()

        # Create case
        case = case_store.create_case(
            transaction_step=req.step, transaction_amount=req.amount,
            transaction_type=req.tx_type, name_orig=req.name_orig,
            name_dest=req.name_dest, ml_score=ml_score,
            risk_tier=risk_decision.risk_tier,
            composite_score=risk_decision.composite_score,
            behavioral_signals=signals.to_dict(),
            policy_rules_triggered=[r.rule_name for r in policy_result.triggered_rules],
            policy_result=policy_result.to_dict(),
            ai_recommendation=inv.action, ai_explanation=inv.explanation,
            ai_evidence_cited=inv.evidence_cited, ai_provider=inv.provider,
        )

        # Audit: case created
        audit_trail.append(
            event_type="CASE_CREATED", transaction_step=req.step,
            transaction_amount=req.amount, transaction_type=req.tx_type,
            name_orig=req.name_orig, name_dest=req.name_dest,
            ml_score=ml_score, risk_tier=risk_decision.risk_tier,
            composite_score=risk_decision.composite_score,
            behavioral_signals=signals.to_dict(),
            policy_rules_triggered=[r.rule_name for r in policy_result.triggered_rules],
            ai_recommendation=inv.action, ai_explanation=inv.explanation,
            case_id=case.case_id,
        )
    else:
        # Audit: auto-allow
        audit_trail.append(
            event_type="RISK_DECISION_AUTO_ALLOW",
            transaction_step=req.step, transaction_amount=req.amount,
            transaction_type=req.tx_type, name_orig=req.name_orig,
            name_dest=req.name_dest, ml_score=ml_score,
            risk_tier=risk_decision.risk_tier,
            composite_score=risk_decision.composite_score,
            behavioral_signals=signals.to_dict(),
            policy_rules_triggered=[],
        )

    # 6. Record in behavioral history AFTER scoring
    behavioral_engine.record(
        step=req.step, amount=req.amount,
        name_orig=req.name_orig, name_dest=req.name_dest,
        tx_type=req.tx_type,
    )

    return {
        "ml_score": ml_score,
        "behavioral_signals": signals.to_dict(),
        "policy_result": policy_result.to_dict(),
        "risk_decision": risk_decision.to_dict(),
        "investigation": investigation_result,
    }


# ============================================================================
# API: GET /api/cases — List cases
# ============================================================================

@app.get("/api/cases")
async def list_cases(status: Optional[str] = None, tier: Optional[str] = None):
    """List all risk cases, optionally filtered."""
    cases = case_store.list_cases(status=status, risk_tier=tier)
    return [c.to_dict() for c in cases]


# ============================================================================
# API: POST /api/cases/{id}/review — Human review
# ============================================================================

@app.post("/api/cases/{case_id}/review")
async def review_case(case_id: str, req: ReviewRequest):
    """Record a human analyst's review of a case."""
    case = case_store.get_case(case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Case {case_id} not found")

    valid_actions = [a.value for a in AnalystAction]
    if req.action not in valid_actions:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action: {req.action}. Must be one of {valid_actions}"
        )

    try:
        updated = case_store.review_case(case_id, req.action, req.notes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Audit: record review
    audit_trail.append(
        event_type="HUMAN_REVIEW",
        transaction_step=updated.transaction_step,
        transaction_amount=updated.transaction_amount,
        transaction_type=updated.transaction_type,
        name_orig=updated.name_orig, name_dest=updated.name_dest,
        ml_score=updated.ml_score, risk_tier=updated.risk_tier,
        composite_score=updated.composite_score,
        behavioral_signals=updated.behavioral_signals,
        policy_rules_triggered=updated.policy_rules_triggered,
        ai_recommendation=updated.ai_recommendation,
        ai_explanation=updated.ai_explanation,
        human_decision=req.action, human_notes=req.notes,
        case_id=case_id,
    )

    return updated.to_dict()


# ============================================================================
# API: GET /api/audit — Audit trail
# ============================================================================

@app.get("/api/audit")
async def get_audit():
    """Retrieve all audit trail records."""
    return audit_trail.read_all()


# ============================================================================
# API: GET /api/timeline/{customer_id} — Customer timeline
# ============================================================================

@app.get("/api/timeline/{customer_id}")
async def get_timeline(customer_id: str):
    """Get the risk timeline for a specific customer."""
    if customer_id in scenario_timeline:
        return scenario_timeline[customer_id]
    return []


# ============================================================================
# API: GET /api/metrics — Model validation metrics
# ============================================================================

@app.get("/api/metrics")
async def get_metrics():
    """Return validation metrics from Phase 1B model training."""
    return {
        "label": "Validation Results - PaySim Production Model",
        "note": "All metrics computed on validation set (steps 409-557). Test set UNTOUCHED.",
        "models": validation_metrics,
    }


# ============================================================================
# API: POST /api/scenario/synthetic — Run synthetic escalation
# ============================================================================

@app.post("/api/scenario/synthetic")
async def run_synthetic():
    """Run the synthetic escalation scenario through the full real pipeline."""
    # Reset state for clean run
    behavioral_engine.reset()
    policy_engine.blocklist = set()  # start clean — dynamic blocklisting
    case_store.cases.clear()
    audit_trail.clear()

    # Run scenario
    scenario_results = run_scenario(
        behavioral_engine=behavioral_engine,
        policy_engine=policy_engine,
        risk_engine=risk_engine,
        verbose=False,
    )

    timeline_data = []

    for result in scenario_results:
        txn = result.transaction
        risk = result.risk_decision

        entry = result.to_dict()

        # Run investigation for cases requiring review
        if risk.requires_review:
            evidence = build_evidence(
                txn_amount=txn.amount, txn_type=txn.tx_type, txn_step=txn.step,
                ml_score=result.ml_score, risk_tier=risk.risk_tier,
                composite_score=risk.composite_score,
                behavioral_signals=result.behavioral_signals,
                policy_result=result.policy_result,
            )
            inv = investigate(evidence)
            entry["investigation"] = inv.to_dict()

            # Create case
            case = case_store.create_case(
                transaction_step=txn.step, transaction_amount=txn.amount,
                transaction_type=txn.tx_type, name_orig=txn.name_orig,
                name_dest=txn.name_dest, ml_score=result.ml_score,
                risk_tier=risk.risk_tier, composite_score=risk.composite_score,
                behavioral_signals=result.behavioral_signals.to_dict(),
                policy_rules_triggered=risk.triggered_rules,
                policy_result=result.policy_result,
                ai_recommendation=inv.action, ai_explanation=inv.explanation,
                ai_evidence_cited=inv.evidence_cited, ai_provider=inv.provider,
                is_synthetic=True,
            )
            entry["case_id"] = case.case_id

            # Audit
            audit_trail.append(
                event_type="CASE_CREATED",
                transaction_step=txn.step, transaction_amount=txn.amount,
                transaction_type=txn.tx_type, name_orig=txn.name_orig,
                name_dest=txn.name_dest, ml_score=result.ml_score,
                risk_tier=risk.risk_tier, composite_score=risk.composite_score,
                behavioral_signals=result.behavioral_signals.to_dict(),
                policy_rules_triggered=risk.triggered_rules,
                ai_recommendation=inv.action, ai_explanation=inv.explanation,
                case_id=case.case_id, is_synthetic=True,
            )
        else:
            entry["investigation"] = None
            entry["case_id"] = None
            audit_trail.append(
                event_type="RISK_DECISION_AUTO_ALLOW",
                transaction_step=txn.step, transaction_amount=txn.amount,
                transaction_type=txn.tx_type, name_orig=txn.name_orig,
                name_dest=txn.name_dest, ml_score=result.ml_score,
                risk_tier=risk.risk_tier, composite_score=risk.composite_score,
                behavioral_signals=result.behavioral_signals.to_dict(),
                policy_rules_triggered=[], is_synthetic=True,
            )

        timeline_data.append(entry)

    # Store for timeline queries
    from src.scenarios.synthetic import VICTIM_ID
    scenario_timeline[VICTIM_ID] = timeline_data

    return {
        "label": SYNTHETIC_LABEL,
        "customer_id": VICTIM_ID,
        "transactions": timeline_data,
        "cases_created": len(case_store.cases),
        "audit_entries": audit_trail.count(),
    }


# ============================================================================
# STARTUP
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
