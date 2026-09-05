"""
Phase 3: Behavioral Engine
============================
Computes behavioral risk signals from a customer's transaction history.

Signals:
  1. Velocity — transaction count in recent time windows
  2. Counterparty repetition — how many times this sender→receiver pair has occurred
  3. Amount escalation — ratio of current amount to customer's historical mean/max
  4. Baseline deviation — z-score of current amount vs customer's own distribution

IMPORTANT (PaySim limitation):
  99.9% of PaySim customers have exactly 1 transaction. These signals are
  effectively undefined for the native PaySim population. They are primarily
  demonstrated through the synthetic escalation scenario. This is NOT faked —
  the engine genuinely computes signals from whatever history exists. On PaySim,
  that history is almost always empty, so signals default to zero/neutral.

Design:
  - The engine is stateful: it maintains a transaction history store.
  - For each new transaction, it looks up the customer's past transactions
    and computes signals ONLY from transactions that occurred BEFORE the current one.
  - No future leakage — strictly causal computation.
"""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from collections import defaultdict


@dataclass
class BehavioralSignals:
    """Computed behavioral signals for a single transaction."""

    # Velocity: how many transactions this customer has made recently
    velocity_1h: int = 0        # transactions in last 1 step
    velocity_3h: int = 0        # transactions in last 3 steps
    velocity_24h: int = 0       # transactions in last 24 steps
    total_tx_count: int = 0     # total historical transactions

    # Counterparty repetition
    counterparty_tx_count: int = 0  # times this sender->receiver pair occurred
    is_new_counterparty: bool = True  # first time transacting with this receiver
    unique_counterparties: int = 0   # total unique receivers for this sender

    # Amount escalation
    amount_escalation_ratio: float = 0.0  # current / historical mean
    amount_vs_max: float = 0.0            # current / historical max
    amount_vs_median: float = 0.0         # current / historical median

    # Baseline deviation
    amount_zscore: float = 0.0       # (current - mean) / std
    customer_mean_amount: float = 0.0
    customer_std_amount: float = 0.0
    customer_max_amount: float = 0.0

    # Time-based
    time_since_last_tx: Optional[int] = None  # steps since last transaction

    # Destination-side signals
    dest_incoming_tx_count: int = 0  # how many txns this destination has received

    # Whether enough history exists for meaningful signals
    has_sufficient_history: bool = False

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """Human-readable summary of behavioral signals."""
        parts = []
        if self.total_tx_count > 0:
            parts.append(f"Customer has {self.total_tx_count} prior txn(s)")
        else:
            parts.append("First-time customer (no history)")

        if self.velocity_24h > 1:
            parts.append(f"Velocity: {self.velocity_24h} txns in last 24h")

        if not self.is_new_counterparty:
            parts.append(f"Repeat counterparty ({self.counterparty_tx_count} prior)")

        if self.has_sufficient_history:
            if self.amount_escalation_ratio > 2.0:
                parts.append(f"Amount escalation: {self.amount_escalation_ratio:.1f}x avg")
            if self.amount_zscore > 2.0:
                parts.append(f"Amount z-score: {self.amount_zscore:.1f} (unusual)")

        return "; ".join(parts) if parts else "No behavioral signals"


@dataclass
class TransactionRecord:
    """Minimal transaction record for history tracking."""
    step: int
    amount: float
    dest: str
    tx_type: str


class BehavioralEngine:
    """
    Stateful behavioral engine that computes risk signals from transaction history.

    Usage:
        engine = BehavioralEngine()
        # Process transactions in chronological order
        for txn in transactions:
            signals = engine.score(txn)
            engine.record(txn)  # add to history AFTER scoring
    """

    # Minimum transactions needed for "sufficient history"
    MIN_HISTORY_FOR_SIGNALS = 2

    def __init__(self):
        # Customer history: nameOrig -> list of TransactionRecord
        self.customer_history: Dict[str, List[TransactionRecord]] = defaultdict(list)
        # Counterparty tracking: (nameOrig, nameDest) -> count
        self.counterparty_counts: Dict[tuple, int] = defaultdict(int)
        # Destination incoming counts: nameDest -> count
        self.dest_incoming_counts: Dict[str, int] = defaultdict(int)

    def reset(self):
        """Clear all history. Used between independent scenarios."""
        self.customer_history.clear()
        self.counterparty_counts.clear()
        self.dest_incoming_counts.clear()

    def score(
        self,
        step: int,
        amount: float,
        name_orig: str,
        name_dest: str,
        tx_type: str = "TRANSFER",
    ) -> BehavioralSignals:
        """
        Compute behavioral signals for a transaction.

        MUST be called BEFORE record() for the same transaction to ensure
        no future leakage — we only look at past transactions.
        """
        signals = BehavioralSignals()
        history = self.customer_history.get(name_orig, [])

        # --- Total transaction count ---
        signals.total_tx_count = len(history)
        signals.has_sufficient_history = len(history) >= self.MIN_HISTORY_FOR_SIGNALS

        if len(history) == 0:
            # First transaction — everything defaults to zero/neutral
            signals.is_new_counterparty = True
            signals.dest_incoming_tx_count = self.dest_incoming_counts.get(name_dest, 0)
            return signals

        # --- Velocity ---
        signals.velocity_1h = sum(1 for t in history if step - t.step <= 1)
        signals.velocity_3h = sum(1 for t in history if step - t.step <= 3)
        signals.velocity_24h = sum(1 for t in history if step - t.step <= 24)

        # --- Counterparty repetition ---
        pair_key = (name_orig, name_dest)
        signals.counterparty_tx_count = self.counterparty_counts.get(pair_key, 0)
        signals.is_new_counterparty = signals.counterparty_tx_count == 0
        signals.unique_counterparties = len(
            set(t.dest for t in history)
        )

        # --- Amount statistics from history ---
        amounts = np.array([t.amount for t in history])
        signals.customer_mean_amount = float(np.mean(amounts))
        signals.customer_std_amount = float(np.std(amounts)) if len(amounts) > 1 else 0.0
        signals.customer_max_amount = float(np.max(amounts))

        # --- Amount escalation ---
        if signals.customer_mean_amount > 0:
            signals.amount_escalation_ratio = amount / signals.customer_mean_amount
        if signals.customer_max_amount > 0:
            signals.amount_vs_max = amount / signals.customer_max_amount

        median_amt = float(np.median(amounts))
        if median_amt > 0:
            signals.amount_vs_median = amount / median_amt

        # --- Baseline deviation (z-score) ---
        if signals.customer_std_amount > 0:
            signals.amount_zscore = (
                (amount - signals.customer_mean_amount) / signals.customer_std_amount
            )
        elif signals.customer_mean_amount > 0:
            # Only 1 prior txn, std=0 — use ratio as proxy
            signals.amount_zscore = abs(amount - signals.customer_mean_amount) / signals.customer_mean_amount

        # --- Time since last transaction ---
        last_step = max(t.step for t in history)
        signals.time_since_last_tx = step - last_step

        # --- Destination incoming ---
        signals.dest_incoming_tx_count = self.dest_incoming_counts.get(name_dest, 0)

        return signals

    def record(
        self,
        step: int,
        amount: float,
        name_orig: str,
        name_dest: str,
        tx_type: str = "TRANSFER",
    ):
        """
        Record a transaction in history. Call AFTER score() for the same transaction.
        """
        self.customer_history[name_orig].append(
            TransactionRecord(step=step, amount=amount, dest=name_dest, tx_type=tx_type)
        )
        self.counterparty_counts[(name_orig, name_dest)] += 1
        self.dest_incoming_counts[name_dest] += 1
