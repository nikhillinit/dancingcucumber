from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable


@dataclass(frozen=True)
class PersonaVerdict:
    size_multiplier: float  # intent in [0, 1]; clamped on apply -- can only downgrade, never upsize
    explanation: str


# A critic inspects the typed decision and returns a verdict. In v1 this is backed
# by a mocked/canned scorer in tests and (later) an LLM explainer; it never sets size directly.
PersonaCritic = Callable[[Any], PersonaVerdict]


def apply_overlay(decision: Any, critic: PersonaCritic) -> Any:
    """Apply a persona verdict to a typed Decision.

    Trust boundary (spec section 5): the multiplier is clamped to [0, 1], so a persona
    may veto (-> hold) or downgrade size, but can NEVER upsize or touch risk limits.
    Decoupled from Decision's module via dataclasses.replace to avoid a circular import.
    """
    verdict = critic(decision)
    multiplier = max(0.0, min(1.0, float(verdict.size_multiplier)))
    new_quantity = int(decision.quantity * multiplier)
    action = decision.action if new_quantity > 0 else "hold"
    reasoning = f"{decision.reasoning} | persona({multiplier:.0%}): {verdict.explanation}"
    return replace(decision, quantity=new_quantity, action=action, reasoning=reasoning)
