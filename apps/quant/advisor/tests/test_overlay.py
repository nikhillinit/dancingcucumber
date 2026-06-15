from advisor.personas.overlay import PersonaVerdict, apply_overlay
from advisor.pipeline.run import Decision


def _decision(action="buy", quantity=100):
    return Decision(ticker="AAPL", action=action, quantity=quantity,
                    bundle_direction="bullish", reasoning="ensemble bullish")


def test_persona_veto_forces_hold():
    out = apply_overlay(_decision(), lambda d: PersonaVerdict(0.0, "distress red flag"))
    assert out.action == "hold"
    assert out.quantity == 0
    assert "distress red flag" in out.reasoning


def test_persona_downgrade_reduces_quantity():
    out = apply_overlay(_decision(quantity=100), lambda d: PersonaVerdict(0.5, "rich valuation"))
    assert out.quantity == 50
    assert out.action == "buy"


def test_persona_cannot_upsize_multiplier_is_clamped():
    # a persona that "wants" 5x is clamped to 1.0 -- the trust boundary
    out = apply_overlay(_decision(quantity=100), lambda d: PersonaVerdict(5.0, "love it"))
    assert out.quantity == 100
    assert out.action == "buy"


def test_persona_explanation_is_appended():
    out = apply_overlay(_decision(), lambda d: PersonaVerdict(1.0, "value+quality concur"))
    assert "value+quality concur" in out.reasoning
