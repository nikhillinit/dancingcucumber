from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]

RETIRED_PATHS = {
    "production_ready_system.py",
    "robust_trading_system.py",
    "automated_trading_system.py",
    "enhanced_training_system.py",
    "personalized_portfolio_system.py",
    "single_user_ai_system.py",
    "apps/quant/finrl_trading_agent.py",
    "apps/quant/qlib_factor_generator.py",
    "apps/quant/autogluon_ensemble.py",
    "AIHedgeFund/apps/quant/finrl_trading_agent.py",
    "AIHedgeFund/apps/quant/qlib_factor_generator.py",
    "AIHedgeFund/apps/quant/autogluon_ensemble.py",
    "AIHedgeFund/apps/quant/ai_debate_orchestrator.py",
    "AIHedgeFund/apps/quant/consensus_engine.py",
    "COMPLETE_MASTER_ANALYSIS.py",
    "FINAL_RECOMMENDATIONS.py",
    "accuracy_enhancement_plan.py",
    "accuracy_improvement_guide.py",
    "advanced_model_training.py",
    "api_config.py",
    "api_setup_guide.py",
    "bt_integration.py",
    "cash_reconciliation.py",
    "complete_unified_analysis.py",
    "congressional_tracker_real.py",
    "congressional_trading_tracker.py",
    "current_events_integration.py",
    "deployment_summary.py",
    "earnings_call_analyzer.py",
    "enhanced_evaluation_system.py",
    "enhanced_fidelity_backtest.py",
    "enhanced_free_data_system.py",
    "enhanced_production_system.py",
    "event_gated_backtest.py",
    "external_intelligence_coordinator.py",
    "external_intelligence_coordinator_backup.py",
    "external_intelligence_summary.py",
    "external_intelligence_system.py",
    "fed_speech_analyzer.py",
    "fidelity_automated_trading.py",
    "fidelity_execution_guide.py",
    "final_92_percent_system.py",
    "final_system_integration_test.py",
    "final_system_summary.py",
    "finrl_demo.py",
    "finrl_integration.py",
    "free_data_quick_start.py",
    "free_data_sources.py",
    "historical_data_optimizer.py",
    "historical_enhancement_system.py",
    "immediate_deployment_guide.py",
    "immediate_improvements.py",
    "immediate_yahoo_system.py",
    "index_fund_analysis.py",
    "insider_trading_analyzer.py",
    "instant_recommendation.py",
    "limited_budget_analysis.py",
    "live_trading_system.py",
    "master_trading_system.py",
    "minimal_viable_system.py",
    "multi_agent_personas.py",
    "next_generation_ml_system.py",
    "optimized_ensemble_system.py",
    "options_flow_integration_demo.py",
    "options_flow_tracker.py",
    "paper_trade_now.py",
    "production_ai_system.py",
    "production_daily_optimizer.py",
    "quick_enhanced_test.py",
    "quick_finrl_validation.py",
    "quick_live_test.py",
    "quick_start.py",
    "quick_unified_analysis.py",
    "real_data_integration.py",
    "real_portfolio_analyzer.py",
    "real_time_update_check.py",
    "regime_trainer.py",
    "run_alpha_validation.py",
    "run_analysis.py",
    "run_master_system.py",
    "sec_edgar_monitor.py",
    "signal_orthogonalization_system.py",
    "simple_alpha_test.py",
    "simplified_ensemble_strategy.py",
    "start_paper_trading.py",
    "statistical_validation_suite.py",
    "stefan_jansen_integration.py",
    "system_validation_summary.py",
    "test_ai_hedge_fund.py",
    "test_bt_accuracy.py",
    "test_bt_accuracy_simple.py",
    "test_data_simple.py",
    "test_finrl_lightweight.py",
    "test_fred_connection.py",
    "test_free_data_system.py",
    "test_optimized_ensemble.py",
    "test_options_flow_system.py",
    "test_simple_backtest.py",
    "test_standalone_backtest.py",
    "test_yahoo_basic.py",
    "test_yfinance.py",
    "three_persona_consensus.py",
    "timestamp_integrity_system.py",
    "training_improvement_guide.py",
    "ultimate_hedge_fund_system.py",
    "unified_data_system.py",
    "unified_intelligence_system.py",
    "updated_portfolio_analysis.py",
    "validate_alpha_claims.py",
    "virattt_integration_analysis.py",
    "walk_forward_validator.py",
    "yahoo_finance_direct.py",
    "zero_cost_optimizer.py",
}

RETIRED_MODULES = {
    Path(path).stem
    for path in RETIRED_PATHS
}


def _tracked_python_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        REPO_ROOT / line.strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]


IMPORT_RE = re.compile(r"^\s*import\s+(?P<modules>.+)$")
FROM_RE = re.compile(r"^\s*from\s+(?P<module>[\.\w]+)\s+import\s+")


def _imported_roots(source: str) -> set[str]:
    imported: set[str] = set()
    for line in source.splitlines():
        import_match = IMPORT_RE.match(line)
        if import_match:
            imported.update(
                module.strip().split()[0].split(".", maxsplit=1)[0]
                for module in import_match.group("modules").split(",")
            )
            continue

        from_match = FROM_RE.match(line)
        if from_match:
            imported.add(from_match.group("module").lstrip(".").split(".", maxsplit=1)[0])

    return imported


def test_retired_cleanup_targets_are_absent() -> None:
    remaining = sorted(
        path
        for path in RETIRED_PATHS
        if (REPO_ROOT / path).exists()
    )

    assert remaining == []


def test_tracked_python_modules_do_not_import_retired_modules() -> None:
    offenders: list[str] = []
    retired_paths = {REPO_ROOT / path for path in RETIRED_PATHS}

    for py_file in _tracked_python_files():
        if py_file in retired_paths or not py_file.exists():
            continue

        imported = _imported_roots(py_file.read_text(encoding="utf-8"))
        blocked = sorted(imported & RETIRED_MODULES)
        if blocked:
            rel_path = py_file.relative_to(REPO_ROOT).as_posix()
            offenders.append(f"{rel_path}: {', '.join(blocked)}")

    assert offenders == []


def test_no_tracked_python_scripts_at_repo_root() -> None:
    result = subprocess.run(
        ["git", "ls-files", "*.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    sep = chr(47)
    root_scripts = [
        line.strip()
        for line in result.stdout.splitlines()
        if line.strip() and sep not in line.strip()
    ]
    assert root_scripts == []


def test_nested_aihedgefund_directory_absent() -> None:
    assert not (REPO_ROOT / "AIHedgeFund").exists()
