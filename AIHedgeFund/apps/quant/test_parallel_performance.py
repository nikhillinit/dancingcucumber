"""
Performance Test Suite for Parallel ML Implementation
=====================================================
Tests parallelization efficiency and multi-agent coordination
"""

import asyncio
import time
import numpy as np
import pandas as pd
from typing import Dict, List
import multiprocessing as mp
import psutil
import logging

# Import our parallel modules
from parallel_ml_orchestrator import ParallelMLOrchestrator
from multi_agent_trading_system import MultiAgentTradingModel
from async_sentiment_analyzer import AsyncSentimentAnalyzer
from parallel_factor_generator import ParallelFactorGenerator
from lightweight_neural_predictor import LightweightNeuralPredictor, BatchPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerformanceTester:
    """Test suite for parallel performance"""

    def __init__(self):
        self.results = {}
        self.system_info = self._get_system_info()

    def _get_system_info(self) -> Dict:
        """Get system information"""
        return {
            'cpu_count': mp.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else 0,
            'memory_total': psutil.virtual_memory().total / (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available / (1024**3)
        }

    async def test_parallel_ml_orchestrator(self, n_symbols: int = 10):
        """Test ParallelMLOrchestrator performance"""
        logger.info(f"Testing ParallelMLOrchestrator with {n_symbols} symbols...")

        orchestrator = ParallelMLOrchestrator(n_workers=mp.cpu_count())
        symbols = [f"TEST{i}" for i in range(n_symbols)]

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)  # MB

        # Generate signals
        signals = await orchestrator.generate_signals_parallel(symbols)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)

        orchestrator.cleanup()

        return {
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'signals_per_second': len(signals) / (end_time - start_time),
            'n_signals': len(signals)
        }

    async def test_multi_agent_system(self, n_agents: int = 10):
        """Test multi-agent trading system"""
        logger.info(f"Testing MultiAgentTradingModel with {n_agents} agents...")

        model = MultiAgentTradingModel(
            n_momentum_agents=n_agents // 3,
            n_value_agents=n_agents // 3,
            n_arbitrage_agents=n_agents // 3
        )

        # Generate sample market data
        market_data = self._generate_market_data(5)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)

        # Execute trading round
        decisions = await model.execute_trading_round(market_data)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)

        return {
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'decisions_per_second': len(decisions) / (end_time - start_time),
            'n_decisions': len(decisions),
            'n_agents': n_agents
        }

    async def test_async_sentiment(self, n_sources: int = 50):
        """Test async sentiment analyzer"""
        logger.info(f"Testing AsyncSentimentAnalyzer with {n_sources} sources...")

        analyzer = AsyncSentimentAnalyzer(max_concurrent_requests=20)

        # Generate test sources
        sources = {
            'news': [f"Test news article {i}" for i in range(n_sources // 3)],
            'social': [f"Test social post {i}" for i in range(n_sources // 3)],
            'forums': [f"Test forum thread {i}" for i in range(n_sources // 3)]
        }

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)

        # Analyze sentiment
        results = await analyzer.analyze_multiple_sources('TEST', sources)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)

        return {
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'sources_per_second': n_sources / (end_time - start_time),
            'n_sources': n_sources
        }

    async def test_factor_generation(self, n_factors: int = 100):
        """Test parallel factor generation"""
        logger.info(f"Testing ParallelFactorGenerator...")

        generator = ParallelFactorGenerator(n_workers=mp.cpu_count())
        data = self._generate_market_data(1)['TEST0']

        results = {}
        for mode in ['ray', 'joblib', 'thread']:
            start_time = time.time()
            start_memory = psutil.Process().memory_info().rss / (1024**2)

            factors = await generator.generate_all_factors(data, parallel_mode=mode)

            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / (1024**2)

            results[mode] = {
                'execution_time': end_time - start_time,
                'memory_used': end_memory - start_memory,
                'factors_per_second': len(factors) / (end_time - start_time),
                'n_factors': len(factors)
            }

        return results

    async def test_neural_predictor(self, n_samples: int = 1000):
        """Test lightweight neural predictor"""
        logger.info(f"Testing LightweightNeuralPredictor with {n_samples} samples...")

        predictor = LightweightNeuralPredictor(n_models=5)

        # Generate training data
        X_train = np.random.randn(n_samples, 20)
        y_train = np.random.randn(n_samples)

        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / (1024**2)

        # Train models
        await predictor.train_parallel(X_train, y_train)

        # Make predictions
        X_test = np.random.randn(100, 20)
        predictor.predict_batch(X_test)

        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / (1024**2)

        return {
            'execution_time': end_time - start_time,
            'memory_used': end_memory - start_memory,
            'samples_per_second': n_samples / (end_time - start_time),
            'n_samples': n_samples
        }

    def _generate_market_data(self, n_symbols: int) -> Dict:
        """Generate sample market data"""
        data = {}
        for i in range(n_symbols):
            df = pd.DataFrame({
                'open': np.random.random(100) * 100 + 100,
                'high': np.random.random(100) * 100 + 105,
                'low': np.random.random(100) * 100 + 95,
                'close': np.random.random(100) * 100 + 100,
                'volume': np.random.random(100) * 1000000
            })
            data[f'TEST{i}'] = df
        return data

    async def run_all_tests(self):
        """Run all performance tests"""
        logger.info("=" * 60)
        logger.info("PARALLEL ML PERFORMANCE TEST SUITE")
        logger.info("=" * 60)
        logger.info(f"System Info: {self.system_info}")
        logger.info("=" * 60)

        # Test 1: ML Orchestrator
        try:
            result = await self.test_parallel_ml_orchestrator(10)
            self.results['ml_orchestrator'] = result
            logger.info(f"ML Orchestrator: {result['execution_time']:.2f}s, "
                       f"{result['signals_per_second']:.1f} signals/s")
        except Exception as e:
            logger.error(f"ML Orchestrator test failed: {e}")

        # Test 2: Multi-Agent System
        try:
            result = await self.test_multi_agent_system(9)
            self.results['multi_agent'] = result
            logger.info(f"Multi-Agent: {result['execution_time']:.2f}s, "
                       f"{result['decisions_per_second']:.1f} decisions/s")
        except Exception as e:
            logger.error(f"Multi-Agent test failed: {e}")

        # Test 3: Async Sentiment
        try:
            result = await self.test_async_sentiment(30)
            self.results['sentiment'] = result
            logger.info(f"Sentiment: {result['execution_time']:.2f}s, "
                       f"{result['sources_per_second']:.1f} sources/s")
        except Exception as e:
            logger.error(f"Sentiment test failed: {e}")

        # Test 4: Factor Generation
        try:
            result = await self.test_factor_generation()
            self.results['factors'] = result
            for mode, metrics in result.items():
                logger.info(f"Factors ({mode}): {metrics['execution_time']:.2f}s, "
                           f"{metrics['factors_per_second']:.1f} factors/s")
        except Exception as e:
            logger.error(f"Factor generation test failed: {e}")

        # Test 5: Neural Predictor
        try:
            result = await self.test_neural_predictor(500)
            self.results['neural'] = result
            logger.info(f"Neural: {result['execution_time']:.2f}s, "
                       f"{result['samples_per_second']:.1f} samples/s")
        except Exception as e:
            logger.error(f"Neural predictor test failed: {e}")

        return self.results

    def compare_parallel_vs_sequential(self):
        """Compare parallel vs sequential performance"""
        logger.info("\n" + "=" * 60)
        logger.info("PARALLEL VS SEQUENTIAL COMPARISON")
        logger.info("=" * 60)

        # Simulate sequential processing
        n_tasks = 100
        task_time = 0.01  # 10ms per task

        sequential_time = n_tasks * task_time
        parallel_time = sequential_time / self.system_info['cpu_count']

        speedup = sequential_time / parallel_time
        efficiency = speedup / self.system_info['cpu_count']

        logger.info(f"Theoretical speedup: {speedup:.2f}x")
        logger.info(f"Parallel efficiency: {efficiency:.1%}")
        logger.info(f"Sequential time: {sequential_time:.2f}s")
        logger.info(f"Parallel time: {parallel_time:.2f}s")

        return {
            'speedup': speedup,
            'efficiency': efficiency,
            'sequential_time': sequential_time,
            'parallel_time': parallel_time
        }

    def generate_report(self):
        """Generate performance report"""
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANCE TEST REPORT")
        logger.info("=" * 60)

        total_memory = sum(
            r.get('memory_used', 0)
            for r in self.results.values()
            if isinstance(r, dict)
        )

        logger.info(f"Total memory used: {total_memory:.1f} MB")

        # Calculate average performance
        if 'ml_orchestrator' in self.results:
            logger.info(f"\nML Orchestrator Performance:")
            logger.info(f"  - Execution time: {self.results['ml_orchestrator']['execution_time']:.2f}s")
            logger.info(f"  - Throughput: {self.results['ml_orchestrator']['signals_per_second']:.1f} signals/s")

        if 'multi_agent' in self.results:
            logger.info(f"\nMulti-Agent System Performance:")
            logger.info(f"  - Execution time: {self.results['multi_agent']['execution_time']:.2f}s")
            logger.info(f"  - Throughput: {self.results['multi_agent']['decisions_per_second']:.1f} decisions/s")

        logger.info("\n" + "=" * 60)
        logger.info("TEST COMPLETE")
        logger.info("=" * 60)


async def main():
    """Run performance tests"""
    tester = PerformanceTester()

    # Run all tests
    results = await tester.run_all_tests()

    # Compare parallel vs sequential
    comparison = tester.compare_parallel_vs_sequential()

    # Generate report
    tester.generate_report()

    # Print summary
    print("\n🚀 PERFORMANCE SUMMARY:")
    print(f"  CPU Cores: {tester.system_info['cpu_count']}")
    print(f"  Theoretical Speedup: {comparison['speedup']:.2f}x")
    print(f"  Tests Passed: {len(results)}")


if __name__ == "__main__":
    # Initialize Ray for tests
    import ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # Run tests
    asyncio.run(main())

    # Cleanup
    ray.shutdown()