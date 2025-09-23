"""
Transformer-Based Price Prediction with Multi-Agent Processing
==============================================================
Advanced transformer models for financial time series prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import ray
import asyncio
from collections import deque
import math
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


@dataclass
class TransformerPrediction:
    """Transformer model prediction"""
    symbol: str
    predicted_price: float
    predicted_return: float
    attention_weights: np.ndarray
    confidence: float
    time_horizon: int
    feature_importance: Dict[str, float]
    timestamp: datetime


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0).transpose(0, 1))

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class FinancialTransformer(nn.Module):
    """Transformer model for financial time series"""

    def __init__(
        self,
        input_dim: int = 10,
        d_model: int = 128,
        n_heads: int = 8,
        n_layers: int = 4,
        d_ff: int = 512,
        output_dim: int = 1,
        dropout: float = 0.1,
        max_seq_len: int = 100
    ):
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model

        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        # Output layers
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, output_dim)
        )

        # Attention weights storage
        self.attention_weights = None

    def forward(self, x, mask=None):
        """
        Forward pass
        x: [batch_size, seq_len, input_dim]
        """
        # Project input
        x = self.input_projection(x)

        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]

        # Transformer encoding
        encoded = self.transformer_encoder(x, mask=mask)

        # Use last timestep for prediction
        output = self.output_projection(encoded[:, -1, :])

        return output, encoded

    def get_attention_weights(self, x):
        """Extract attention weights for interpretability"""
        # This would require modifying transformer to store attention weights
        # Simplified for demonstration
        return torch.rand(x.size(0), x.size(1), x.size(1))


class MultiScaleTransformer(nn.Module):
    """Multi-scale transformer for different time horizons"""

    def __init__(
        self,
        input_dim: int = 10,
        scales: List[int] = [5, 20, 60]  # Different time scales
    ):
        super().__init__()

        self.scales = scales

        # Create transformer for each scale
        self.transformers = nn.ModuleList([
            FinancialTransformer(
                input_dim=input_dim,
                d_model=64 if scale < 20 else 128,
                n_heads=4 if scale < 20 else 8,
                n_layers=2 if scale < 20 else 4,
                max_seq_len=scale
            )
            for scale in scales
        ])

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(len(scales), 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x_dict):
        """
        Forward pass with multiple scales
        x_dict: {scale: tensor} for each scale
        """
        outputs = []

        for i, scale in enumerate(self.scales):
            if scale in x_dict:
                out, _ = self.transformers[i](x_dict[scale])
                outputs.append(out)

        # Fuse predictions from different scales
        if outputs:
            stacked = torch.stack(outputs, dim=-1)
            fused = self.fusion(stacked)
            return fused.squeeze(-1)
        else:
            return torch.zeros(1)


class CrossAssetTransformer(nn.Module):
    """Transformer with cross-asset attention"""

    def __init__(
        self,
        n_assets: int = 10,
        input_dim: int = 10,
        d_model: int = 128,
        n_heads: int = 8
    ):
        super().__init__()

        self.n_assets = n_assets

        # Asset embedding
        self.asset_embedding = nn.Embedding(n_assets, d_model)

        # Shared transformer
        self.transformer = FinancialTransformer(
            input_dim=input_dim,
            d_model=d_model,
            n_heads=n_heads
        )

        # Cross-asset attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

    def forward(self, x_assets, asset_ids):
        """
        Forward pass with cross-asset attention
        x_assets: [batch_size, n_assets, seq_len, input_dim]
        asset_ids: [batch_size, n_assets]
        """
        batch_size, n_assets, seq_len, input_dim = x_assets.shape

        # Process each asset
        asset_representations = []

        for i in range(n_assets):
            asset_data = x_assets[:, i, :, :]
            _, encoded = self.transformer(asset_data)

            # Add asset embedding
            asset_emb = self.asset_embedding(asset_ids[:, i])
            asset_emb = asset_emb.unsqueeze(1).expand(-1, seq_len, -1)
            encoded = encoded + asset_emb

            # Take mean across time
            asset_rep = encoded.mean(dim=1)
            asset_representations.append(asset_rep)

        # Stack representations
        asset_reps = torch.stack(asset_representations, dim=1)

        # Apply cross-asset attention
        attended, attention_weights = self.cross_attention(
            asset_reps, asset_reps, asset_reps
        )

        return attended, attention_weights


class FinancialDataset(Dataset):
    """Dataset for financial time series"""

    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int = 60,
        prediction_horizon: int = 1,
        features: List[str] = None
    ):
        self.data = data
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.features = features or ['open', 'high', 'low', 'close', 'volume']

        # Normalize data
        self.mean = data[self.features].mean()
        self.std = data[self.features].std()
        self.normalized_data = (data[self.features] - self.mean) / self.std

    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_horizon

    def __getitem__(self, idx):
        # Get sequence
        x = self.normalized_data.iloc[idx:idx+self.sequence_length].values
        # Get target (next price)
        y = self.normalized_data['close'].iloc[idx+self.sequence_length+self.prediction_horizon-1]

        return torch.FloatTensor(x), torch.FloatTensor([y])


@ray.remote
class TransformerTrainingAgent:
    """Agent for transformer model training"""

    def __init__(self, agent_id: str, model_config: Dict[str, Any]):
        self.agent_id = agent_id
        self.model_config = model_config
        self.model = None
        self.optimizer = None
        self.best_loss = float('inf')

    def initialize_model(self):
        """Initialize transformer model"""
        self.model = FinancialTransformer(**self.model_config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        return True

    async def train_model(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        epochs: int = 50
    ) -> Dict[str, Any]:
        """Train transformer model"""

        if self.model is None:
            self.initialize_model()

        # Create datasets
        train_dataset = FinancialDataset(train_data)
        val_dataset = FinancialDataset(val_data)

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Training loop
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for batch_x, batch_y in train_loader:
                self.optimizer.zero_grad()

                predictions, _ = self.model(batch_x)
                loss = F.mse_loss(predictions, batch_y)

                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    predictions, _ = self.model(batch_x)
                    loss = F.mse_loss(predictions, batch_y)
                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)

            # Save best model
            if avg_val_loss < self.best_loss:
                self.best_loss = avg_val_loss
                # In production, save model checkpoint here

            # Early stopping
            if epoch > 10 and val_losses[-1] > val_losses[-5]:
                break

        return {
            'final_train_loss': train_losses[-1],
            'final_val_loss': val_losses[-1],
            'best_val_loss': self.best_loss,
            'epochs_trained': len(train_losses)
        }

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        if self.model is None:
            return np.array([])

        self.model.eval()
        dataset = FinancialDataset(data)
        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        predictions = []
        with torch.no_grad():
            for batch_x, _ in loader:
                preds, _ = self.model(batch_x)
                predictions.extend(preds.numpy())

        return np.array(predictions)

    def get_attention_weights(self, data: pd.DataFrame) -> np.ndarray:
        """Extract attention weights"""
        if self.model is None:
            return np.array([])

        self.model.eval()
        dataset = FinancialDataset(data)
        x = torch.FloatTensor(dataset.normalized_data.values[-60:]).unsqueeze(0)

        with torch.no_grad():
            attention = self.model.get_attention_weights(x)

        return attention.numpy()


@ray.remote
class MultiScaleTransformerAgent:
    """Agent for multi-scale transformer processing"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.model = MultiScaleTransformer()
        self.scales = [5, 20, 60]

    async def prepare_multi_scale_data(
        self,
        data: pd.DataFrame
    ) -> Dict[int, torch.Tensor]:
        """Prepare data at multiple scales"""
        multi_scale_data = {}

        for scale in self.scales:
            if len(data) >= scale:
                # Resample data to different scales
                if scale == 5:
                    scale_data = data.tail(scale)
                elif scale == 20:
                    scale_data = data.tail(scale).resample('5min').mean()
                else:  # scale == 60
                    scale_data = data.tail(scale).resample('15min').mean()

                # Normalize
                scale_data = (scale_data - scale_data.mean()) / scale_data.std()

                multi_scale_data[scale] = torch.FloatTensor(scale_data.values).unsqueeze(0)

        return multi_scale_data

    async def predict_multi_scale(
        self,
        data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate multi-scale predictions"""
        multi_scale_data = await self.prepare_multi_scale_data(data)

        self.model.eval()
        with torch.no_grad():
            prediction = self.model(multi_scale_data)

        return {
            'prediction': prediction.item(),
            'scales_used': list(multi_scale_data.keys()),
            'confidence': self._calculate_confidence(multi_scale_data)
        }

    def _calculate_confidence(self, multi_scale_data: Dict[int, torch.Tensor]) -> float:
        """Calculate prediction confidence based on scale agreement"""
        # Simplified confidence calculation
        n_scales = len(multi_scale_data)
        if n_scales == len(self.scales):
            return 0.9  # All scales available
        elif n_scales >= 2:
            return 0.7  # Multiple scales
        else:
            return 0.5  # Single scale


@ray.remote
class AttentionAnalysisAgent:
    """Agent for attention weight analysis"""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.attention_history = deque(maxlen=100)

    async def analyze_attention_patterns(
        self,
        attention_weights: np.ndarray,
        feature_names: List[str]
    ) -> Dict[str, Any]:
        """Analyze attention patterns for interpretability"""

        # Average attention across batch and heads
        avg_attention = attention_weights.mean(axis=0)

        # Find most attended positions
        temporal_importance = avg_attention.mean(axis=0)
        most_important_times = np.argsort(temporal_importance)[-5:]

        # Find most attended features (if attention is feature-wise)
        if len(avg_attention.shape) > 1:
            feature_importance = avg_attention.mean(axis=1)
            importance_dict = {
                feature_names[i]: float(feature_importance[i])
                for i in range(min(len(feature_names), len(feature_importance)))
            }
        else:
            importance_dict = {}

        # Detect attention patterns
        patterns = self._detect_patterns(avg_attention)

        return {
            'temporal_importance': temporal_importance.tolist(),
            'most_important_timesteps': most_important_times.tolist(),
            'feature_importance': importance_dict,
            'attention_patterns': patterns
        }

    def _detect_patterns(self, attention: np.ndarray) -> List[str]:
        """Detect common attention patterns"""
        patterns = []

        # Check for recency bias
        if attention.shape[0] > 10:
            recent_weight = attention[-5:].mean()
            older_weight = attention[:-5].mean()
            if recent_weight > older_weight * 1.5:
                patterns.append('recency_bias')

        # Check for periodicity
        if len(attention) > 20:
            # Simple autocorrelation check
            autocorr = np.correlate(attention, attention, mode='same')
            if np.max(autocorr[5:15]) > 0.7 * autocorr[0]:
                patterns.append('periodic_attention')

        # Check for concentration
        sorted_attention = np.sort(attention)[::-1]
        if sorted_attention[:5].sum() > 0.5 * attention.sum():
            patterns.append('concentrated_attention')

        return patterns


class TransformerPredictorOrchestrator:
    """Orchestrate transformer-based prediction with multi-agent processing"""

    def __init__(self, n_agents: int = 3):
        ray.init(ignore_reinit_error=True)

        # Initialize agents
        self.training_agents = [
            TransformerTrainingAgent.remote(
                f"trainer_{i}",
                {
                    'input_dim': 10,
                    'd_model': 128,
                    'n_heads': 8,
                    'n_layers': 4
                }
            )
            for i in range(n_agents)
        ]

        self.multi_scale_agent = MultiScaleTransformerAgent.remote("multi_scale")
        self.attention_agent = AttentionAnalysisAgent.remote("attention")

        # Model ensemble
        self.ensemble_predictions = deque(maxlen=100)
        self.performance_metrics = {}

    async def train_ensemble(
        self,
        train_data: Dict[str, pd.DataFrame],
        val_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """Train ensemble of transformer models"""

        training_tasks = []

        # Distribute training across agents
        symbols = list(train_data.keys())
        for i, agent in enumerate(self.training_agents):
            # Each agent trains on different symbols
            agent_symbols = symbols[i::len(self.training_agents)]

            for symbol in agent_symbols:
                if symbol in train_data and symbol in val_data:
                    task = agent.train_model.remote(
                        train_data[symbol],
                        val_data[symbol],
                        epochs=30
                    )
                    training_tasks.append((symbol, task))

        # Gather results
        training_results = {}
        for symbol, task in training_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )
            training_results[symbol] = result

        return training_results

    async def generate_predictions(
        self,
        data: Dict[str, pd.DataFrame]
    ) -> List[TransformerPrediction]:
        """Generate predictions using transformer ensemble"""

        predictions = []
        prediction_tasks = []

        for symbol, df in data.items():
            # Multi-scale prediction
            multi_scale_task = self.multi_scale_agent.predict_multi_scale.remote(df)

            # Regular transformer predictions from ensemble
            for agent in self.training_agents:
                pred_task = agent.predict.remote(df)
                prediction_tasks.append((symbol, 'transformer', pred_task))

            prediction_tasks.append((symbol, 'multi_scale', multi_scale_task))

        # Gather predictions
        symbol_predictions = {}

        for symbol, model_type, task in prediction_tasks:
            result = await asyncio.wrap_future(
                asyncio.get_event_loop().run_in_executor(None, ray.get, task)
            )

            if symbol not in symbol_predictions:
                symbol_predictions[symbol] = []

            symbol_predictions[symbol].append({
                'type': model_type,
                'prediction': result
            })

        # Create final predictions
        for symbol, preds in symbol_predictions.items():
            # Ensemble predictions
            if model_type == 'transformer':
                price_preds = [p['prediction'] for p in preds if p['type'] == 'transformer']
                if price_preds:
                    ensemble_pred = np.mean(price_preds)
                else:
                    ensemble_pred = 0
            else:
                ensemble_pred = preds[0]['prediction'].get('prediction', 0) if preds else 0

            # Get attention weights for interpretability
            if symbol in data:
                attention_task = self.training_agents[0].get_attention_weights.remote(data[symbol])
                attention_weights = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, attention_task)
                )

                # Analyze attention
                attention_analysis_task = self.attention_agent.analyze_attention_patterns.remote(
                    attention_weights,
                    ['open', 'high', 'low', 'close', 'volume']
                )
                attention_analysis = await asyncio.wrap_future(
                    asyncio.get_event_loop().run_in_executor(None, ray.get, attention_analysis_task)
                )
            else:
                attention_weights = np.array([])
                attention_analysis = {}

            # Create prediction object
            current_price = data[symbol]['close'].iloc[-1] if symbol in data else 100
            predicted_return = (ensemble_pred - current_price) / current_price if current_price > 0 else 0

            prediction = TransformerPrediction(
                symbol=symbol,
                predicted_price=current_price * (1 + ensemble_pred),
                predicted_return=predicted_return,
                attention_weights=attention_weights,
                confidence=self._calculate_ensemble_confidence(preds),
                time_horizon=1,
                feature_importance=attention_analysis.get('feature_importance', {}),
                timestamp=datetime.now()
            )

            predictions.append(prediction)

        return predictions

    def _calculate_ensemble_confidence(self, predictions: List[Dict]) -> float:
        """Calculate confidence based on prediction agreement"""
        if len(predictions) < 2:
            return 0.5

        # Calculate standard deviation of predictions
        pred_values = [p.get('prediction', 0) for p in predictions if 'prediction' in p]
        if pred_values:
            std = np.std(pred_values)
            mean = np.mean(pred_values)
            # Lower std = higher confidence
            if mean != 0:
                cv = std / abs(mean)  # Coefficient of variation
                confidence = max(0.3, 1 - cv)
            else:
                confidence = 0.5
        else:
            confidence = 0.5

        return min(confidence, 0.95)

    def get_model_performance(self) -> Dict[str, Any]:
        """Get ensemble performance metrics"""
        return {
            'n_models': len(self.training_agents),
            'predictions_generated': len(self.ensemble_predictions),
            'avg_confidence': np.mean([p.confidence for p in self.ensemble_predictions])
                            if self.ensemble_predictions else 0,
            'timestamp': datetime.now()
        }

    def cleanup(self):
        """Clean up Ray resources"""
        ray.shutdown()


# Example usage
async def main():
    """Example usage of transformer predictor"""
    orchestrator = TransformerPredictorOrchestrator(n_agents=3)

    # Generate sample data
    symbols = ['AAPL', 'GOOGL', 'MSFT']
    train_data = {}
    val_data = {}
    test_data = {}

    for symbol in symbols:
        # Create synthetic price data
        dates = pd.date_range(end=datetime.now(), periods=1000, freq='1h')
        prices = 100 + np.cumsum(np.random.randn(1000) * 0.5)

        df = pd.DataFrame({
            'open': prices + np.random.randn(1000) * 0.2,
            'high': prices + np.abs(np.random.randn(1000) * 0.3),
            'low': prices - np.abs(np.random.randn(1000) * 0.3),
            'close': prices,
            'volume': np.random.gamma(2, 1000000, 1000)
        }, index=dates)

        # Split data
        train_data[symbol] = df.iloc[:700]
        val_data[symbol] = df.iloc[700:850]
        test_data[symbol] = df.iloc[850:]

    # Train ensemble
    print("Training transformer ensemble...")
    training_results = await orchestrator.train_ensemble(train_data, val_data)

    print("\nTraining Results:")
    for symbol, metrics in training_results.items():
        print(f"  {symbol}:")
        print(f"    Best Val Loss: {metrics.get('best_val_loss', 0):.4f}")
        print(f"    Epochs: {metrics.get('epochs_trained', 0)}")

    # Generate predictions
    print("\nGenerating predictions...")
    predictions = await orchestrator.generate_predictions(test_data)

    print(f"\nGenerated {len(predictions)} predictions:")
    for pred in predictions:
        print(f"  {pred.symbol}:")
        print(f"    Predicted Return: {pred.predicted_return:.2%}")
        print(f"    Confidence: {pred.confidence:.1%}")
        print(f"    Top Features: {list(pred.feature_importance.keys())[:3]}")

    # Performance metrics
    metrics = orchestrator.get_model_performance()
    print("\nModel Performance:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    orchestrator.cleanup()


if __name__ == "__main__":
    asyncio.run(main())