"""
Neural Network Price Prediction Models
======================================
LSTM and Transformer models for advanced price prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import yfinance as yf
from datetime import datetime, timedelta
import math
from transformers import (
    TimeSeriesTransformerModel,
    TimeSeriesTransformerConfig
)

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result from neural network prediction"""
    symbol: str
    predicted_price: float
    confidence_interval: Tuple[float, float]
    direction: str  # "up", "down", "neutral"
    confidence_score: float
    features_importance: Dict[str, float]
    model_type: str
    timestamp: datetime


class TimeSeriesDataset(Dataset):
    """PyTorch dataset for time series data"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerPredictor(nn.Module):
    """Transformer-based price predictor"""

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

        self.output_projection = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Project input
        x = self.input_projection(x)
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Output projection
        x = self.output_projection(x[:, -1, :])
        return x


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LSTMPredictor(nn.Module):
    """Enhanced LSTM with attention for price prediction"""

    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super(LSTMPredictor, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers,
            batch_first=True, dropout=dropout, bidirectional=True
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim * 2, num_heads=8, dropout=dropout)

        # Output layers
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.fc3 = nn.Linear(64, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)

        # Self-attention
        lstm_out = lstm_out.transpose(0, 1)  # (seq_len, batch, features)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_out = attn_out.transpose(0, 1)  # (batch, seq_len, features)

        # Take the last output
        out = attn_out[:, -1, :]

        # Fully connected layers
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        out = self.fc3(out)

        return out


class TransformerPredictor(nn.Module):
    """Transformer model for price prediction"""

    def __init__(self, input_dim: int, d_model: int = 512, nhead: int = 8, num_layers: int = 6):
        super(TransformerPredictor, self).__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            activation='gelu'
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # Project input to model dimension
        x = self.input_projection(x)

        # Add positional encoding
        x = self.positional_encoding(x)

        # Transformer expects (seq_len, batch, features)
        x = x.transpose(0, 1)

        # Pass through transformer
        x = self.transformer(x)

        # Take the last output
        x = x[-1, :, :]

        # Project to output
        out = self.output_projection(x)

        return out


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class NeuralPricePredictor:
    """Main class for neural network price prediction"""

    def __init__(self, model_type: str = "lstm", sequence_length: int = 60):
        self.model_type = model_type
        self.sequence_length = sequence_length
        self.model = None
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.feature_columns = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare technical features for training"""

        # Price features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['high_low_ratio'] = (df['High'] - df['Low']) / df['Close']
        df['close_open_ratio'] = (df['Close'] - df['Open']) / df['Open']

        # Volume features
        df['volume_ratio'] = df['Volume'] / df['Volume'].rolling(20).mean()
        df['dollar_volume'] = df['Close'] * df['Volume']

        # Moving averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['Close'].rolling(period).mean()
            df[f'sma_ratio_{period}'] = df['Close'] / df[f'sma_{period}']

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12).mean()
        exp2 = df['Close'].ewm(span=26).mean()
        df['macd'] = exp1 - exp2
        df['macd_signal'] = df['macd'].ewm(span=9).mean()

        # Bollinger Bands
        bb_period = 20
        bb_std = df['Close'].rolling(bb_period).std()
        bb_middle = df['Close'].rolling(bb_period).mean()
        df['bb_upper'] = bb_middle + 2 * bb_std
        df['bb_lower'] = bb_middle - 2 * bb_std
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()

        # Clean up
        df = df.dropna()

        return df

    def create_sequences(self, data: pd.DataFrame, target_col: str = 'Close') -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for time series prediction"""

        # Select feature columns
        self.feature_columns = [col for col in data.columns if col != target_col]

        X, y = [], []

        for i in range(len(data) - self.sequence_length - 1):
            X.append(data[self.feature_columns].iloc[i:i + self.sequence_length].values)
            y.append(data[target_col].iloc[i + self.sequence_length])

        return np.array(X), np.array(y)

    def train(self, symbol: str, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.001):
        """Train the neural network model"""

        logger.info(f"Training {self.model_type} model for {symbol}")

        # Fetch data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="2y")

        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        # Prepare features
        df = self.prepare_features(df)

        # Create sequences
        X, y = self.create_sequences(df)

        # Split data (80/20)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Scale data
        X_train_reshaped = X_train.reshape(-1, X_train.shape[-1])
        X_test_reshaped = X_test.reshape(-1, X_test.shape[-1])

        X_train_scaled = self.scaler_X.fit_transform(X_train_reshaped)
        X_test_scaled = self.scaler_X.transform(X_test_reshaped)

        X_train_scaled = X_train_scaled.reshape(X_train.shape)
        X_test_scaled = X_test_scaled.reshape(X_test.shape)

        y_train_scaled = self.scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
        y_test_scaled = self.scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        # Create datasets
        train_dataset = TimeSeriesDataset(X_train_scaled, y_train_scaled)
        test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Initialize model
        input_dim = X_train.shape[-1]

        if self.model_type == "lstm":
            self.model = LSTMPredictor(input_dim).to(self.device)
        elif self.model_type == "transformer":
            self.model = TransformerPredictor(input_dim).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Training setup
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # Training loop
        best_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation phase
            self.model.eval()
            test_loss = 0

            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = self.model(batch_X).squeeze()
                    loss = criterion(outputs, batch_y)
                    test_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_test_loss = test_loss / len(test_loader)

            scheduler.step(avg_test_loss)

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}, Test Loss = {avg_test_loss:.4f}")

            # Early stopping
            if avg_test_loss < best_loss:
                best_loss = avg_test_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        logger.info(f"Training completed. Best test loss: {best_loss:.4f}")

    def predict(self, symbol: str, days_ahead: int = 1) -> PredictionResult:
        """Make price prediction for a symbol"""

        if self.model is None:
            raise ValueError("Model not trained yet")

        # Fetch recent data
        ticker = yf.Ticker(symbol)
        df = ticker.history(period="6mo")

        if df.empty:
            raise ValueError(f"No data available for {symbol}")

        # Prepare features
        df = self.prepare_features(df)

        # Get last sequence
        last_sequence = df[self.feature_columns].iloc[-self.sequence_length:].values
        last_sequence_scaled = self.scaler_X.transform(last_sequence.reshape(-1, last_sequence.shape[-1]))
        last_sequence_scaled = last_sequence_scaled.reshape(1, self.sequence_length, -1)

        # Make prediction
        self.model.eval()
        with torch.no_grad():
            input_tensor = torch.FloatTensor(last_sequence_scaled).to(self.device)
            prediction_scaled = self.model(input_tensor).cpu().numpy()
            prediction = self.scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))[0, 0]

        # Calculate confidence and direction
        current_price = df['Close'].iloc[-1]
        price_change = (prediction - current_price) / current_price

        if abs(price_change) < 0.01:
            direction = "neutral"
        elif price_change > 0:
            direction = "up"
        else:
            direction = "down"

        # Confidence based on prediction magnitude and model performance
        confidence_score = min(100, 50 + abs(price_change) * 500)  # Simple heuristic

        # Confidence interval (simplified - would need proper uncertainty quantification)
        std_dev = df['Close'].pct_change().std() * current_price
        confidence_interval = (prediction - 2 * std_dev, prediction + 2 * std_dev)

        return PredictionResult(
            symbol=symbol,
            predicted_price=float(prediction),
            confidence_interval=confidence_interval,
            direction=direction,
            confidence_score=confidence_score,
            features_importance={},  # Would need attention weights or SHAP values
            model_type=self.model_type,
            timestamp=datetime.now()
        )


class EnsemblePredictor:
    """Ensemble of multiple neural network models"""

    def __init__(self):
        self.models = {
            'lstm': NeuralPricePredictor('lstm'),
            'transformer': NeuralPricePredictor('transformer')
        }

    def train_all(self, symbol: str, **kwargs):
        """Train all models in the ensemble"""
        for name, model in self.models.items():
            logger.info(f"Training {name} model")
            model.train(symbol, **kwargs)

    def predict(self, symbol: str) -> PredictionResult:
        """Make ensemble prediction"""
        predictions = []

        for name, model in self.models.items():
            try:
                pred = model.predict(symbol)
                predictions.append(pred)
            except Exception as e:
                logger.warning(f"Error in {name} prediction: {e}")

        if not predictions:
            raise ValueError("No successful predictions from ensemble")

        # Average predictions
        avg_price = np.mean([p.predicted_price for p in predictions])
        avg_confidence = np.mean([p.confidence_score for p in predictions])

        # Determine direction by majority vote
        directions = [p.direction for p in predictions]
        direction = max(set(directions), key=directions.count)

        # Combine confidence intervals
        lower_bounds = [p.confidence_interval[0] for p in predictions]
        upper_bounds = [p.confidence_interval[1] for p in predictions]
        confidence_interval = (np.min(lower_bounds), np.max(upper_bounds))

        return PredictionResult(
            symbol=symbol,
            predicted_price=avg_price,
            confidence_interval=confidence_interval,
            direction=direction,
            confidence_score=avg_confidence,
            features_importance={},
            model_type="ensemble",
            timestamp=datetime.now()
        )