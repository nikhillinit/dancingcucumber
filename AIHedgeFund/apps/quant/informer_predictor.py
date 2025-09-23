"""
Informer Time Series Predictor
==============================
Advanced attention-based architecture for long sequence forecasting
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import math

logger = logging.getLogger(__name__)


@dataclass
class InformerConfig:
    """Configuration for Informer model"""
    seq_len: int = 96  # Input sequence length
    label_len: int = 48  # Start token length
    pred_len: int = 24  # Prediction length
    enc_in: int = 7  # Encoder input size (features)
    dec_in: int = 7  # Decoder input size
    c_out: int = 1  # Output size
    d_model: int = 512  # Model dimension
    n_heads: int = 8  # Number of attention heads
    e_layers: int = 2  # Number of encoder layers
    d_layers: int = 1  # Number of decoder layers
    d_ff: int = 2048  # Feedforward dimension
    factor: int = 5  # ProbSparse attention factor
    dropout: float = 0.05
    embed_type: str = 'timeF'  # Time features encoding
    activation: str = 'gelu'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class ProbAttention(nn.Module):
    """ProbSparse self-attention mechanism"""

    def __init__(
        self,
        mask_flag: bool = True,
        factor: int = 5,
        scale: Optional[float] = None,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):
        """Calculate prob attention"""

        B, H, L_Q, D = Q.shape
        _, _, L_K, _ = K.shape

        # Calculate sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, D)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # Find top_k queries
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # Use reduced Q for calculation
        Q_reduce = Q[torch.arange(B)[:, None, None],
                     torch.arange(H)[None, :, None],
                     M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        """Get initial context"""

        B, H, L_V, D = V.shape
        if not self.mask_flag:
            V_sum = V.mean(dim=-2)
            context = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:
            context = V.cumsum(dim=-2)

        return context

    def forward(self, queries, keys, values, attn_mask=None):
        """ProbSparse attention forward pass"""

        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()

        U_part = min(U_part, L_K)
        u = min(u, L_Q)

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Scale
        scale = self.scale or 1. / math.sqrt(D)
        scores_top = scores_top * scale

        # Context
        context = self._get_initial_context(values, L_Q)

        # Update context
        if scores_top is not None:
            scores_top = F.softmax(scores_top, dim=-1)
            context[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   index, :] = torch.matmul(scores_top, values).type_as(context)

        return context.transpose(2, 1).contiguous()


class InformerEncoder(nn.Module):
    """Informer encoder with ProbSparse attention"""

    def __init__(self, config: InformerConfig):
        super().__init__()

        self.attention = ProbAttention(
            factor=config.factor,
            attention_dropout=config.dropout
        )

        self.conv1 = nn.Conv1d(
            in_channels=config.d_model,
            out_channels=config.d_ff,
            kernel_size=1
        )

        self.conv2 = nn.Conv1d(
            in_channels=config.d_ff,
            out_channels=config.d_model,
            kernel_size=1
        )

        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)

        self.dropout = nn.Dropout(config.dropout)

        self.activation = F.relu if config.activation == 'relu' else F.gelu

    def forward(self, x, attn_mask=None):
        """Encoder forward pass"""

        # Self attention
        new_x = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x = self.norm1(x)

        # Feed forward
        y = x.transpose(-1, 1)
        y = self.dropout(self.activation(self.conv1(y)))
        y = self.dropout(self.conv2(y))
        y = y.transpose(-1, 1)

        return self.norm2(x + y)


class Informer(nn.Module):
    """Informer model for time series forecasting"""

    def __init__(self, config: InformerConfig):
        super().__init__()
        self.config = config

        # Input embedding
        self.enc_embedding = nn.Linear(config.enc_in, config.d_model)
        self.dec_embedding = nn.Linear(config.dec_in, config.d_model)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.d_model)

        # Encoder
        self.encoders = nn.ModuleList([
            InformerEncoder(config)
            for _ in range(config.e_layers)
        ])

        # Decoder
        self.decoders = nn.ModuleList([
            InformerEncoder(config)  # Reuse encoder architecture
            for _ in range(config.d_layers)
        ])

        # Output projection
        self.projection = nn.Linear(config.d_model, config.c_out)

    def forward(self, x_enc, x_dec):
        """Forward pass"""

        # Embedding
        enc_out = self.enc_embedding(x_enc)
        enc_out = self.pos_encoding(enc_out)

        # Encoder
        for encoder in self.encoders:
            enc_out = encoder(enc_out)

        # Decoder input
        dec_out = self.dec_embedding(x_dec)
        dec_out = self.pos_encoding(dec_out)

        # Decoder
        for decoder in self.decoders:
            dec_out = decoder(dec_out)

        # Project to output
        output = self.projection(dec_out)

        return output


class TimeSeriesDataset(Dataset):
    """Dataset for time series prediction"""

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int,
        label_len: int,
        pred_len: int,
        features: List[str],
        target: str
    ):
        self.data = data
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.features = features
        self.target = target

        # Prepare data
        self.data_x = data[features].values
        self.data_y = data[target].values

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, index):
        # Input sequence
        s_begin = index
        s_end = s_begin + self.seq_len
        x = self.data_x[s_begin:s_end]

        # Label sequence (for decoder input)
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        y_seq = self.data_x[r_begin:r_end]

        # Target
        y = self.data_y[s_end:s_end + self.pred_len]

        return torch.FloatTensor(x), torch.FloatTensor(y_seq), torch.FloatTensor(y)


class InformerPredictor:
    """High-level interface for Informer predictions"""

    def __init__(self, config: Optional[InformerConfig] = None):
        self.config = config or InformerConfig()
        self.model = Informer(self.config)
        self.model.to(self.config.device)
        self.scaler = None

    def prepare_data(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        target_col: str
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Prepare data for training"""

        # Normalize data
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()

        df[feature_cols] = self.scaler.fit_transform(df[feature_cols])

        # Split data
        n = len(df)
        train_size = int(n * 0.7)
        val_size = int(n * 0.15)

        train_data = df[:train_size]
        val_data = df[train_size:train_size + val_size]
        test_data = df[train_size + val_size:]

        # Create datasets
        train_dataset = TimeSeriesDataset(
            train_data,
            self.config.seq_len,
            self.config.label_len,
            self.config.pred_len,
            feature_cols,
            target_col
        )

        val_dataset = TimeSeriesDataset(
            val_data,
            self.config.seq_len,
            self.config.label_len,
            self.config.pred_len,
            feature_cols,
            target_col
        )

        test_dataset = TimeSeriesDataset(
            test_data,
            self.config.seq_len,
            self.config.label_len,
            self.config.pred_len,
            feature_cols,
            target_col
        )

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

        return train_loader, val_loader, test_loader

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100
    ):
        """Train the model"""

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        criterion = nn.MSELoss()

        best_val_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0

            for batch_x, batch_y_seq, batch_y in train_loader:
                batch_x = batch_x.to(self.config.device)
                batch_y_seq = batch_y_seq.to(self.config.device)
                batch_y = batch_y.to(self.config.device)

                optimizer.zero_grad()

                # Decoder input
                dec_inp = torch.zeros_like(batch_y_seq).to(self.config.device)
                dec_inp[:, :self.config.label_len, :] = batch_y_seq[:, :self.config.label_len, :]

                outputs = self.model(batch_x, dec_inp)
                loss = criterion(outputs[:, -self.config.pred_len:, :], batch_y)

                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            # Validation
            self.model.eval()
            val_loss = 0

            with torch.no_grad():
                for batch_x, batch_y_seq, batch_y in val_loader:
                    batch_x = batch_x.to(self.config.device)
                    batch_y_seq = batch_y_seq.to(self.config.device)
                    batch_y = batch_y.to(self.config.device)

                    dec_inp = torch.zeros_like(batch_y_seq).to(self.config.device)
                    dec_inp[:, :self.config.label_len, :] = batch_y_seq[:, :self.config.label_len, :]

                    outputs = self.model(batch_x, dec_inp)
                    loss = criterion(outputs[:, -self.config.pred_len:, :], batch_y)

                    val_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), 'informer_best.pth')

            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")

    def predict(
        self,
        data: pd.DataFrame,
        feature_cols: List[str]
    ) -> np.ndarray:
        """Make predictions"""

        self.model.eval()

        # Normalize input
        if self.scaler:
            data[feature_cols] = self.scaler.transform(data[feature_cols])

        # Prepare input
        x = torch.FloatTensor(data[feature_cols].values).unsqueeze(0)
        x = x.to(self.config.device)

        # Create decoder input
        dec_inp = torch.zeros(1, self.config.label_len + self.config.pred_len, x.shape[-1])
        dec_inp[:, :self.config.label_len, :] = x[:, -self.config.label_len:, :]
        dec_inp = dec_inp.to(self.config.device)

        # Predict
        with torch.no_grad():
            output = self.model(x, dec_inp)
            predictions = output[:, -self.config.pred_len:, :].cpu().numpy()

        return predictions.squeeze()

    def predict_multi_step(
        self,
        data: pd.DataFrame,
        feature_cols: List[str],
        steps: int = 24
    ) -> pd.DataFrame:
        """Multi-step ahead prediction"""

        predictions = []

        for i in range(steps):
            pred = self.predict(data, feature_cols)
            predictions.append(pred[0])

            # Update data with prediction for next step
            new_row = data.iloc[-1:].copy()
            new_row[feature_cols[0]] = pred[0]
            data = pd.concat([data.iloc[1:], new_row], ignore_index=True)

        return pd.DataFrame(predictions, columns=['prediction'])


# Integration with existing system
class InformerIntegration:
    """Integrate Informer with AI Hedge Fund"""

    def __init__(self):
        self.predictor = InformerPredictor()

    def enhance_predictions(
        self,
        symbol: str,
        existing_predictions: pd.DataFrame
    ) -> pd.DataFrame:
        """Enhance existing predictions with Informer"""

        # Prepare features
        features = [
            'open', 'high', 'low', 'close', 'volume',
            'rsi', 'macd', 'bb_upper', 'bb_lower'
        ]

        # Get historical data
        import yfinance as yf
        data = yf.download(symbol, period='2y')

        # Calculate technical indicators
        data = self._add_technical_indicators(data)

        # Make predictions
        informer_preds = self.predictor.predict_multi_step(
            data.tail(96),
            features,
            steps=24
        )

        # Combine with existing predictions
        enhanced = existing_predictions.copy()
        enhanced['informer_prediction'] = informer_preds['prediction']
        enhanced['combined_prediction'] = (
            enhanced['prediction'] * 0.6 +
            enhanced['informer_prediction'] * 0.4
        )

        return enhanced

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to dataframe"""

        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2

        # Bollinger Bands
        df['bb_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)

        # Rename columns to lowercase
        df.columns = [col.lower() for col in df.columns]

        return df.dropna()