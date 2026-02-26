# WHL 2026 Analytics - Comprehensive Improvement Plan

## Executive Summary

This document outlines a multi-phase improvement plan for the WHL analytics pipeline across 8 key dimensions. Current performance (Log Loss: 0.6696, Accuracy: 57.9%) can be significantly improved while maintaining leakage-safe principles.

---

## 1. MODEL PERFORMANCE IMPROVEMENTS

### 1.1 Algorithm Enhancements

#### Gradient Boosting Upgrades
```python
# Current: XGB with conservative params
# Improved: Multi-GBDT ensemble with different objectives

# 1. CatBoost Integration
# - Handles categorical features natively
# - Ordered boosting reduces overfitting
# - Built-in feature importance

from catboost import CatBoostClassifier
cat_model = CatBoostClassifier(
    iterations=2000,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=3.0,
    random_seed=42,
    verbose=False,
    loss_function='Logloss'
)

# 2. LightGBM Optimization
# - Dart booster for better generalization
# - Feature fraction for diversity

lgb_model = lgb.LGBMClassifier(
    n_estimators=2000,
    learning_rate=0.02,
    max_depth=5,
    num_leaves=31,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    boosting_type='dart',
    drop_rate=0.1,
    min_data_in_leaf=20
)

# 3. XGBoost with Monotonic Constraints
# - Enforce hockey domain knowledge
# - Higher Elo -> higher win probability

constraints = {
    'elo_diff': 1,  # Positive: higher Elo should increase win prob
    'xG_diff': 1,
    'form_5': 1
}
```

#### Neural Network Architecture
```python
# Current: Basic MLP
# Improved: TabNet + Residual MLP

# TabNet for attention-based feature selection
from pytorch_tabnet.tab_model import TabNetClassifier

tabnet = TabNetClassifier(
    n_d=32,  # width of decision prediction
    n_a=32,  # width of attention embedding
    n_steps=5,  # number of decision steps
    gamma=1.5,  # relaxation factor
    lambda_sparse=1e-4,  # sparsity regularization
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=2e-2),
    scheduler_params={"step_size":50, "gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='entmax'
)

# Residual MLP with BatchNorm and Dropout
class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64, 32]):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(self._make_block(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.blocks = nn.ModuleList(layers)
        self.output = nn.Linear(prev_dim, 1)
        
    def _make_block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
        )
    
    def forward(self, x):
        x = self.input_bn(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            if x.shape == residual.shape:
                x = x + residual  # Skip connection
            x = F.relu(x)
        return torch.sigmoid(self.output(x))
```

### 1.2 Advanced Ensemble Methods

#### Stacking Architecture
```python
# Three-level stacking ensemble

# Level 0: Base Models (diverse learners)
base_models = [
    ('elo', EloModel()),
    ('sgd', SGDClassifier()),
    ('ridge', RidgeClassifier()),
    ('lgb', LGBMClassifier()),
    ('cat', CatBoostClassifier()),
    ('xgb', XGBClassifier()),
    ('svm', SVC(probability=True)),
    ('mlp', MLPClassifier())
]

# Level 1: Meta-learner per model family
meta_learners = {
    'linear': LogisticRegression(C=0.1),
    'tree': XGBClassifier(max_depth=3),
    'nn': MLPClassifier(hidden_layer_sizes=(32, 16))
}

# Level 2: Final blender (convex combination with constraints)
from scipy.optimize import minimize

def optimize_blend_weights(oof_predictions, y_true):
    """Optimize blend weights using log loss with L2 regularization"""
    def objective(weights):
        weights = np.maximum(weights, 0)  # Non-negative
        weights = weights / weights.sum()  # Sum to 1
        blended = np.average(oof_predictions, axis=0, weights=weights)
        return log_loss(y_true, blended) + 0.001 * np.sum(weights**2)
    
    result = minimize(objective, 
                     x0=np.ones(len(base_models)) / len(base_models),
                     method='SLSQP',
                     bounds=[(0, 1)] * len(base_models),
                     constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1})
    return result.x
```

#### Dynamic Model Selection
```python
# Context-aware model selection based on matchup characteristics

def select_model_by_context(features):
    """
    Different models for different scenarios:
    - Elo-heavy for mismatched teams (high Elo diff)
    - xG-heavy for even matchups
    - Form-heavy for teams with recent changes
    """
    elo_diff = abs(features['home_elo'] - features['away_elo'])
    form_variance = features['home_form_std'] + features['away_form_std']
    
    if elo_diff > 200:
        return 'elo_weighted'  # 70% Elo, 30% form
    elif form_variance > 0.15:
        return 'form_weighted'  # 40% form, 40% xG, 20% Elo
    else:
        return 'balanced_blend'
```

### 1.3 Calibration Improvements

#### Isotonic vs Platt Scaling
```python
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

def calibrate_predictions(model, X_train, y_train, X_cal, y_cal, method='isotonic'):
    """
    Calibrate model predictions using held-out calibration set
    """
    # Train on training set
    model.fit(X_train, y_train)
    
    # Calibrate on separate calibration set
    calibrated = CalibratedClassifierCV(model, method=method, cv='prefit')
    calibrated.fit(X_cal, y_cal)
    
    return calibrated

# Beta calibration for probabilities (better than Platt for some distributions)
from betacal import BetaCalibration

def beta_calibrate(probs, y_true):
    """
    Beta calibration assuming probabilities follow Beta distribution
    Better calibrated tails than Platt scaling
    """
    bc = BetaCalibration(parameters="abm")  # 3-parameter version
    bc.fit(probs.reshape(-1, 1), y_true)
    return bc.predict(probs.reshape(-1, 1))
```

#### Temperature Scaling
```python
import torch
import torch.nn as nn
import torch.optim as optim

def temperature_scaling(logits, labels, max_iter=50):
    """
    Learn temperature parameter for calibration
    Maintains ranking while adjusting confidence
    """
    temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.LBFGS([temperature], lr=0.01, max_iter=max_iter)
    
    def eval_loss():
        loss = criterion(logits / temperature, labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    return temperature.item()
```

---

## 2. FEATURE ENGINEERING ENHANCEMENTS

### 2.1 Temporal Features

#### Momentum & Trend Features
```python
def compute_trend_features(panel):
    """
    Extract momentum and trend patterns
    """
    # Exponentially weighted moving averages (more weight to recent games)
    for span in [3, 5, 10]:
        panel[f'xG_ewm_{span}'] = panel.groupby('team')['xG'].transform(
            lambda x: x.shift(1).ewm(span=span, min_periods=1).mean()
        )
    
    # Linear trend (slope) in recent performance
    def linear_trend(x, window=5):
        if len(x) < window:
            return 0
        x_vals = np.arange(window)
        y_vals = x[-window:]
        slope, _, _, _, _ = linregress(x_vals, y_vals)
        return slope
    
    panel['xG_trend_5'] = panel.groupby('team')['xG'].transform(
        lambda x: x.shift(1).rolling(5, min_periods=3).apply(linear_trend)
    )
    
    # Acceleration (second derivative)
    panel['xG_acceleration'] = panel['xG_trend_5'] - panel.groupby('team')['xG_trend_5'].shift(1)
    
    # Volatility clustering (recent std vs historical std)
    panel['volatility_ratio'] = (
        panel.groupby('team')['xG_diff'].transform(lambda x: x.shift(1).rolling(5).std()) /
        panel.groupby('team')['xG_diff'].transform(lambda x: x.shift(1).rolling(20).std())
    )
    
    return panel
```

#### Rest & Schedule Features
```python
def compute_schedule_features(games, rest_df):
    """
    Enhanced rest and schedule features
    """
    # Days of rest (categorical: 0, 1, 2, 3+)
    rest_df['rest_category'] = pd.cut(
        rest_df['days_rest'], 
        bins=[-1, 0, 1, 2, 100],
        labels=['b2b', '1_day', '2_day', '3plus']
    )
    
    # Fatigue index (games in last 7 days)
    games['date'] = pd.to_datetime(games['date'])
    games['games_last_7'] = games.groupby('team').rolling('7D', on='date')['game_id'].count()
    
    # Travel fatigue (if location data available)
    # games['travel_distance'] = compute_haversine_distance(prev_location, curr_location)
    
    # Home/away streak effects
    games['home_streak'] = (
        games.groupby('team')['is_home']
        .apply(lambda x: x * (x.groupby((x != x.shift()).cumsum()).cumcount() + 1))
    )
    
    # Schedule strength (opponent Elo of last 5 games)
    games['schedule_strength_last5'] = games.groupby('team').apply(
        lambda x: x.shift(1).rolling(5)['opp_elo'].mean()
    )
    
    return games
```

### 2.2 Interaction Features

#### Feature Crosses
```python
def create_interaction_features(features_df):
    """
    Create meaningful feature interactions
    """
    df = features_df.copy()
    
    # Elo x Form interaction
    df['elo_form_interaction'] = df['elo_diff'] * df['form_5_diff']
    
    # xG x Quality interaction (high xG should matter more for good teams)
    df['xG_quality_interaction'] = df['xG_diff'] * df['elo_diff']
    
    # PP x PK interaction (special teams battle)
    df['special_teams_ratio'] = df['home_pp60'] / (df['away_pk60'] + 0.1)
    
    # Depth x Consistency interaction
    df['depth_consistency'] = df['line_disparity'] * (1 - df['xG_volatility'])
    
    # Clutch factor (performance in close games)
    df['clutch_ratio'] = df['close_wins'] / (df['close_games'] + 1)
    
    # Recency-weighted Elo (blend of current Elo and recent form)
    df['elo_recent_blend'] = 0.7 * df['elo'] + 0.3 * df['form_elo']
    
    return df
```

#### Polynomial Features with Regularization
```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectKBest, mutual_info_classif

def create_polynomial_features(X_train, X_test, degree=2, k_best=50):
    """
    Create polynomial features with selection to avoid explosion
    """
    poly = PolynomialFeatures(degree=degree, interaction_only=True, include_bias=False)
    
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Select top k features by mutual information
    selector = SelectKBest(mutual_info_classif, k=min(k_best, X_train_poly.shape[1]))
    X_train_selected = selector.fit_transform(X_train_poly, y_train)
    X_test_selected = selector.transform(X_test_poly)
    
    return X_train_selected, X_test_selected, selector.get_support()
```

### 2.3 Embedding Features

#### Team Embeddings
```python
import torch.nn as nn

class TeamEmbeddingModel(nn.Module):
    """
    Learn dense embeddings for teams based on their characteristics
    """
    def __init__(self, n_teams=32, embedding_dim=16):
        super().__init__()
        self.team_embeddings = nn.Embedding(n_teams, embedding_dim)
        
        # Learnable team archetype weights
        self.archetype_proj = nn.Linear(embedding_dim, 4)  # 4 archetypes
        
    def forward(self, team_ids):
        emb = self.team_embeddings(team_ids)
        archetype_weights = F.softmax(self.archetype_proj(emb), dim=-1)
        return emb, archetype_weights

# Usage: Learn embeddings from historical matchups
def train_team_embeddings(games, n_epochs=100):
    model = TeamEmbeddingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(n_epochs):
        # Contrastive learning: similar teams should have similar embeddings
        # Teams with similar win patterns vs common opponents
        loss = contrastive_loss(model, games)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return model.team_embeddings.weight.detach().numpy()
```

#### Line Combination Embeddings
```python
def compute_line_embeddings(line_data):
    """
    Create embeddings for line combinations
    Based on xG generation, chemistry, consistency
    """
    # Aggregate line-level stats to team level
    line_features = line_data.groupby(['team', 'line_id']).agg({
        'xG_per_60': 'mean',
        'corsi': 'mean',
        'goals': 'sum',
        'minutes': 'sum',
        'game_id': 'nunique'  # games played together
    }).reset_index()
    
    # Chemistry score (actual vs expected based on individual players)
    line_features['chemistry'] = (
        line_features['xG_per_60'] / line_features['expected_xG']
    )
    
    # Use PCA to create line embedding
    from sklearn.decomposition import PCA
    pca = PCA(n_components=5)
    embeddings = pca.fit_transform(line_features[['xG_per_60', 'corsi', 'chemistry']])
    
    return embeddings
```

---

## 3. CODE QUALITY & ARCHITECTURE

### 3.1 Modular Architecture

```
whl_analytics/
├── config/
│   ├── __init__.py
│   ├── settings.py          # Centralized configuration
│   └── model_configs.yaml   # Model hyperparameters
├── data/
│   ├── __init__.py
│   ├── loaders.py           # Data loading utilities
│   ├── validators.py        # Data validation checks
│   └── transformers.py      # Feature engineering
├── features/
│   ├── __init__.py
│   ├── base.py              # Base feature transformer
│   ├── temporal.py          # Time-based features
│   ├── interactions.py      # Feature crosses
│   └── embeddings.py        # Learned embeddings
├── models/
│   ├── __init__.py
│   ├── base.py              # Base model interface
│   ├── elo.py               # Elo rating system
│   ├── ensemble.py          # Ensemble methods
│   ├── neural.py            # Neural networks
│   └── calibration.py       # Probability calibration
├── training/
│   ├── __init__.py
│   ├── cv.py                # Cross-validation strategies
│   ├── leakage.py           # Leakage prevention
│   └── pipeline.py          # Training orchestration
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py           # Custom metrics
│   ├── calibration.py       # Calibration assessment
│   └── robustness.py        # Robustness testing
├── interpretability/
│   ├── __init__.py
│   ├── shap_explainer.py    # SHAP values
│   ├── feature_importance.py
│   └── report_generator.py
└── utils/
    ├── __init__.py
    ├── logging.py
    └── visualization.py
```

### 3.2 Configuration Management

```python
# config/settings.py
from dataclasses import dataclass
from typing import List, Dict
import yaml

@dataclass
class EloConfig:
    k_factor: float = 8.0
    home_advantage: float = 60.0
    xg_weight: float = 0.15
    decay: bool = True
    regression: float = 0.10
    regression_period: int = 200

@dataclass
class ModelConfig:
    name: str
    params: Dict
    enabled: bool = True

@dataclass
class PipelineConfig:
    data_path: str
    output_path: str
    random_seed: int = 42
    test_size: float = 0.15
    
    elo: EloConfig = EloConfig()
    models: List[ModelConfig] = None
    
    @classmethod
    def from_yaml(cls, path: str):
        with open(path) as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

# config/model_configs.yaml
models:
  - name: "sgd"
    enabled: true
    params:
      alpha: 0.005
      max_iter: 2000
      
  - name: "xgboost"
    enabled: true
    params:
      n_estimators: 800
      max_depth: 3
      learning_rate: 0.02
      
  - name: "lightgbm"
    enabled: true
    params:
      n_estimators: 2000
      learning_rate: 0.02
      num_leaves: 31
```

### 3.3 Base Classes and Interfaces

```python
# models/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any

class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_fitted = False
        self.metadata = {}
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BaseModel':
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Generate probability predictions"""
        pass
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance if available"""
        return {}
    
    def save(self, path: str):
        """Save model to disk"""
        import joblib
        joblib.dump(self, path)
    
    @classmethod
    def load(cls, path: str):
        """Load model from disk"""
        import joblib
        return joblib.load(path)

# models/elo.py
class EloModel(BaseModel):
    """Elo rating system implementation"""
    
    def __init__(self, config: EloConfig):
        super().__init__(config)
        self.ratings = {}
        self.history = []
    
    def fit(self, games: pd.DataFrame) -> 'EloModel':
        """
        Compute Elo ratings from historical games
        """
        for _, game in games.iterrows():
            self._update_ratings(game)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, matchup: pd.DataFrame) -> np.ndarray:
        """
        Predict win probability from pregame ratings
        """
        probs = []
        for _, row in matchup.iterrows():
            home_elo = self.ratings.get(row['home_team'], 1500)
            away_elo = self.ratings.get(row['away_team'], 1500)
            
            prob = self._expected_score(home_elo, away_elo, row.get('is_home', True))
            probs.append([1 - prob, prob])
        
        return np.array(probs)
```

### 3.4 Pipeline Orchestration

```python
# training/pipeline.py
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from typing import List
import logging

logger = logging.getLogger(__name__)

class WHLPipeline:
    """
    Main training pipeline with leakage prevention
    """
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.models: List[BaseModel] = []
        self.feature_transformer = None
        self.ensemble = None
    
    def build_feature_pipeline(self):
        """
        Build sklearn-compatible feature pipeline
        """
        numeric_features = ['elo', 'xG_diff', 'form_5', 'rest_days']
        categorical_features = ['rest_category', 'home_streak_length']
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.feature_transformer = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        return self
    
    def fit(self, data: pd.DataFrame) -> 'WHLPipeline':
        """
        Full training pipeline with time-series CV
        """
        logger.info("Starting training pipeline")
        
        # 1. Data validation
        self._validate_data(data)
        
        # 2. Feature engineering
        features = self._build_features(data)
        
        # 3. Time-series split
        splits = self._create_time_splits(data)
        
        # 4. Train base models with OOF predictions
        oof_predictions = self._train_base_models(features, splits)
        
        # 5. Train ensemble on OOF predictions
        self.ensemble = self._train_ensemble(oof_predictions, data['target'])
        
        logger.info("Training complete")
        return self
    
    def _create_time_splits(self, data: pd.DataFrame):
        """
        Create time-series aware splits
        """
        from sklearn.model_selection import TimeSeriesSplit
        
        # Ensure data is sorted by date
        data = data.sort_values('date')
        
        tscv = TimeSeriesSplit(n_splits=5)
        splits = []
        
        for train_idx, val_idx in tscv.split(data):
            # Additional check: no overlap in game dates
            train_dates = data.iloc[train_idx]['date'].max()
            val_dates = data.iloc[val_idx]['date'].min()
            
            assert train_dates < val_dates, "Data leakage detected in split!"
            splits.append((train_idx, val_idx))
        
        return splits
```

### 3.5 Comprehensive Testing

```python
# tests/test_leakage.py
import pytest
import pandas as pd
import numpy as np

def test_no_future_data_in_training():
    """
    Ensure training set never contains games after validation set
    """
    pipeline = WHLPipeline(config)
    
    for train_idx, val_idx in pipeline.splits:
        train_dates = data.iloc[train_idx]['date']
        val_dates = data.iloc[val_idx]['date']
        
        assert train_dates.max() < val_dates.min(), \
            f"Training data contains future games: {train_dates.max()} >= {val_dates.min()}"

def test_feature_leakage():
    """
    Verify features only use pregame information
    """
    game_id = 100
    
    # Get features for game 100
    features = build_pregame_features(game_id)
    
    # Features should not reference games >= 100
    for col in features.columns:
        if 'cum_' in col or 'roll_' in col:
            # These should be computed from games before 100
            pass

def test_target_leakage_in_features():
    """
    Ensure no feature is perfectly correlated with target
    """
    X, y = build_features_and_target()
    
    for col in X.columns:
        corr = np.corrcoef(X[col], y)[0, 1]
        assert abs(corr) < 0.99, \
            f"Feature {col} is perfectly correlated with target - likely leakage"

# tests/test_models.py
def test_model_calibration():
    """
    Test that model predictions are well-calibrated
    """
    model = load_trained_model()
    X_test, y_test = load_test_data()
    
    probs = model.predict_proba(X_test)[:, 1]
    
    # Calibration curve
    from sklearn.calibration import calibration_curve
    prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
    
    # Check calibration error
    calibration_error = np.mean(np.abs(prob_true - prob_pred))
    assert calibration_error < 0.05, f"Poor calibration: {calibration_error}"

def test_model_consistency():
    """
    Test that model produces consistent predictions
    """
    model = load_trained_model()
    X = load_sample_data()
    
    pred1 = model.predict_proba(X)
    pred2 = model.predict_proba(X)
    
    np.testing.assert_array_almost_equal(pred1, pred2)
```

---

## 4. INTERPRETABILITY & EXPLAINABILITY

### 4.1 SHAP Integration

```python
# interpretability/shap_explainer.py
import shap

class SHAPExplainer:
    """
    SHAP-based model explanation
    """
    
    def __init__(self, model, feature_names):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
    
    def fit(self, X_background):
        """
        Initialize SHAP explainer
        """
        if hasattr(self.model, 'predict_proba'):
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.explainer = shap.KernelExplainer(
                lambda x: self.model.predict_proba(x)[:, 1],
                X_background
            )
        return self
    
    def explain_prediction(self, X_instance):
        """
        Generate explanation for single prediction
        """
        shap_values = self.explainer.shap_values(X_instance)
        
        # Create explanation dict
        explanation = {
            'base_value': self.explainer.expected_value,
            'prediction': np.sum(shap_values) + self.explainer.expected_value,
            'contributions': dict(zip(self.feature_names, shap_values))
        }
        
        # Sort by absolute contribution
        explanation['top_positive'] = sorted(
            [(f, v) for f, v in explanation['contributions'].items() if v > 0],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        
        explanation['top_negative'] = sorted(
            [(f, v) for f, v in explanation['contributions'].items() if v < 0],
            key=lambda x: x[1]
        )[:5]
        
        return explanation
    
    def generate_force_plot(self, X_instance, output_path=None):
        """
        Generate force plot for visualization
        """
        shap_values = self.explainer.shap_values(X_instance)
        
        plot = shap.force_plot(
            self.explainer.expected_value,
            shap_values,
            X_instance,
            feature_names=self.feature_names,
            matplotlib=True
        )
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight')
        
        return plot
```

### 4.2 Plain-English Explanations

```python
# interpretability/natural_language.py
class ExplanationGenerator:
    """
    Generate human-readable explanations
    """
    
    TEMPLATES = {
        'elo_advantage': "{home_team} has a significant rating advantage ({elo_diff} Elo points)",
        'momentum': "{home_team} has strong momentum, winning {win_streak} consecutive games",
        'rest_disadvantage': "{home_team} is playing on {rest_days} days rest vs {away_rest} for {away_team}",
        'line_strength': "{home_team}'s top line has generated {top_line_xg} xG/60 vs {away_team}'s {away_top_line}",
        'special_teams': "{home_team}'s power play ({pp_pct}%) matches up well against {away_team}'s penalty kill ({pk_pct}%)",
        'close_matchup': "This is a tightly matched game with only {elo_diff} Elo points separating the teams"
    }
    
    def generate_matchup_summary(self, features, prediction):
        """
        Generate natural language summary
        """
        factors = []
        
        # Elo difference
        elo_diff = abs(features['home_elo'] - features['away_elo'])
        if elo_diff > 100:
            favored = features['home_team'] if features['home_elo'] > features['away_elo'] else features['away_team']
            factors.append(f"**Elo Advantage**: {favored} has a {elo_diff:.0f} point rating edge")
        
        # Recent form
        form_diff = features['home_form_5'] - features['away_form_5']
        if abs(form_diff) > 0.1:
            better_form = features['home_team'] if form_diff > 0 else features['away_team']
            factors.append(f"**Momentum**: {better_form} has better recent form ({abs(form_diff):.1%} point advantage)")
        
        # Rest advantage
        rest_diff = features['home_rest'] - features['away_rest']
        if abs(rest_diff) >= 2:
            more_rest = features['home_team'] if rest_diff > 0 else features['away_team']
            factors.append(f"**Rest Advantage**: {more_rest} has {abs(rest_diff)} more days of rest")
        
        # Special teams
        pp_diff = features['home_pp60'] - features['away_pp60']
        if abs(pp_diff) > 0.2:
            better_pp = features['home_team'] if pp_diff > 0 else features['away_team']
            factors.append(f"**Special Teams**: {better_pp} has a stronger power play")
        
        # Line disparity
        if features['home_line_disparity'] > 1.5:
            factors.append(f"**Star-Driven**: {features['home_team']} relies heavily on their top line")
        
        # Generate summary
        winner = prediction['predicted_winner']
        confidence = prediction['confidence']
        
        summary = f"""
## Matchup Analysis: {features['home_team']} vs {features['away_team']}

**Prediction**: {winner} wins with {confidence:.1%} confidence

### Key Factors:
"""
        for i, factor in enumerate(factors, 1):
            summary += f"{i}. {factor}\n"
        
        summary += f"""
### Model Confidence Breakdown:
- Elo Model: {prediction['elo_prob']:.1%}
- Form Model: {prediction['form_prob']:.1%}
- Ensemble Consensus: {prediction['consensus_prob']:.1%}
- Uncertainty: {prediction['uncertainty']:.3f}
"""
        
        return summary
```

### 4.3 Interactive Dashboard

```python
# interpretability/dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

def create_interactive_dashboard():
    """
    Streamlit dashboard for exploring predictions
    """
    st.title("WHL 2026 Analytics Dashboard")
    
    # Sidebar filters
    st.sidebar.header("Filters")
    team_filter = st.sidebar.multiselect("Teams", all_teams, default=all_teams[:4])
    date_range = st.sidebar.date_input("Date Range", [season_start, season_end])
    min_confidence = st.sidebar.slider("Minimum Confidence", 0.5, 1.0, 0.6)
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["Power Rankings", "Matchup Predictor", "Model Analysis"])
    
    with tab1:
        st.header("Team Power Rankings")
        
        rankings = load_power_rankings()
        rankings = rankings[rankings['team'].isin(team_filter)]
        
        # Interactive table
        st.dataframe(rankings.sort_values('power_score', ascending=False))
        
        # Visualizations
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(rankings, x='elo', y='xG_diff', 
                           size='wins', color='team',
                           title='Elo vs Expected Goals')
            st.plotly_chart(fig)
        
        with col2:
            fig = px.bar(rankings.sort_values('power_score', ascending=True).tail(10),
                        x='power_score', y='team', orientation='h',
                        title='Top 10 Teams by Power Score')
            st.plotly_chart(fig)
    
    with tab2:
        st.header("Matchup Predictor")
        
        col1, col2 = st.columns(2)
        with col1:
            home_team = st.selectbox("Home Team", all_teams)
        with col2:
            away_team = st.selectbox("Away Team", [t for t in all_teams if t != home_team])
        
        if st.button("Predict Matchup"):
            prediction = predict_matchup(home_team, away_team)
            
            # Gauge chart for win probability
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction['home_win_prob'] * 100,
                title = {'text': f"{home_team} Win Probability"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 33], 'color': "lightgray"},
                        {'range': [33, 66], 'color': "gray"},
                        {'range': [66, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            st.plotly_chart(fig)
            
            # Explanation
            st.subheader("Prediction Explanation")
            st.markdown(prediction['explanation'])
    
    with tab3:
        st.header("Model Analysis")
        
        # Model performance comparison
        metrics = load_model_metrics()
        
        fig = px.bar(metrics, x='model', y=['log_loss', 'accuracy'],
                    barmode='group', title='Model Performance Comparison')
        st.plotly_chart(fig)
        
        # Feature importance
        importance = load_feature_importance()
        fig = px.bar(importance.sort_values('importance').tail(15),
                    x='importance', y='feature', orientation='h',
                    title='Top 15 Feature Importances')
        st.plotly_chart(fig)
```

---

## 5. ROBUSTNESS & UNCERTAINTY QUANTIFICATION

### 5.1 Monte Carlo Dropout

```python
# evaluation/uncertainty.py
import torch
import torch.nn as nn

class MCDropoutModel(nn.Module):
    """
    Neural network with Monte Carlo Dropout for uncertainty estimation
    """
    
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x, mc_dropout=True):
        x = F.relu(self.fc1(x))
        x = self.dropout1(x) if mc_dropout else x
        x = F.relu(self.fc2(x))
        x = self.dropout2(x) if mc_dropout else x
        return torch.sigmoid(self.fc3(x))
    
    def predict_with_uncertainty(self, X, n_samples=100):
        """
        Generate predictions with epistemic uncertainty
        """
        self.train()  # Keep dropout active
        
        predictions = []
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(X, mc_dropout=True)
                predictions.append(pred.cpu().numpy())
        
        predictions = np.array(predictions)
        
        return {
            'mean': predictions.mean(axis=0),
            'std': predictions.std(axis=0),
            'ci_lower': np.percentile(predictions, 2.5, axis=0),
            'ci_upper': np.percentile(predictions, 97.5, axis=0)
        }
```

### 5.2 Conformal Prediction

```python
# evaluation/conformal.py
from typing import Tuple
import numpy as np

class ConformalPredictor:
    """
    Conformal prediction for calibrated uncertainty intervals
    """
    
    def __init__(self, model, calibration_data: Tuple[np.ndarray, np.ndarray]):
        self.model = model
        self.calibration_scores = self._compute_calibration_scores(*calibration_data)
    
    def _compute_calibration_scores(self, X_cal, y_cal):
        """
        Compute non-conformity scores on calibration set
        """
        probs = self.model.predict_proba(X_cal)[:, 1]
        
        # Non-conformity score: |true - predicted|
        scores = np.abs(y_cal - probs)
        
        return scores
    
    def predict_with_interval(self, X, confidence=0.95):
        """
        Generate prediction intervals using conformal prediction
        """
        probs = self.model.predict_proba(X)[:, 1]
        
        # Find quantile for desired confidence
        alpha = 1 - confidence
        q = np.quantile(self.calibration_scores, 1 - alpha)
        
        intervals = []
        for prob in probs:
            lower = max(0, prob - q)
            upper = min(1, prob + q)
            intervals.append({'lower': lower, 'upper': upper, 'point': prob})
        
        return intervals
```

### 5.3 Sensitivity Analysis

```python
# evaluation/robustness.py

def feature_perturbation_test(model, X, y, feature_names, perturbation_scale=0.1):
    """
    Test model robustness to feature perturbations
    """
    baseline_preds = model.predict_proba(X)[:, 1]
    baseline_loss = log_loss(y, baseline_preds)
    
    sensitivity = {}
    
    for i, feature in enumerate(feature_names):
        # Add Gaussian noise to feature
        X_perturbed = X.copy()
        noise = np.random.normal(0, perturbation_scale * np.std(X[:, i]), X.shape[0])
        X_perturbed[:, i] += noise
        
        perturbed_preds = model.predict_proba(X_perturbed)[:, 1]
        perturbed_loss = log_loss(y, perturbed_preds)
        
        sensitivity[feature] = {
            'loss_increase': perturbed_loss - baseline_loss,
            'prediction_std': np.std(perturbed_preds - baseline_preds),
            'rank_correlation': spearmanr(baseline_preds, perturbed_preds)[0]
        }
    
    return sensitivity

def adversarial_robustness_test(model, X, y, epsilon=0.01):
    """
    Test robustness to adversarial perturbations (Fast Gradient Sign Method)
    """
    # Compute gradients (for gradient-based models)
    if hasattr(model, 'coef_'):
        # Linear model
        grad = np.sign(model.coef_)
        X_adv = X + epsilon * grad
    else:
        # Black-box: random perturbation in high-gradient direction
        X_adv = X + epsilon * np.random.randn(*X.shape)
    
    original_preds = model.predict_proba(X)[:, 1]
    adversarial_preds = model.predict_proba(X_adv)[:, 1]
    
    return {
        'prediction_change': np.mean(np.abs(original_preds - adversarial_preds)),
        'accuracy_drop': accuracy_score(y, original_preds > 0.5) - 
                        accuracy_score(y, adversarial_preds > 0.5)
    }
```

---

## 6. DATA PROCESSING & VALIDATION

### 6.1 Automated Data Quality Checks

```python
# data/validators.py
from dataclasses import dataclass
from typing import List, Dict
import pandas as pd

@dataclass
class ValidationResult:
    passed: bool
    message: str
    severity: str  # 'error', 'warning', 'info'

class DataValidator:
    """
    Comprehensive data validation pipeline
    """
    
    def __init__(self):
        self.checks = []
    
    def validate(self, data: pd.DataFrame) -> List[ValidationResult]:
        """
        Run all validation checks
        """
        results = []
        
        results.extend(self._check_missing_values(data))
        results.extend(self._check_data_types(data))
        results.extend(self._check_value_ranges(data))
        results.extend(self._check_consistency(data))
        results.extend(self._check_leakage(data))
        
        return results
    
    def _check_missing_values(self, data: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        missing_pct = data.isnull().mean()
        for col, pct in missing_pct.items():
            if pct > 0.5:
                results.append(ValidationResult(
                    passed=False,
                    message=f"Column {col} has {pct:.1%} missing values",
                    severity='error'
                ))
            elif pct > 0.1:
                results.append(ValidationResult(
                    passed=True,
                    message=f"Column {col} has {pct:.1%} missing values",
                    severity='warning'
                ))
        
        return results
    
    def _check_consistency(self, data: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        # Check game totals add up
        if 'home_score' in data.columns and 'away_score' in data.columns:
            total_games = len(data)
            consistent = (data['home_score'] + data['away_score'] == 
                         data['total_goals']).all()
            
            if not consistent:
                results.append(ValidationResult(
                    passed=False,
                    message="Score totals don't match individual scores",
                    severity='error'
                ))
        
        # Check Elo ratings are within reasonable bounds
        if 'elo' in data.columns:
            if (data['elo'] < 1000).any() or (data['elo'] > 2500).any():
                results.append(ValidationResult(
                    passed=False,
                    message="Elo ratings outside reasonable bounds [1000, 2500]",
                    severity='error'
                ))
        
        return results
    
    def _check_leakage(self, data: pd.DataFrame) -> List[ValidationResult]:
        results = []
        
        # Check for features that perfectly predict target
        if 'target' in data.columns:
            for col in data.columns:
                if col == 'target':
                    continue
                
                corr = data[col].corr(data['target'])
                if abs(corr) > 0.99:
                    results.append(ValidationResult(
                        passed=False,
                        message=f"Column {col} is perfectly correlated with target - likely leakage",
                        severity='error'
                    ))
        
        return results
```

### 6.2 Data Profiling

```python
# data/profiling.py
import pandas as pd
from ydata_profiling import ProfileReport

def generate_data_profile(data: pd.DataFrame, output_path: str):
    """
    Generate comprehensive data profiling report
    """
    profile = ProfileReport(
        data,
        title="WHL Data Profile",
        explorative=True,
        correlations={
            'pearson': {'calculate': True},
            'spearman': {'calculate': True},
            'kendall': {'calculate': True}
        }
    )
    
    profile.to_file(output_path)
    return profile

def detect_data_drift(new_data: pd.DataFrame, reference_data: pd.DataFrame):
    """
    Detect drift between new data and reference data
    """
    from scipy.stats import ks_2samp, chi2_contingency
    
    drift_report = {}
    
    for col in reference_data.columns:
        if reference_data[col].dtype in ['int64', 'float64']:
            # Kolmogorov-Smirnov test for continuous variables
            stat, p_value = ks_2samp(reference_data[col].dropna(), 
                                      new_data[col].dropna())
            drift_report[col] = {
                'test': 'ks',
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            }
        else:
            # Chi-square test for categorical variables
            contingency = pd.crosstab(reference_data[col], new_data[col])
            stat, p_value, _, _ = chi2_contingency(contingency)
            drift_report[col] = {
                'test': 'chi2',
                'statistic': stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05
            }
    
    return drift_report
```

---

## 7. VISUALIZATION & REPORTING

### 7.1 Advanced Visualizations

```python
# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Circle, FancyBboxPatch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def create_rating_evolution_plot(ratings_history, teams=None):
    """
    Animated Elo rating evolution over time
    """
    fig = go.Figure()
    
    if teams is None:
        teams = ratings_history['team'].unique()[:10]  # Top 10 for clarity
    
    for team in teams:
        team_data = ratings_history[ratings_history['team'] == team]
        fig.add_trace(go.Scatter(
            x=team_data['date'],
            y=team_data['rating'],
            mode='lines',
            name=team,
            line=dict(width=2)
        ))
    
    fig.update_layout(
        title='Elo Rating Evolution Over Time',
        xaxis_title='Date',
        yaxis_title='Elo Rating',
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig

def create_matchup_heatmap(matchup_matrix, teams):
    """
    Heatmap of win probabilities between all teams
    """
    fig = go.Figure(data=go.Heatmap(
        z=matchup_matrix,
        x=teams,
        y=teams,
        colorscale='RdYlGn',
        zmid=0.5,
        hoverongaps=False,
        text=matchup_matrix.round(2),
        texttemplate='%{text}',
        textfont={"size": 10}
    ))
    
    fig.update_layout(
        title='Tournament Matchup Win Probability Matrix',
        xaxis_title='Away Team',
        yaxis_title='Home Team',
        width=1000,
        height=1000
    )
    
    return fig

def create_calibration_dashboard(y_true, predictions_dict):
    """
    Multi-model calibration comparison
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (model_name, y_prob) in enumerate(predictions_dict.items()):
        ax = axes[idx // 3, idx % 3]
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
        
        ax.plot(prob_pred, prob_true, 's-', label=model_name)
        ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
        
        # Histogram of predicted probabilities
        ax2 = ax.twinx()
        ax2.hist(y_prob, range=(0, 1), bins=10, alpha=0.3, color='gray')
        ax2.set_ylabel('Count')
        
        ax.set_xlabel('Mean Predicted Probability')
        ax.set_ylabel('Fraction of Positives')
        ax.set_title(f'{model_name} Calibration')
        ax.legend()
    
    plt.tight_layout()
    return fig
```

### 7.2 Automated Report Generation

```python
# utils/reporting.py
from jinja2 import Template
import base64
from datetime import datetime

REPORT_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>WHL 2026 Analytics Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        .header { background: #1a5490; color: white; padding: 20px; }
        .section { margin: 30px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #1a5490; color: white; }
        .metric { display: inline-block; margin: 10px 20px; }
        .metric-value { font-size: 24px; font-weight: bold; color: #1a5490; }
        .chart { margin: 20px 0; text-align: center; }
    </style>
</head>
<body>
    <div class="header">
        <h1>WHL 2026 Analytics Report</h1>
        <p>Generated: {{ timestamp }}</p>
    </div>
    
    <div class="section">
        <h2>Executive Summary</h2>
        <div class="metric">
            <div class="metric-value">{{ metrics.holdout_log_loss }}</div>
            <div>Holdout Log Loss</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ metrics.holdout_accuracy }}%</div>
            <div>Holdout Accuracy</div>
        </div>
        <div class="metric">
            <div class="metric-value">{{ metrics.holdout_auc }}</div>
            <div>Holdout AUC</div>
        </div>
    </div>
    
    <div class="section">
        <h2>Power Rankings</h2>
        <table>
            <tr>
                <th>Rank</th>
                <th>Team</th>
                <th>Elo Rating</th>
                <th>Win %</th>
                <th>xG Diff/60</th>
                <th>Power Score</th>
            </tr>
            {% for team in power_rankings %}
            <tr>
                <td>{{ team.rank }}</td>
                <td>{{ team.name }}</td>
                <td>{{ team.elo }}</td>
                <td>{{ team.win_pct }}</td>
                <td>{{ team.xg_diff }}</td>
                <td>{{ team.power_score }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Tournament Predictions - Top Matchups</h2>
        <table>
            <tr>
                <th>Home Team</th>
                <th>Away Team</th>
                <th>Home Win %</th>
                <th>Predicted Winner</th>
                <th>Confidence</th>
            </tr>
            {% for matchup in top_matchups %}
            <tr>
                <td>{{ matchup.home_team }}</td>
                <td>{{ matchup.away_team }}</td>
                <td>{{ matchup.home_win_prob }}</td>
                <td>{{ matchup.predicted_winner }}</td>
                <td>{{ matchup.confidence }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
    
    <div class="section">
        <h2>Visualizations</h2>
        {% for chart in charts %}
        <div class="chart">
            <h3>{{ chart.title }}</h3>
            <img src="data:image/png;base64,{{ chart.image }}" style="max-width: 100%;"/>
        </div>
        {% endfor %}
    </div>
    
    <div class="section">
        <h2>Model Performance</h2>
        <table>
            <tr>
                <th>Model</th>
                <th>Log Loss</th>
                <th>Accuracy</th>
                <th>AUC</th>
                <th>Status</th>
            </tr>
            {% for model in model_performance %}
            <tr>
                <td>{{ model.name }}</td>
                <td>{{ model.log_loss }}</td>
                <td>{{ model.accuracy }}</td>
                <td>{{ model.auc }}</td>
                <td>{{ model.status }}</td>
            </tr>
            {% endfor %}
        </table>
    </div>
</body>
</html>
"""

def generate_html_report(data, output_path):
    """
    Generate comprehensive HTML report
    """
    template = Template(REPORT_TEMPLATE)
    
    # Convert charts to base64
    charts_base64 = []
    for chart in data['charts']:
        import io
        buffer = io.BytesIO()
        chart.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode()
        charts_base64.append({'title': chart.get_title(), 'image': image_base64})
    
    html = template.render(
        timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        metrics=data['metrics'],
        power_rankings=data['power_rankings'],
        top_matchups=data['top_matchups'],
        charts=charts_base64,
        model_performance=data['model_performance']
    )
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path
```

---

## 8. PERFORMANCE OPTIMIZATION

### 8.1 Parallel Processing

```python
# utils/parallel.py
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp

def parallel_model_training(models, X_train, y_train, n_jobs=-1):
    """
    Train multiple models in parallel
    """
    if n_jobs == -1:
        n_jobs = mp.cpu_count()
    
    def train_single_model(model_data):
        name, model = model_data
        model.fit(X_train, y_train)
        return name, model
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(train_single_model, models.items()))
    
    return dict(results)

def parallel_permutation_search(weight_library, oof_preds, y_true, n_jobs=-1):
    """
    Parallel evaluation of blend permutations
    """
    def evaluate_blend(weights):
        blended = np.average(oof_preds, axis=0, weights=weights)
        return log_loss(y_true, blended)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(evaluate_blend, weight_library))
    
    return results
```

### 8.2 Memory Optimization

```python
# utils/memory.py
import pandas as pd
import numpy as np

def optimize_dataframe_memory(df):
    """
    Reduce memory usage of DataFrame
    """
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory reduced from {start_mem:.2f} MB to {end_mem:.2f} MB')
    
    return df

class DataGenerator:
    """
    Generator for large datasets to avoid loading everything into memory
    """
    
    def __init__(self, file_path, batch_size=10000):
        self.file_path = file_path
        self.batch_size = batch_size
    
    def __iter__(self):
        for chunk in pd.read_csv(self.file_path, chunksize=self.batch_size):
            yield chunk
```

### 8.3 Caching

```python
# utils/caching.py
import hashlib
import joblib
from functools import wraps
from pathlib import Path

def cache_result(cache_dir='.cache'):
    """
    Decorator to cache function results
    """
    Path(cache_dir).mkdir(exist_ok=True)
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            key = hashlib.md5(
                str((func.__name__, args, kwargs)).encode()
            ).hexdigest()
            
            cache_path = Path(cache_dir) / f'{key}.joblib'
            
            if cache_path.exists():
                return joblib.load(cache_path)
            
            result = func(*args, **kwargs)
            joblib.dump(result, cache_path)
            return result
        
        return wrapper
    return decorator

# Usage
@cache_result(cache_dir='.cache/features')
def build_features(data_path, config):
    # Expensive feature engineering
    return features
```

---

## 9. PRIORITIZATION ROADMAP

### Phase 1: Quick Wins (Week 1-2) - Expected Improvement: 5-10%

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 1 | Add CatBoost & LightGBM models | Low | High |
| 2 | Implement temperature scaling calibration | Low | Medium |
| 3 | Add momentum/trend features | Low | High |
| 4 | Create comprehensive test suite | Medium | Medium |
| 5 | Add SHAP explanations | Low | High |

### Phase 2: Core Improvements (Week 3-4) - Expected Improvement: 10-15%

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 6 | Implement stacking ensemble | Medium | High |
| 7 | Add team embeddings | Medium | High |
| 8 | Feature interaction engineering | Medium | Medium |
| 9 | Model architecture refactoring | High | Medium |
| 10 | Interactive dashboard | Medium | Low |

### Phase 3: Advanced Features (Week 5-6) - Expected Improvement: 5-10%

| Priority | Task | Effort | Impact |
|----------|------|--------|--------|
| 11 | Monte Carlo Dropout uncertainty | Medium | Medium |
| 12 | Conformal prediction intervals | Medium | Medium |
| 13 | Advanced feature selection | Medium | Medium |
| 14 | Automated data validation | Low | Medium |
| 15 | Performance optimization | Medium | Low |

### Implementation Priority Matrix

```
            High Impact
                ▲
                │
    Stacking ◄──┼──► CatBoost/LightGBM
    Ensemble    │      Team Embeddings
                │
    ────────────┼─────────────► Low Effort
                │
    Interactive │      Automated
    Dashboard   │      Reporting
                │
                ▼
            Low Impact
```

---

## 10. SUCCESS METRICS

### Performance Targets

| Metric | Current | Phase 1 Target | Phase 2 Target | Phase 3 Target |
|--------|---------|----------------|----------------|----------------|
| Holdout Log Loss | 0.6696 | 0.6650 | 0.6550 | 0.6400 |
| Accuracy | 57.9% | 59.0% | 61.0% | 63.0% |
| AUC | 0.5645 | 0.5750 | 0.5900 | 0.6000 |
| Calibration Error | ~0.05 | 0.04 | 0.03 | 0.02 |
| Inference Time | ~100ms | ~80ms | ~60ms | ~50ms |

### Quality Metrics

- **Test Coverage**: Target 80%+ code coverage
- **Documentation**: All public APIs documented
- **Type Hints**: 100% type coverage with mypy
- **Data Leakage**: Zero tolerance - automated checks
- **Overfitting**: Max gap of 0.03 between train/test

---

## Summary

This improvement plan targets **~4.4% log loss reduction** (0.6696 → 0.6400) through:

1. **Algorithm Diversity**: CatBoost, LightGBM, stacking ensembles
2. **Feature Depth**: Temporal patterns, interactions, embeddings
3. **Uncertainty Quantification**: Monte Carlo dropout, conformal prediction
4. **Code Quality**: Modular architecture, comprehensive testing
5. **Interpretability**: SHAP values, natural language explanations

All improvements maintain the **leakage-safe, overfit-controlled** principles that make the current system robust.