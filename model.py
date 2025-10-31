# model.py
# Utilità semplici per addestrare un modello regressore per prova/educazione.
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

def train_model(X_train, y_train, X_val, y_val):
    if XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
    else:
        model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    mse = mean_squared_error(y_val, preds)
    r2 = r2_score(y_val, preds)
    return model, {'mse': float(mse), 'r2': float(r2)}

def predict_next(model, feats: np.ndarray):
    return model.predict(feats)