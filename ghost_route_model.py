from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

def train_ghost_model(X_train, y_train):
    """
    Trains the Ghost Route Algorithm to predict route profitability.
    """
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Benchmark against Linear Regression baseline
    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"Model RMSE: {rmse}") 
    return model