import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

def bart_harmonize(X, batch, covars=None, n_estimators=50, max_depth=10, n_jobs=-1):
    """
    Random Forest-based harmonization.
    
    Parameters
    ----------
    X : np.ndarray
        Input data
    batch : np.ndarray
        Batch labels
    covars : np.ndarray, optional
        Covariates
    n_estimators : int, default=50
        Number of trees
    max_depth : int, default=10
        Max tree depth
    n_jobs : int, default=-1
        Parallel jobs
    
    Returns
    -------
    X_harmonized : np.ndarray
        Harmonized data
    """
    batch_oh = OneHotEncoder(sparse_output=False).fit_transform(batch.reshape(-1, 1))
    
    if covars is not None:
        Xcov = np.column_stack([batch_oh, covars])
    else:
        Xcov = batch_oh
    
    X_out = np.zeros_like(X)
    p = X.shape
    
    print(f"   BART/RF: processing {p} features...")
    for j in tqdm(range(p), desc="   BART/RF", leave=False):
        yj = X[:, j]
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=n_jobs,
            random_state=0
        )
        model.fit(Xcov, yj)
        
        if covars is not None:
            neutral = np.zeros((Xcov.shape, Xcov.shape))
            neutral[:, :batch_oh.shape] = batch_oh
        else:
            neutral = Xcov
        
        pred_batch = model.predict(neutral)
        X_out[:, j] = yj - (pred_batch - pred_batch.mean())
    
    return X_out
