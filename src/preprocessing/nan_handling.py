import numpy as np

def handle_nans(X, method='impute', verbose=True):
    n_nans = np.isnan(X).sum()
    
    if n_nans == 0:
        if verbose:
            print("   ✓ No NaN detected")
        return X
    
    if verbose:
        n_nan_features = np.isnan(X).any(axis=0).sum()
        n_allnan_features = np.isnan(X).all(axis=0).sum()
        print(f"   Detected {n_nans} NaN ({100*n_nans/X.size:.2f}%)")
        print(f"   Affected {n_nan_features}/{X.shape} features")
        if n_allnan_features > 0:
            print(f"  All-NaN: {n_allnan_features} features")
    
    if method == 'impute':
        X_clean = X.copy()
        for j in range(X.shape):
            col = X[:, j]
            if np.isnan(col).all():
                X_clean[:, j] = 0.0
            elif np.isnan(col).any():
                X_clean[:, j] = np.where(np.isnan(col), np.nanmean(col), col)
        if verbose:
            print(f"   ✓ NaN handled (mean imputation)")
    elif method == 'zero':
        X_clean = np.nan_to_num(X, nan=0.0)
        if verbose:
            print(f"   ✓ NaN replaced with 0")
    else:
        raise ValueError(f"Unknown method: {method}")
    
    assert X_clean.shape == X.shape
    assert not np.isnan(X_clean).any()
    
    return X_clean
