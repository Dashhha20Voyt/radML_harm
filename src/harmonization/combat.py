import numpy as np

def combat_robust(X, batch, covars=None, eps=1e-3):
    """
    Robust ComBat harmonization with fallback on NaN.
    
    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Input data matrix
    batch : np.ndarray, shape (n_samples,)
        Batch labels
    covars : np.ndarray, optional
        Covariates to preserve
    eps : float, default=1e-3
        Regularization parameter
    
    Returns
    -------
    X_harmonized : np.ndarray
        Harmonized data matrix
    """
    X = X.copy()
    X_original = X.copy()
    n, p = X.shape
    batch = np.asarray(batch)
    batches = np.unique(batch)
    B = len(batches)
    
    if covars is None:
        design = np.ones((n, 1))
    else:
        covars = np.asarray(covars)
        design = np.column_stack([np.ones(n), covars])
    
    inv_dd = np.linalg.inv(design.T @ design + eps * np.eye(design.shape))
    beta_hat = inv_dd @ design.T @ X
    fitted = design @ beta_hat
    res = X - fitted
    
    s_data = np.zeros((B, p))
    n_batch = np.zeros(B, dtype=int)
    for i, b in enumerate(batches):
        mask = (batch == b)
        n_b = mask.sum()
        n_batch[i] = n_b
        s_data[i, :] = res[mask, :].mean(axis=0)
    
    var_pooled = np.zeros(p)
    for j in range(p):
        ss = 0.0
        for i in range(B):
            mask = (batch == batches[i])
            ss += ((res[mask, j] - s_data[i, j])**2).sum()
        var_pooled[j] = ss / (n - B)
        if var_pooled[j] < eps:
            var_pooled[j] = res[:, j].var() + eps
    
    s_data_full = np.zeros_like(res)
    for i, b in enumerate(batches):
        mask = (batch == b)
        s_data_full[mask, :] = s_data[i, :]
    
    stand = (res - s_data_full) / np.sqrt(var_pooled + eps)
    
    if np.isnan(stand).any():
        print(f" ComBat: NaN detected → fallback")
        return X_original
    
    gamma_hat = s_data
    delta_hat = np.zeros_like(gamma_hat)
    for i, b in enumerate(batches):
        mask = (batch == b)
        delta_hat[i, :] = res[mask, :].var(axis=0, ddof=1) + eps
    
    gamma_bar = gamma_hat.mean(axis=0)
    t2 = gamma_hat.var(axis=0, ddof=1) + eps
    
    a_prior = (2 * (delta_hat.mean(axis=0)**2)) / (delta_hat.var(axis=0, ddof=1) + eps) + 2
    b_prior = (delta_hat.mean(axis=0) * (a_prior - 1))
    
    gamma_star = np.zeros_like(gamma_hat)
    delta_star = np.zeros_like(delta_hat)
    
    for i in range(B):
        n_b = n_batch[i]
        gamma_star[i, :] = (t2 * n_b * gamma_hat[i, :] + var_pooled * gamma_bar) / (t2 * n_b + var_pooled + eps)
        a = a_prior
        b = b_prior
        d = (n_b - 1) * delta_hat[i, :]
        delta_star[i, :] = (d + 2*b) / (n_b + 2*a + 2 + eps)
        delta_star[i, :] = np.maximum(delta_star[i, :], eps)
    
    stand_adj = np.zeros_like(stand)
    for i, b in enumerate(batches):
        mask = (batch == b)
        stand_adj[mask, :] = (stand[mask, :] - gamma_star[i, :]) / np.sqrt(delta_star[i, :] + eps)
    
    X_harmonized = stand_adj * np.sqrt(var_pooled + eps) + fitted
    
    if np.isnan(X_harmonized).any():
        print(f"   ComBat: NaN in result → fallback")
        return X_original
    
    return X_harmonized
