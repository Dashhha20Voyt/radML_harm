import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.covariance import LedoitWolf
from scipy.linalg import logm, expm, sqrtm

def riemannian_harmonization(X, batch, covars=None, pca_k=60, n_iter=2):
    n, p = X.shape
    k = min(pca_k, p, n - 2)
    
    print(f"   Riemannian: PCA {p} → {k} components")
    
    pca = PCA(n_components=k, random_state=0)
    Xp = pca.fit_transform(X)
    
    if covars is not None:
        covar_pred = np.zeros_like(Xp)
        for j in range(k):
            lr = LinearRegression()
            lr.fit(covars, Xp[:, j])
            covar_pred[:, j] = lr.predict(covars)
        Xp_resid = Xp - covar_pred
    else:
        Xp_resid = Xp.copy()
        covar_pred = np.zeros_like(Xp)
    
    Xp_aligned = Xp_resid.copy()
    unique_batches = np.unique(batch)
    
    for it in range(n_iter):
        mus = []
        Sigmas = []
        
        for b in unique_batches:
            mask = (batch == b)
            Xb = Xp_aligned[mask]
            
            if Xb.shape < 2:
                mus.append(np.zeros(k))
                Sigmas.append(np.eye(k))
            else:
                mu_b = Xb.mean(axis=0)
                Sigma_b = LedoitWolf().fit(Xb).covariance_
                mus.append(mu_b)
                Sigmas.append(Sigma_b)
        
        logs = [logm(S) for S in Sigmas]
        Lbar = sum(logs) / len(logs)
        Sigma_ref = expm(Lbar)
        
        try:
            Sref_sqrt = sqrtm(Sigma_ref)
        except Exception:
            Sref_sqrt = np.eye(k)
        
        A_ops = []
        for S in Sigmas:
            try:
                S_sqrt = sqrtm(S)
                S_inv = np.linalg.pinv(S_sqrt, rcond=1e-8)
                A_ops.append(Sref_sqrt @ S_inv)
            except Exception:
                A_ops.append(np.eye(k))
        
        mu_ref = np.mean(mus, axis=0)
        
        Xp_new_resid = np.zeros_like(Xp_aligned)
        for i, b in enumerate(unique_batches):
            mask = (batch == b)
            Xp_new_resid[mask] = mu_ref + (Xp_aligned[mask] - mus[i]) @ A_ops[i].T
        
        Xp_aligned = Xp_new_resid + covar_pred
    
    X_harmonized = pca.inverse_transform(Xp_aligned)
    
    return X_harmonized
