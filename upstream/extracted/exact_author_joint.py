"""Exact shared-prompt Gaussian posteriors under author uniqueness.

Original Gaussian model and whitening: prvsiyan/public_gaussian.py.
Our inference replaces factorial assignment enumeration and large-group
mean-field updates with subset dynamic programming (at most2**18 states).

logp(a|U) = sum_i -.5||U_i-m[a_i]||^2
            +.5 sum_d gamma[d]*(sum_i U_i[d]-sum_i m[a_i,d])^2.
For unique authors, the second term depends only on the selected author set,
so assignment permutations can be summed exactly with subset DP.
"""
from functools import lru_cache
import numpy as np
from scipy.special import logsumexp

@lru_cache(maxsize=4)
def _levels(k):
    masks=np.arange(1<<k,dtype=np.int64)
    counts=np.fromiter((int(x).bit_count() for x in masks),dtype=np.uint8,count=len(masks))
    return tuple(masks[counts==n] for n in range(k+1))

class ExactAuthorPosterior:
    def __init__(self,means,prompt_variances):
        self.m=np.asarray(means,dtype=np.float64);self.lam=np.asarray(prompt_variances,dtype=np.float64)
        self.k,self.d=self.m.shape
        assert 1<=self.k<=18 and self.lam.shape==(self.d,) and (self.lam>=0).all()
        assert np.isfinite(self.m).all() and np.isfinite(self.lam).all()
        self.levels=_levels(self.k);self.subset_means={}
    def group(self,U,temperature=1.,extra_log_scores=None,model_weight=1.,extra_weight=1.):
        U=np.asarray(U,dtype=np.float64);n=len(U)
        assert U.shape==(n,self.d) and 1<=n<=self.k and temperature>0 and np.isfinite(U).all()
        assert model_weight>=0 and extra_weight>=0
        extra=None if extra_log_scores is None else np.asarray(extra_log_scores,dtype=np.float64)
        if extra is not None:assert extra.shape==(n,self.k) and np.isfinite(extra).all()
        if n==1:
            value=-.5*((U[:,None,:]-self.m[None,:,:])**2/(1+self.lam)).sum(2)/temperature
            value*=model_weight
            if extra is not None:value+=extra_weight*extra
            return value-logsumexp(value,axis=1,keepdims=True)
        if n not in self.subset_means:
            masks=self.levels[n]
            membership=((masks[:,None]>>np.arange(self.k))&1).astype(np.float64)
            self.subset_means[n]=membership@self.m
        terminal=self.levels[n];gamma=self.lam/(1+n*self.lam)
        residual=U.sum(0)[None,:]-self.subset_means[n]
        correction=.5*np.einsum('nd,d,nd->n',residual,gamma,residual)/temperature
        correction*=model_weight
        correction-=correction.max()
        A=-.5*((U[:,None,:]-self.m[None,:,:])**2).sum(2)/temperature
        A*=model_weight
        if extra is not None:A+=extra_weight*extra
        A-=A.max(1,keepdims=True)
        forward=np.full(1<<self.k,-np.inf);forward[0]=0.
        for count in range(1,n+1):
            masks=self.levels[count]
            for author in range(self.k):
                bit=1<<author;chosen=masks[(masks&bit)!=0]
                forward[chosen]=np.logaddexp(forward[chosen],forward[chosen^bit]+A[count-1,author])
        backward=np.full_like(forward,-np.inf);backward[terminal]=correction
        for count in range(n-1,-1,-1):
            masks=self.levels[count]
            for author in range(self.k):
                bit=1<<author;chosen=masks[(masks&bit)==0]
                backward[chosen]=np.logaddexp(backward[chosen],A[count,author]+backward[chosen|bit])
        assert np.isfinite(backward[0])
        result=np.empty((n,self.k))
        for row in range(n):
            masks=self.levels[row]
            for author in range(self.k):
                bit=1<<author;chosen=masks[(masks&bit)==0]
                result[row,author]=logsumexp(forward[chosen]+A[row,author]+backward[chosen|bit])-backward[0]
        # Uniqueness is global, but each row's posterior must still normalise.
        assert np.max(np.abs(logsumexp(result,axis=1)))<1e-8
        result-=logsumexp(result,axis=1,keepdims=True)
        return result

def scores(U,groups,prm,extra_log_scores=None,model_weight=1.,extra_weight=1.):
    U=np.asarray(U);groups=np.asarray(groups);result=np.empty((len(U),len(prm['m'])))
    solver=ExactAuthorPosterior(prm['m'],prm['lam'])
    for group in np.unique(groups):
        rows=np.flatnonzero(groups==group)
        result[rows]=solver.group(U[rows],float(prm['T1'] if len(rows)==1 else prm['Tg']),extra_log_scores=None if extra_log_scores is None else extra_log_scores[rows],model_weight=model_weight,extra_weight=extra_weight)
    return result
