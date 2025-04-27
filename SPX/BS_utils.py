import numpy as np
from scipy.stats import norm

def BS(S=35,K=35,r=0.05,vol=0.2,T=0.5,q=0):
    dp=1/vol/np.sqrt(T)*(np.log(S/K)+(r-q+vol**2/2)*T)
    dm=dp-vol*np.sqrt(T)
    C=S*norm.cdf(dp)*np.exp(-q*T)-K*np.exp(-r*T)*norm.cdf(dm)
    return C,C-S*np.exp(-q*T)+K*np.exp(-r*T)

def BS_Vega(S=35,K=35,r=0.05,vol=0.2,T=0.5,q=0):
    dp=1/vol/np.sqrt(T)*(np.log(S/K)+(r+vol**2/2)*T)
    return S*np.exp(-(dp)**2/2)/np.sqrt(np.pi*2/T)*np.exp(-q*T)

def log_BS_Vega(S=35,K=35,r=0.05,vol=0.2,T=0.5,q=0):
    dp=1/vol/np.sqrt(T)*(np.log(S/K)+(r+vol**2/2)*T)
    return np.log(S)-(dp)**2/2-0.5*np.log(np.pi*2/T)-q*T

def imp_vol_BS(P=2,S=35,K=35,r=0.05,T=0.5,method='Newton',tol=1E-8,max_step=10,q=0,lb=1E-7):
    if method=='Newton':
        l=0
        money=S*np.exp((r-q)*T)/K
        vol0=np.sqrt(2*np.abs(np.log(money))/T)
        C,_=BS(S=S,K=K,r=r,vol=vol0,T=T,q=q)
        step=np.sign(C-P)*np.exp(np.log(np.abs(C-P))-log_BS_Vega(S=S,K=K,r=0.04,vol=vol0,T=T,q=q))
        if np.abs(step)>max_step:
            step=np.sign(step)*max_step
        vol1=np.maximum(vol0-step,lb)
        while np.abs(vol1-vol0)>tol:
            vol0=vol1
            C,_=BS(S=S,K=K,r=r,vol=vol0,T=T,q=q)
            step=np.sign(C-P)*np.exp(np.log(np.abs(C-P))-log_BS_Vega(S=S,K=K,r=0.04,vol=vol0,T=T,q=q))
            if np.abs(step)>max_step:
                step=np.sign(step)*max_step
            vol1=np.maximum(vol0-step,lb)
            l=l+1
            if l%100==0:
                max_step=max_step/2
        return vol1

def imp_S_BS(P=2,K=35,r=0.05,vol=0.2,T=0.5,method='Newton',tol=1E-8,max_step=100,q=0):
    if method=='Newton':
        l=0
        mins=P
        maxs=np.exp(q*T)*P+K*np.exp(-(r-q)*T)
        assert mins <= maxs
        S0=(mins+maxs)/2
        C,_=BS(S=S0,K=K,r=r,vol=vol,T=T,q=q)
        step=(C-P)/BS_Delta(S0,K,r,vol,T,q)
        if np.abs(step)>max_step:
            step=np.sign(step)*max_step
        S1=np.maximum(S0-step,mins)
        S1=np.minimum(maxs,S1)
        while np.abs(S1-S0)>tol:
            S0=S1
            C,_=BS(S=S0,K=K,r=r,vol=vol,T=T,q=q)
            step=(C-P)/BS_Delta(S0,K,r,vol,T,q)
            if np.abs(step)>max_step:
                step=np.sign(step)*max_step
            S1=np.maximum(S0-step,mins)
            S1=np.minimum(maxs,S1)
            l=l+1
            if l%100==0:
                print(l,S0,BS_Delta(S0,K,r,vol,T,q))
                max_step=max_step/2
        return S1
    
def BS_Delta(S=35,K=35,r=0.05,vol=0.2,T=0.5,q=0,option='call'):
    dp=1/vol/np.sqrt(T)*(np.log(S/K)+(r-q+vol**2/2)*T)
    if option=='call':
        return norm.cdf(dp)*np.exp(-q*T)
    else:
        return (norm.cdf(dp)-1)*np.exp(-q*T)
    
def log_abs_BS_Delta(S=35,K=35,r=0.05,vol=0.2,T=0.5,q=0,option='call'):
    dp=1/vol/np.sqrt(T)*(np.log(S/K)+(r-q+vol**2/2)*T)
    if option=='call':
        return np.log(norm.cdf(dp))-q*T
    else:
        return np.log(norm.cdf(-dp))-q*T
    
def dvolds(S=35,K=35,r=0.05,vol=0.2,T=0.5,q=0,option='call'):
    if option=='call':
        return -np.exp(log_abs_BS_Delta(S,K,r,vol,T,q,option)-log_BS_Vega(S,K,r,vol,T,q))
    else:
        return np.exp(log_abs_BS_Delta(S,K,r,vol,T,q,option)-log_BS_Vega(S,K,r,vol,T,q))
    
def manual_d(P=2,S=35,K=35,r=0.05,T=0.5,q=0,ds=1,lb=1E-20):
    if S > np.exp(q*T)*P+K*np.exp(-(r-q)*T):
        ds=-(S-np.exp(q*T)*P+K*np.exp(-(r-q)*T))-1
    return (imp_vol_BS(P,S+ds,K,r,T,q=q,lb=lb)-imp_vol_BS(P,S,K,r,T,q=q,lb=lb))/ds
