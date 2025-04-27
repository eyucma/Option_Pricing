import numpy as np


def sq(x):
  t=x**2
  if x.ndim>1:
    return t.sum(axis=1)
  else:
    return t

class CL:
    def __init__(self,nodes=None,d=2,M=40):
        if np.any(nodes):
            self.nodes=nodes
            self.d=nodes.shape[1]
        else:
            self.nodes=np.random.uniform(size=(M,d))
            self.d=d
        self.m=len(self.nodes)
    def train(self,x,n=10000,eta=0.1):
        for _ in range(n):
            inp=x[np.random.randint(len(x))]
            ind=np.argmin(sq(inp-self.nodes))
            self.nodes[ind]=(1-eta)*self.nodes[ind]+eta*inp  
    def train_sched(self,x,n=10000,eta=0.1,nu=0.05,T=10,f=10):
        for _ in range(n):
            inp=x[np.random.randint(len(x))]
            ind=np.argmin(sq(inp-self.nodes))
            self.nodes[ind]=(1-eta)*self.nodes[ind]+eta*inp
            if _%f==0:
                u=np.random.uniform(size=len(self.nodes))  
                if self.d>=2:
                    dn=np.einsum('i,ij->ij',u,(inp-self.nodes))
                else:
                    dn=u*(inp-self.nodes)
                self.nodes=self.nodes+nu*np.exp(-_/T)*dn
    def agg(self,x,y):
        cluster=np.zeros(len(x))
        for i in range(len(x)):
            cluster[i]=np.argmin(sq(x[i]-self.nodes))
        self.val=np.zeros(self.m)
        for i in range(self.m):
            l=[y[j] for j in range(len(x)) if cluster[j]==i]
            if len(l)>0:
                self.val[i]=sum(l)/len(l)

