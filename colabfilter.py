import numpy as np
import pdb
def colabfilter(npa,prepa,cln,sigma):
    cln=cln.flatten()
    numcl=np.max(cln).astype('int')
    dpa=np.zeros(npa.shape)
    for i in range(numcl):
        npac=npa[:,cln==i]
        prepac=prepa[:,cln==i]
    #    dpac=Gdenoise(npac,prepac,sigma)
        dpac=svddenoise(npac,sigma,30)
        dpa[:,cln==i]=dpac
    return dpa


def Gdenoise(npa,estpa,sigma):
    s=npa.shape[0]
    meanc=np.mean(estpa,axis=1)
    covc=computecov(estpa)+.0001*np.eye(s)
    invcovc=np.linalg.inv(covc)
    Wp=np.linalg.inv(invcovc+1/(sigma**2)*np.eye(s))
    Mcr=np.tile(invcovc.dot(meanc),(npa.shape[1],1)).T
    dpac=Wp.dot(npa/(sigma**2)+Mcr)
    return dpac

def svddenoise(npa,sigma,tre):
    dim0=npa.shape[0]
    dim1=npa.shape[1]
    U, s, Vh=np.linalg.svd(npa)
    stre=s
    stre[stre<tre*sigma]=0;
    smat=np.zeros((dim0,dim1))
    smat[:dim0,:dim0]=np.diag(stre)
    dpa=np.dot(U,np.dot(smat,Vh))
    return dpa
    
def computecov(data):
    s=data.shape[0]
    acc=np.zeros((s,s))
    mudata=np.mean(data,axis=1)
    for i in range(data.shape[1]):
        veccor=np.outer(data[:,i]-mudata,data[:,i]-mudata)
        acc=acc+veccor
    return acc/data.shape[1]
