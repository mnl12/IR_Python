import numpy as np
import pdb
def im2patch(im,s):
    k=0;
    psize=(s*s,(im.shape[0]-s+1)*(im.shape[1]-s+1))
    disfc=np.floor(s/2)
    pa=np.zeros(psize)
    paidx=np.zeros((2,(im.shape[0]-s+1)*(im.shape[1]-s+1)))
    for i in range(im.shape[0]-s+1):
        for j in range(im.shape[1]-s+1):
            panr=im[i:i+s,j:j+s]
            pa[:,k]=panr.reshape(s*s)
            paidx[:,k]=[i,j]
            k=k+1
    cpidxf=paidx+disfc
    cpidx=cpidxf.astype('int')
    return pa,paidx,cpidx

def patch2im(sizeim,pa,paidx,s):
    im=np.zeros(sizeim)
    imw=np.zeros(sizeim)
    paidx=paidx.astype('int')
    for i in range(paidx.shape[1]):
        pasq=pa[:,i].reshape([s,s])
        idx=paidx[:,i]
        im[idx[0]:idx[0]+s,idx[1]:idx[1]+s]=im[idx[0]:idx[0]+s,idx[1]:idx[1]+s]+pasq
        imw[idx[0]:idx[0]+s,idx[1]:idx[1]+s]=imw[idx[0]:idx[0]+s,idx[1]:idx[1]+s]+1
    reconimage=im/imw
    return reconimage      

def paclustering(pa,cpidx,cln,nnn):
    [pacnt,idxcnt]=cmpclctr(pa,cpidx,cln)
    nncl=nearestcnt(cpidx,idxcnt,nnn)
    nncl=nncl.astype('int')
    for i in range(pa.shape[1]):
        nbrep=nncl.shape[0]
        selpac=pacnt[:,nncl[:,i].flatten()]
        Matpa=np.tile(pa[:,i:i+1],(1,nbrep))
        distant=np.linalg.norm(Matpa-selpac,axis=0)
        argcl=np.argmin(distant)
        cln[i]=nncl[argcl,i]
    return cln

def nearestcnt(idx,idxcnt,nnn):
    nncl=np.zeros((nnn,idx.shape[1]))
    for i in range(idx.shape[1]):
        nbrep=idxcnt.shape[1]
        Matidx=np.tile(idx[:,i:i+1],(1,nbrep))
        distant=np.linalg.norm(Matidx-idxcnt,axis=0)
        clnnn=np.argpartition(distant,nnn)[:nnn]
        nncl[:,i]=clnnn
    return nncl
    

def cmpclctr(pa,cpaidx,cln):
    ncl=max(cln).astype('int')
    cln=cln.flatten()
    pacenter=np.zeros((pa.shape[0],ncl[0]))
    idxcenter=np.zeros((cpaidx.shape[0],ncl[0]))
    for i in range(ncl[0]):
        pacenter[:,i]=np.mean(pa[:,cln==i],axis=1)
        idxcenter[:,i]=np.mean(cpaidx[:,cln==i],axis=1)
    return pacenter,idxcenter    
    

def spclustering(idx,sizeim):
    a1=np.arange(1,sizeim[0],20)
    a2=np.arange(1,sizeim[1],20)
    clidx=np.array(np.meshgrid(a1,a2)).T.reshape(-1,2).T
    cln=np.zeros([idx.shape[1],1])
    for i in range(idx.shape[1]):
        nbrep=clidx.shape[1]
        Matidx=np.tile(idx[:,i:i+1],(1,nbrep))
        distant=np.linalg.norm(Matidx-clidx,axis=0)
        cln[i,0]=np.argmin(distant)
    return cln

    
