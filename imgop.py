import numpy as np
from scipy import misc
from im2patch import im2patch, patch2im
from im2patch import  spclustering, paclustering, nearestcnt
from colabfilter import colabfilter, Gdenoise
import matplotlib.pyplot as plt
import pdb
img=misc.imread('cameraman.png');
imgcl=np.zeros(img.shape)
sigma=20
s=8
#paidx:patch idx, cpix: centeral pixel of patch
imgn=img+sigma*np.random.normal(size=img.shape)
[npa,paidx,cpidx]=im2patch(imgn,s)
cln=spclustering(paidx,imgn.shape)
dpa=npa
for i in range(1):
    cln=paclustering(dpa,cpidx,cln,5)
    dpa=colabfilter(npa,dpa,cln,sigma)
    
rimg=patch2im(imgn.shape,dpa,paidx,s)
psnr1=10*np.log10(255**2/np.mean((imgn-img)**2))
psnr2=10*np.log10(255**2/np.mean((rimg-img)**2))
print(psnr1,psnr2)
pdb.set_trace()
plt.imshow(rimg,cmap='gray')
plt.show()
