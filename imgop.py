import numpy as np
import imageio
from im2patch import im2patch, patch2im
from im2patch import  spclustering, paclustering, nearestcnt
from colabfilter import colabfilter, Gdenoise
import matplotlib.pyplot as plt
img=imageio.imread('cameraman.png');
imgcl=np.zeros(img.shape)
sigma=30
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
plt.figure(1)
w1=plt.subplot(121)
w1.set_title('Noisy image, PSNR=%.2f'%(psnr1))
plt.imshow(imgn,cmap='gray')
w2=plt.subplot(122)
w2.set_title('Denoised Image, PSNR=%.2f' %(psnr2))
plt.imshow(rimg,cmap='gray')
plt.show()
