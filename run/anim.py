#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 23:33:00 2018

@author: ogurcan
"""
import sys
import os
import shutil
import numpy as np
import h5py as h5
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pylab as plt
from mpi4py import MPI
plt.rcParams['text.usetex'] = True
comm=MPI.COMM_WORLD
infl="out.h5"
outfl="out.mp4"

vminom=-10.0
vmaxom=10.0
vminn=-10.0
vmaxn=10.0

fl=h5.File(infl,"r",libver='latest',swmr=True)
uk=fl['fields/uk']
Lx=fl['params/Lx'][()]
Ly=fl['params/Ly'][()]
kx=fl['data/kx'][()]
ky=fl['data/ky'][()]
Npx=fl['params/Npx'][()]
Npy=fl['params/Npy'][()]
nu=fl['params/nu'][()]
C=fl['params/C'][()]
kap=fl['params/kap'][()]
ksqr=kx**2+ky**2

Nx=uk.shape[2]
Ny=(uk.shape[3]*2-2)
dx=Lx/Nx;
dy=Ly/Ny;
x,y=np.meshgrid(np.arange(0,Nx)*dx,np.arange(0,Ny)*dy,indexing='ij')
w, h = 9.6,5.4
#ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=2)
fig,ax=plt.subplots(1,2,sharey=True,figsize=(w,h))
qd=[]
u0=np.fft.irfft2(-uk[0,0,:,:]*ksqr,norm='forward').T
u1=np.fft.irfft2(uk[0,1,:,:],norm='forward').T
qd.append(ax[0].imshow(u0,cmap='seismic',rasterized=True,vmin=vminom,vmax=vmaxom,origin='lower'))
qd.append(ax[1].imshow(u1,cmap='seismic',rasterized=True,vmin=vminn,vmax=vmaxn,origin='lower'))
print(x.shape,y.shape,u0.shape)
ax[0].set_title('$\Omega$',pad=-1)
ax[1].set_title('$n$',pad=-1)
ax[0].tick_params('y', labelleft=False)
ax[0].tick_params('x', labelbottom=False)
ax[1].tick_params('y', labelleft=False)
ax[1].tick_params('x', labelbottom=False)
plt.tight_layout()
t=fl['fields/t'][()]
plt.subplots_adjust(wspace=0.01, hspace=0.01)


nt0=10

Nt=t.shape[0]
for l in range(1):
    fig.colorbar(qd[l],ax=ax,format="%.2g", aspect=50,shrink=0.8,pad=0.02,fraction=0.03,location='bottom')

ax[0].axis('off')
ax[1].axis('off')

tx=fig.text(0.85, 0.1, "t=0")
if (comm.rank==0):
    lt=np.arange(Nt)
    lt_loc=np.array_split(lt,comm.size)
    if not os.path.exists('_tmpimg_folder'):
        os.makedirs('_tmpimg_folder')
else:
    lt_loc=None
lt_loc=comm.scatter(lt_loc,root=0)

for j in lt_loc:
    print(j)
    u0=np.fft.irfft2(-uk[j,0,:,:]*ksqr,norm='forward').T
    u1=np.fft.irfft2(uk[j,1,:,:],norm='forward').T
    qd[0].set_data(u0)
    qd[1].set_data(u1)
    tx.set_text('t='+str(int(t[j])*1.0))
    fig.savefig("_tmpimg_folder/tmpout%04i"%(j+nt0)+".png",dpi=200)#,bbox_inches='tight')
comm.Barrier()

if comm.rank==0:
    u0=np.fft.irfft2(-uk[0,0,:,:]*ksqr,norm='forward').T
    u1=np.fft.irfft2(uk[0,1,:,:],norm='forward').T
    qd[0].set_data(u0)
    qd[1].set_data(u1)
    tx.set_text('')

    fig.text(0.5, 0.85, f"{Npx}x{Npy} Simulation",fontsize=24,ha='center')
    fig.text(0.8, 0.80, "Ö. D.Gürcan",fontsize=12,ha='right')
    fig.text(0.1, 0.75, f"Box size : $[L_x,L_y]=[{Lx/np.pi}\pi,{Ly/np.pi}\pi]$",fontsize=16,ha='left')
    fig.text(0.1, 0.67, f"Padded Resolution : ${Npx} \\times {Npy}$",fontsize=16,ha='left')
    fig.text(0.1, 0.59, f"Viscosity : $\\nu = {nu:.2e}$",fontsize=16,ha='left')
    fig.text(0.1, 0.59, f"$C = {C}$, $\\kappa={kap}$",fontsize=16,ha='left')

    fig.savefig("_tmpimg_folder/tmpout%04i"%(0)+".png",dpi=200)#,bbox_inches='tight')
    for j in range(1,nt0):
        os.system("cp _tmpimg_folder/tmpout%04i"%(0)+".png _tmpimg_folder/tmpout%04i"%(j)+".png")
    
    os.system("ffmpeg -framerate 25 -y -i _tmpimg_folder/tmpout%04d.png -c:v libx264 -pix_fmt yuv420p -vf fps=25 "+outfl)
    shutil.rmtree("_tmpimg_folder")
