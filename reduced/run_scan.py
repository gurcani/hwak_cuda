#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:10:50 2021

@author: ogurcan
"""
import sys
import os
import numpy as np
import h5py as h5
from scipy.integrate import solve_ivp
from scipy.optimize import fmin
from scipy.signal import find_peaks
sys.path.insert(1, os.path.realpath('../'))
from hwak_numpy import hasegawa_wakatani
from time import time
import matplotlib.pylab as plt
from mpi4py import MPI
comm=MPI.COMM_WORLD

def fhwrun(uk,Npx,Npy,Lx,Ly,t):
    nxh=int(Npx/3)
    dtout=t[1]-t[0]
    hw=hasegawa_wakatani(modified=True,
                         wecontinue=False,
                         saveresult=False,
                         flname='out_fit.h5',
                         onlydiag=False,
                         C=1.0,
                         kap=1.0,
                         Npx=Npx,
                         Npy=Npy,
                         Lx=Lx,
                         Ly=Ly,
                         nu=0.0,
                         D=0.0,
                         nuZF=0.0,
                         DZF=0.0,
                         t0=t[0],
                         t1=t[-1]+dtout,
                         Amp0=1e-4,
                         sigk=100.0,
                         solver='vode',
                         nl_method='62',
                         dtstep=dtout,
                         dtout=dtout)
    hw.uk[:,:,:]=uk*np.exp(1j*2*np.pi*np.random.randn(*uk.shape))
    hw.lm[0,0,np.r_[nxh-1,nxh+1],1]-=2.0
#    ps0=hw.lm[:,:,np.r_[nxh-1,nxh+1],:2].copy()
    ukres=np.zeros(t.shape+hw.uk.shape,dtype=complex)

    def fcb(t,ct,j,fl,uk,kx,ky):
#        print('t='+str(t)+', '+str(time()-ct)+" secs elapsed. I="+str(np.sum(np.abs(uk)**2)))
        ukres[j,]=uk

    hw.fcallback=fcb
    hw.run()
    return ukres

t=np.arange(0,200,0.1)

if(comm.rank==0):
    fl=h5.File('init.h5','r')
    uk=fl['uk'][()]
    Lx=fl['Lx'][()]
    Ly=fl['Ly'][()]
    Npx=fl['Npx'][()]
    Npy=fl['Npy'][()]
    fl.close()
else:
    uk,Lx,Ly,Npx,Npy=None,None,None,None,None
    
uk = comm.bcast(uk, root=0)
Lx = comm.bcast(Lx, root=0)
Ly = comm.bcast(Ly, root=0)
Npx = comm.bcast(Npx, root=0)
Npy = comm.bcast(Npy, root=0)

f = lambda q : fhwrun(uk,Npx,Npy,2*np.pi/q,Ly,t)

q=np.arange(0.1,1.4,0.01)
tlres=np.zeros(q.shape)
Nph=64

if (comm.rank==0):
    lph=np.arange(Nph)
    lph_loc=np.array_split(lph,comm.size)
else:
    lph_loc=None
lph_loc=comm.scatter(lph_loc,root=0)

for l in range(q.size):
    print('q='+str(q[l]))
    tl=0
    for j in lph_loc:
        res=f(q[l])
        r=find_peaks(np.abs(res[:,0,0,1]))[0][0]
        tl+=t[r]
    tlr=comm.reduce(tl, op=MPI.SUM, root=0)
    if (comm.rank==0):
        tlres[l]=tlr/Nph
if (comm.rank==0):
    fl=h5.File('qxs_tlr_rph_'+str(Nph)+'.h5','w')
    fl['q']=q
    fl['tl']=tlres
    fl.close()
