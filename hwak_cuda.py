#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 16 11:59:18 2023

@author: ogurcan
"""

import numpy as np
import cupy as cp
from time import time
import h5py as h5
import sys,os
sys.path.insert(1, os.path.realpath(os.path.dirname(__file__)+'/cupy_ivp'))

default_parameters={
    'C':0.1,
    'kap':0.2,
    'nu':2e-4,
    'D':2e-4,
    'Npx':1024,
    'Npy':1024,
    'padx':3/2,
    'pady':3/2,
    'Lx':16*np.pi,
    'Ly':16*np.pi,
    'modified':False,
    'nl_method':'34',
    'nux':'nu',
    'nuy':'nu',
    'Dx':'D',
    'Dy':'D',
    'DZF':0.0,
    'nuZF':0.0,
    'nuksqrpow' : 1,
    'nukxsqrpow' : 'nuksqrpow',
    'nukysqrpow' : 'nuksqrpow',
    'nuZksqrpow' : 0,
    'Amp0' : 1e-4,
    'sigk' : 0.5,
    'sigkx' : 'sigk',
    'sigky' : 'sigk'
}

default_solver_parameters={
    'solver':'DOP853',
    't0':0.0,
    't1':1000.0,
    'dtstep':0.1,
    'dtshow':1.0,
    'dtsave':1.0,
    'dtref':1.0,
    'atol' : 1e-12,
    'rtol' : 1e-6,
    'mxsteps' : 10000,
}

default_controls={
    'platform':'gpu',
    'wecontinue':False,
    'onlydiag':False,
    'saveresult':True,
    'flname':"out.h5",
    'threads_per_block':128,
    'blocks_per_grid':128,
    'nthreads':8
}

def oneover(x):
    res=x*0    
    res[(x!=0)]=1/x[(x!=0)]
    return res

def lincompfreq(lm):
    w,v=np.linalg.eig(lm.T.get())
    ia=np.argsort(w.real,axis=-1)
    lam=np.take_along_axis(w, np.flip(ia,axis=-1), axis=-1).T
    vi=np.zeros_like(v.T)
    vi[0,:,:,:]=np.take_along_axis(v[:,:,0,:], np.flip(ia,axis=-1), axis=-1).T
    vi[1,:,:,:]=np.take_along_axis(v[:,:,1,:], np.flip(ia,axis=-1), axis=-1).T
    return lam,vi

def hermsymznyq(u):
    Nx=u.shape[-2]
    Ny=u.shape[-1]
    ll=tuple([slice(0,l,None) for l in u.shape[:-2]])+(slice(1,int(Nx/2),None),slice(0,1,None))
    l0=ll[:-2]+(slice(0,1,None),slice(0,1,None))
    u[l0]=u[l0].real
    llp=ll[:-2]+(slice(Nx,int(Nx/2),-1),slice(0,1,None))
    u[llp]=u[ll].conj()
    u[ll[:-2]+(slice(int(Nx/2),int(Nx/2)+1,None),slice(0,Ny,None))]=0.0
    u[ll[:-2]+(slice(0,Nx,None),slice(Ny-1,Ny,None))]=0.0

def multin34(u,F,kx,ky): # ~5 ms (cpu), 700 µs (gpu)
    Nx=u.shape[1]
    Ny=u.shape[2]
    Nxh=int(Nx/2)
    s0=slice(None,Nxh,None)
    s1=slice(-Nxh,None,None)
    sy=slice(None,Ny,None)
    for s in [s0,s1]:
        F[0,s,sy]=1j*kx[s,sy]*u[0,s,sy]
        F[1,s,sy]=1j*ky[s,sy]*u[0,s,sy]
        F[2,s,sy]=u[1,s,sy]

def multin62(u,F,kx,ky):
    Nx=u.shape[1]
    Ny=u.shape[2]
    Nxh=int(Nx/2)
    s0=slice(None,Nxh,None)
    s1=slice(-Nxh,None,None)
    sy=slice(None,Ny,None)
    for s in [s0,s1]:
        ksqr=kx[s,sy]**2+ky[s,sy]**2
        F[0,s,sy]=1j*kx[s,sy]*u[0,s,sy]
        F[1,s,sy]=1j*ky[s,sy]*u[0,s,sy]
        F[2,s,sy]=-1j*ksqr*kx[s,sy]*u[0,s,sy]
        F[3,s,sy]=-1j*ksqr*ky[s,sy]*u[0,s,sy]
        F[4,s,sy]=1j*kx[s,sy]*u[1,s,sy]
        F[5,s,sy]=1j*ky[s,sy]*u[1,s,sy]

def multinDSI(u,F,kx,ky):
    Nx=u.shape[1]
    Ny=u.shape[2]
    Nxh=int(Nx/2)
    s0=slice(None,Nxh,None)
    s1=slice(-Nxh+1,None,None)
    sy=slice(None,Ny,None)
    for s in [s0,s1]:
        ksqr=kx[s,sy]**2+ky[s,sy]**2
        F[0,s,sy]=1j*kx[s,sy]*u[0,s,sy]
        F[1,s,sy]=1j*ky[s,sy]*u[0,s,sy]
        F[2,s,sy]=-1j*ksqr*kx[s,sy]*u[0,s,sy]
        F[3,s,sy]=-1j*ksqr*ky[s,sy]*u[0,s,sy]
        F[4,s,sy]=1j*kx[s,sy]*u[1,s,sy]
        F[5,s,sy]=1j*ky[s,sy]*u[1,s,sy]

def multout34(F):
    Npx=F.shape[1]
    Npy=F.shape[2]-2
    Norm=Npx*Npy
    s,sy=slice(None,Npx,None),slice(None,Npy,None)
    #dxphi=F[0,]
    dyphi=F[1,s,sy].copy()
    #n=F[2,]
    F[3,s,sy]=F[2,s,sy]*F[0,s,sy]/Norm
    F[2,s,sy]=F[2,s,sy]*dyphi/Norm
    F[1,s,sy]=(dyphi**2-F[0,s,sy]**2)/Norm
    F[0,s,sy]=F[0,s,sy]*dyphi/Norm

def multout62(F):
    Npx=F.shape[1]
    Npy=F.shape[2]-2
    Norm=Npx*Npy
    s,sy=slice(None,Npx,None),slice(None,Npy,None)
    dyphi=F[1,s,sy].copy()
    F[1,s,sy]= (F[0,s,sy]*F[5,s,sy]-dyphi*F[4,s,sy])/Norm
    F[0,s,sy]= (F[0,s,sy]*F[3,s,sy]-dyphi*F[2,s,sy])/Norm

def multoutDSI(F):
    Npx=F.shape[1]
    Norm=Npx
    s=slice(None,Npx,None)
    dxphi=F[0,s,].copy()
    dyphi=F[1,s,].copy()
    F[0,s,0]=2*cp.sum((dxphi*F[3,s,].conj()-dyphi*F[2,s,].conj()).real,axis=1)/Norm#(dxphi*dyom-dyphi*dxom)/Norm
    F[1,s,0]=2*cp.sum((dxphi*F[5,s,].conj()-dyphi*F[4,s,].conj()).real,axis=1)/Norm #(dxphi*dyn-dyphi*dxn)/Norm
    F[0,s,1:]=(dxphi[:,0]*F[3,s,1:].T-dyphi[:,1:].T*F[2,s,0]).T/Norm#(dxphi*dyom-dyphi*dxom)/Norm
    F[1,s,1:]=(dxphi[:,0]*F[5,s,1:].T-dyphi[:,1:].T*F[4,s,0]).T/Norm #(dxphi*dyn-dyphi*dxn)/Norm

def multvec34(dydt,y,a,x,b):#,f):
    Nx=y.shape[1]
    Ny=y.shape[2]
    Nxh=int(Nx/2)
    s0=slice(None,Nxh,None)
    s1=slice(-Nxh,None,None)
    sy=slice(None,Ny,None)
    for s in [s0,s1]:
        dydt[0,s,sy]=a[0,0,s,sy]*y[0,s,sy]+a[0,1,s,sy]*y[1,s,sy]+b[0,s,sy]*x[0,s,sy]+b[1,s,sy]*x[1,s,sy]#+f[0,s,sy]
        dydt[1,s,sy]=a[1,0,s,sy]*y[0,s,sy]+a[1,1,s,sy]*y[1,s,sy]+b[2,s,sy]*x[2,s,sy]+b[3,s,sy]*x[3,s,sy]#+f[1,s,sy]

def multvec62(dydt,y,a,x,b):#,f):
    Nx=y.shape[1]
    Ny=y.shape[2]
    Nxh=int(Nx/2)
    s0=slice(None,Nxh,None)
    s1=slice(-Nxh,None,None)
    sy=slice(None,Ny,None)
    for s in [s0,s1]:
        dydt[0,s,sy]=a[0,0,s,sy]*y[0,s,sy]+a[0,1,s,sy]*y[1,s,sy]+b[0,s,sy]*x[0,s,sy]#+f[0,s,sy]
        dydt[1,s,sy]=a[1,0,s,sy]*y[0,s,sy]+a[1,1,s,sy]*y[1,s,sy]+b[1,s,sy]*x[1,s,sy]#+f[1,s,sy]

def multvecDSI(dydt,y,a,x,b):#,f):
    Nx=y.shape[1]
    Ny=y.shape[2]
    Nxh=int(Nx/2)
    s0=slice(None,Nxh,None)
    s1=slice(-Nxh,None,None)
    sy=slice(None,Ny,None)
    for s in [s0,s1]:
        dydt[0,s,sy]=a[0,0,s,sy]*y[0,s,sy]+a[0,1,s,sy]*y[1,s,sy]+b[0,s,sy]*x[0,s,sy]#+f[0,s,sy]
        dydt[1,s,sy]=a[1,0,s,sy]*y[0,s,sy]+a[1,1,s,sy]*y[1,s,sy]+b[1,s,sy]*x[1,s,sy]#+f[1,s,sy]

def multvec_lin(dydt,y,a):
    dydt[0,]=a[0,0,]*y[0,]+a[0,1,]*y[1,]
    dydt[1,]=a[1,0,]*y[0,]+a[1,1,]*y[1,]

def init_linmats(pars,kx,ky):
    #Initializing the linear matrices
    C,kap,nux,Dx,nuy,Dy,nuZF,DZF=[pars[l] for l in ['C','kap','nux','Dx','nuy','Dy','nuZF','DZF']]
    nukxsqrpow=pars['nukxsqrpow']
    nukysqrpow=pars['nukysqrpow']
    nuZksqrpow=pars['nuZksqrpow']
    lm=cp.zeros((2,2)+kx.shape,dtype=complex)
    # if(forcing):
    #     forcelm=cp.zeros((2,)+kx.shape,dtype=complex)
    #     forcelm[:]=0
    # else:
    #     forcelm

    ksqr=kx**2+ky**2
    lm[0,0,:,:]=-C*oneover(ksqr)-nux*(kx**2)**nukxsqrpow-nuy*(ky**2)**nukysqrpow
    lm[0,1,:,:]=C*oneover(ksqr)
    lm[1,0,:,:]=-1j*kap*ky+C
    lm[1,1,:,:]=-C-Dx*(kx**2)**nukxsqrpow-Dy*(ky**2)**nukysqrpow
    lm[:,:,0,0]=0.0
    if(pars['modified']):
        lm[0,0,:,0]=-nuZF*(kx[:,0]**2)*nuZksqrpow
        lm[0,1,:,0]=0.0
        lm[1,0,:,0]=0.0
        lm[1,1,:,0]=-DZF*(kx[:,0]**2)*nuZksqrpow

    if((pars.get('nl_method')=='34')):
        nlm=cp.zeros((4,)+kx.shape,dtype=complex)
        nlm[0,:,:]=(kx**2-ky**2)*oneover(ksqr)
        nlm[1,:,:]=(kx*ky)*oneover(ksqr)
        nlm[2,:,:]=1j*kx
        nlm[3,:,:]=-1j*ky
    else:
        if not(pars.get('nl_method') in ['62','DSI']):
            print(f"unkonwn nonlinear method: {pars.get('nl_method')}, assuming 62")
        nlm=cp.zeros((2,)+kx.shape,dtype=float)
        nlm[0,:,:]=1*oneover(ksqr)
        nlm[1,:,:]=-1.0
    nlm[:,0,0]=0.0
    nlm[:,:,-1]=0.0
    nlm[:,int(nlm.shape[1]/2),:]=0.0
    hermsymznyq(lm)
    return lm,nlm #,forcelm

def init_ffts(Npx,Npy,Nfb,Nff):
        datk=cp.zeros((max(Nfb,Nff),Npx,int(Npy/2)+1),dtype=complex)
        def ftmp():
            datk.view(dtype=float)[:Nfb,:,:-2] = cp.fft.irfft2(datk[:Nfb,],norm='forward')
            return datk.view(dtype=float)
        pfb=ftmp
        def ftmp():
            datk[:Nff,]=cp.fft.rfft2(datk.view(dtype=float)[:Nff,:,:-2],norm='backward')
            return datk
        pff=ftmp
        return datk,pfb,pff

def init_fftsDSI(Npx,Npy):
        datk=cp.zeros((6,Npx,int(Npy/2)+1),dtype=complex)
        def ftmp():
            datk[:6,]=cp.fft.ifft(datk[:6,],axis=1,norm='forward')
            return datk
        pfb=ftmp
        def ftmp():
            datk[:2,]=cp.fft.fft(datk[:2,],axis=1,norm='backward')
            return datk
        pff=ftmp
        return datk,pfb,pff

def init_kspace_grid(Nx,Ny,Lx,Ly):
    dkx=2*np.pi/Lx
    dky=2*np.pi/Ly
    kxl=np.r_[0:int(Nx/2),-int(Nx/2):0]*dkx
    kyl=np.r_[0:int(Ny/2+1)]*dky
    kx,ky=cp.meshgrid(cp.array(kxl),cp.array(kyl),indexing='ij')
    return kx,ky

def init_fields(uk,kx,ky,A=1e-4,sigkx=0.5,sigky=0.5):
    kx0,ky0=0,0
    th=cp.zeros(kx.shape)
    th[:,:]=cp.random.rand(kx.shape[0],kx.shape[1])*2*cp.pi;
    phik0=A*cp.exp(-(kx-kx0)**2/2/sigkx**2-(ky-ky0)**2/2/sigky**2)*cp.exp(1j*th);
    nk0=A*cp.exp(-(kx-kx0)**2/2/sigkx**2-(ky-ky0)**2/2/sigky**2)*cp.exp(1j*th);
    uk[0,:,:]=phik0
    uk[1,:,:]=nk0
    uk[:,0,0]=0.0
    hermsymznyq(uk)

# def load_pars(fl):
#     pars={}
#     for l,m in fl['params'].items():
#         pars[l]=m[()]
#     return pars

def save_pars(fl,pars):
    if not ('params' in fl):
        fl.create_group('params')
    for l,m in pars.items():
        if l not in fl['params'].keys():
            fl['params'][l]=m

def save_fields(fl,**kwargs):
    if not ('fields' in fl):
        grp=fl.create_group('fields')
    else:
        grp=fl['fields']
    for l,m in kwargs.items():
        if not l in grp:
            if(np.isscalar(m)):
                grp.create_dataset(l,(1,),maxshape=(None,),dtype=type(m))
                if(not fl.swmr_mode):
                    fl.swmr_mode = True
            else:
                grp.create_dataset(l,(1,)+m.shape,chunks=(1,)+m.shape,maxshape=(None,)+m.shape,dtype=m.dtype)
                if(not fl.swmr_mode):
                    fl.swmr_mode = True
            lptr=grp[l]
            lptr[-1,]=m
        else:
            lptr=grp[l]
            lptr.resize((lptr.shape[0]+1,)+lptr.shape[1:])
            lptr[-1,]=m
        lptr.flush()
        fl.flush()

def save_data(fl,**kwargs):
    if not ('data' in fl):
        grp=fl.create_group('data')
    else:
        grp=fl['data']
    for l,m in kwargs.items():
        if(l not in grp.keys()):
            grp[l]=m
    
class hasegawa_wakatani:
    def __init__(self,**kwargs):
        controls=default_controls.copy()
        params=default_parameters.copy()
        svpars=default_solver_parameters.copy()
#        force_handler=None
        for l,m in kwargs.items():
            if(l in default_controls.keys()):
                controls[l]=m
            elif(l in default_parameters.keys()):
                params[l]=m
            elif(l in default_solver_parameters.keys()):
                svpars[l]=m
            # elif(l in ['force_handler']):
            #     force_handler=m
            else:
                print(l,'is neither a parameter nor a control flag')
        if('onlydiag' in kwargs.keys() and 'saveresult' not in kwargs.keys() and controls['onlydiag']):
            controls['saveresult']==False
        if(controls['onlydiag']):
            fl=h5.File(controls['flname'],'r',libver='latest',swmr=True)
            nl_method=params['nl_method']
#            params=load_pars(fl)           
            for l,m in fl['params'].items():
                params[l]=m[()]
            params['nl_method']=nl_method
        elif(controls['wecontinue']):
            fl=h5.File(controls['flname'], 'r+',libver='latest')
            fl.swmr_mode = True
            nl_method=params['nl_method']
            for l,m in fl['params'].items():
                params[l]=m[()]
            # params=load_pars(fl)
            params['nl_method']=nl_method
        else:
            if(controls['flname'] and controls['saveresult']):
                fl=h5.File(controls['flname'],'w',libver='latest')
                fl.swmr_mode = True
            else:
                fl=None
        for l,m in params.items():
            if(m in params):
                params[l]=params[m]
        print(params)
        Npx,Npy=params['Npx'],params['Npy']
        padx,pady=params['padx'],params['pady']
        Lx,Ly=params['Lx'],params['Ly']
        Nx,Ny=int(Npx/padx/2)*2,int(Npy/pady/2)*2
        if(controls['onlydiag'] or controls['wecontinue']):
            kx=fl['data/kx'][()]
            ky=fl['data/ky'][()]
        else:
            kx,ky=init_kspace_grid(Nx,Ny,Lx,Ly)
        lm,nlm=init_linmats(params,kx,ky)
#        lm,nlm,forcelm=init_linmats(params,kx,ky)
        uk=cp.zeros((2,)+kx.shape,dtype=complex)
        if(controls['onlydiag'] or controls['wecontinue']):
            uk[:]=fl['fields/uk'][-1,]
            t0=fl['fields/t'][-1]
        else:
            Amp0=params['Amp0']
            sigkx=params['sigkx']
            sigky=params['sigky']
            init_fields(uk,kx,ky,A=Amp0,sigkx=sigkx,sigky=sigky)
            t0=svpars['t0']
        if(controls['saveresult']):
            save_pars(fl,params)
            save_data(fl,kx=kx.get(),ky=ky.get())
        threads_per_block=controls['threads_per_block']
        blocks_per_grid=controls['blocks_per_grid']
        if(params.get('nl_method')=='34'):
            multin=multin34
            multout=multout34#[threads_per_block,blocks_per_grid]
            multvec=multvec34
            datk,pfb,pff=init_ffts(Npx,Npy,3,4)
        elif(params.get('nl_method')=='DSI'):
            multin=multinDSI
            multout=multoutDSI
            multvec=multvecDSI
            datk,pfb,pff=init_fftsDSI(Npx,Npy)
        else:
            if not( (params.get('nl_method')=='62')):
                print(f"unkonwn nonlinear method: {params.get('nl_method')}, assuming 62")
            multin=multin62
            multout=multout62
            multvec=multvec62
            datk,pfb,pff=init_ffts(Npx,Npy,6,2)
        self.datk=datk
        self.pfb=pfb
        self.pff=pff
        self.threads_per_block=threads_per_block
        self.blocks_per_grid=blocks_per_grid
        self.multin=multin
        self.multout=multout
        self.multvec=multvec
        self.params=params
        self.kx=kx
        self.ky=ky
        self.lm=lm
        self.nlm=nlm
#        self.forcelm=forcelm
        self.uk=uk
        self.dukdt=cp.zeros_like(uk)
        self.svpars=svpars
        self.controls=controls
        self.fl=fl
        self.t0=t0
#        self.force_handler=force_handler
        if not(self.controls['onlydiag']):
            self.r=self.init_solver()

    def init_solver(self):
        t1,dtstep,dtshow,dtsave,dtref,atol,rtol,mxsteps=[self.svpars[l] for l in ['t1','dtstep','dtshow','dtsave','dtref','atol','rtol','mxsteps']]
        t0=self.t0
        rhs=self.rhs
        if(self.svpars['solver']=='RK45'):
            import cupy_ivp as cpi
            f = lambda t,y : rhs (t, y, self.dukdt)
            r = cpi.RK45(f,t0,self.uk.ravel().view(dtype=float),t1,max_step=dtstep,atol=atol,rtol=rtol)
            def integr(ti):
                while(r.t<ti):
                    r.step()
            r.integrate=integr
            r.gety = lambda ti : r.dense_output()(ti)
        elif(self.svpars['solver']=='RK23'):
            import cupy_ivp as cpi
            f = lambda t,y : rhs (t, y, self.dukdt)
            r = cpi.RK23(f,t0,self.uk.ravel().view(dtype=float),t1,max_step=dtstep,atol=atol,rtol=rtol)
            def integr(ti):
                while(r.t<ti):
                    r.step()
            r.integrate=integr
            r.gety = lambda ti : r.dense_output()(ti)
        elif(self.svpars['solver']=='DOP853'):
            import cupy_ivp as cpi
            f = lambda t,y : rhs (t, y, self.dukdt)
            r = cpi.DOP853(f,t0,self.uk.ravel().view(dtype=float),t1,max_step=dtstep,atol=atol,rtol=rtol)
            def integr(ti):
                while(r.t<ti and r.status=='running'):
                    r.step()
            r.integrate=integr
            r.gety = lambda ti : r.dense_output()(ti)
        r.t0,r.t1,r.dtstep,r.dtshow,r.dtsave,r.dtref=t0,t1,dtstep,dtshow,dtsave,dtref
        return r

    def linfreq(self):
        lam,xi=lincompfreq(self.lm)
        return 1j*lam

    def rhs(self,t,y,dukdt):
        uk=y.view(dtype=complex).reshape(self.uk.shape)
        datk=self.datk
        datk.fill(0)
        self.multin(uk,datk,self.kx,self.ky)
        dat=self.pfb()
        self.multout(dat)
        datk=self.pff()
        self.multvec(self.dukdt,uk,self.lm,datk,self.nlm)#,self.forcelm)
        hermsymznyq(self.dukdt)
        return dukdt.ravel().view(dtype=float)

    def run(self):
        r=self.r
        self.ukcur=self.uk.get()
        t0,t1,dtstep,dtshow,dtsave,dtref=r.t0,r.t1,r.dtstep,r.dtshow,r.dtsave,r.dtref
        t=t0
        i=0
        j=0
        l=0
        m=0
        ct=time()
        trnd=int(-np.log10(min(dtstep,dtsave,dtshow,dtref)/100))
        tnext=round(t0+(i+1)*dtstep,trnd)
        tsavenext=round(t0+(j+1)*dtsave,trnd)
        tshownext=round(t0+(j+1)*dtsave,trnd)
        trefnext=round(t0+(l+1)*dtref,trnd)
        
        if(not self.controls['wecontinue'] and self.controls['saveresult']):
            save_fields(self.fl,uk=self.ukcur,t=t)
        while(r.t<t1):
            r.integrate(tnext)
            i+=1
            tnext=round(t0+(i+1)*dtstep,trnd)
            if(r.t>=trefnext):
                l+=1
                trefnext=round(t0+(l+1)*dtref,trnd)
                # if(self.force_handler is not None):
                #     self.force_handler(self,r.t)
            if(r.t>=tshownext):
                m+=1
                tshownext=round(t0+(m+1)*dtshow,trnd)
                print('t='+str(r.t)+', '+str(time()-ct)+" secs elapsed. I="+str(np.sum(np.abs(r.gety(t).view(dtype=complex).reshape(self.ukcur.shape))**2)))
            if(r.t>=tsavenext):
                t=tsavenext
                self.ukcur[:]=r.gety(t).get().view(dtype=complex).reshape(self.ukcur.shape)
                if(self.controls['saveresult']):
                    save_fields(self.fl,uk=self.ukcur,t=t)
                if(hasattr(self,'fcallback')):
                    if(callable(self.fcallback)):
                        self.fcallback(t,ct,j,self.fl,self.ukcur,self.kx,self.ky)
                j+=1
                tsavenext=round(t0+(j+1)*dtsave,trnd)
        self.fl.close()
