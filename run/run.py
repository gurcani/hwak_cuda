#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 16:10:50 2021

@author: ogurcan
"""
import sys
import os
sys.path.insert(1, os.path.realpath('../'))
import numpy as np
from hwak_cuda_all import hasegawa_wakatani

hw=hasegawa_wakatani(modified=True,
                     wecontinue=False,
                     onlydiag=False,
                     flname="out.h5",
                     C=1.0,
                     kap=1.0,
                     Npx=2048,
                     Npy=2048,
                     Lx=16*np.pi,
                     Ly=16*np.pi,
                     nu=1e-5,
                     D=1e-5,
                     nuZF=0.0,
                     DZF=0.0,
                     t1=500,
                     Amp0=1e-4,
                     solver='DOP853',
                     nl_method='34',
                     dtstep=0.1,
                     dtsave=1.0,
                     dtshow=1.0)
hw.run()
