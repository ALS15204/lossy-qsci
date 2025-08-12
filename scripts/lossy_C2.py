#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 12:51:39 2024

@author: kesson
"""

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))
import numpy as np
import lib.diag as d
import lib.RLE as r
import lib.FED as fed
from math import comb
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import scipy as sp
import time

def bitmasks(n,m):
    if m < n:
        if m > 0:
            for x in bitmasks(n-1,m-1):
                # yield bitarray([1]) + x
                yield np.array([1] + list(x))
            for x in bitmasks(n-1,m):
                # yield bitarray([0]) + x
                yield np.array([0] + list(x))
        else:
            # yield n * bitarray('0')
            yield np.array([0]*n)
    else:
        # yield n * bitarray('1')
        yield np.array([1]*n)
        
def to10(s):
    a=[]
    a.extend(int(i,2) for i in s)
    return np.array(a)
#%%
#generate LiH molecule
t1=time.time()
dist = np.array([0.9])
atoms = ['C','C']
Rs = np.array([0, 1])
# atoms = ['Li','H']
# Rs = np.array([0, 1])
thetas = [0,0]
basis = '6-31g'
mult = 1
# # Activate orbitals (ACM, *right?*)
orb_choice = list(range(4,10))
hardcore = 0
mols = []
for i in dist:
    R = i*Rs
    mols.append(d.plane_molecule(atoms, R, thetas, basis, mult))

#The number of selected fermionic modes
M = len(orb_choice)*2

#The number of electrons
N = mols[0].nelectron-2*orb_choice[0]

# Q=M

#Ground State Energy -- optional
# basis = d.new_hot(mols[0], orb_choice[0], orb_choice, hardcore)
basis=[]
tol=comb(M,N)
basis.extend(iter(bitmasks(M,N)))
basis=np.array(basis)
functions = [[d.compute_integrals, d.get_active_space_integrals, basis]]
ene, gs, MF, mat= d.spectrum(mols, functions, orb_choice[0], orb_choice)
# stuff, MF = d.integrals(mols, functions, orb_choice)
# one = stuff[0][0]
# two = stuff[0][1]
# core = stuff[0][2]
Q = int(np.log2(comb(M,N)))
t2=time.time()
print('gs: ', ene[0][0], t2-t1)

def random_lossy(M,N,Q):
    return r.encoder(M, N, Q, 1)

def Energy_farm(R,noise,Q):
    # Q = int(np.log2(comb(M,N)))+1
    # Q=14
    stuff, MF = d.integrals(mols, functions, orb_choice)
    one = stuff[0][0]
    two = stuff[0][1]
    core = stuff[0][2]
    qsci_basis=[to10('1'*N+'0'*(M-N))]
    qsci_basis1=np.array(qsci_basis)
    oldu=0
    u=0
    max_iter=20
    r_fid=0
    for _ in range(max_iter):
        # if i<max_iter//2:
        B=random_lossy(M, N, Q)
        # else:
        # Q=int(np.log2(comb(M,N)))-1
        # g_opt = generator_opt(qsci_basis1, num_generations=50, population_size=50)
        # B = g_opt.genetic_algorithm()

        print('Decoding...')
        model_set = NN_decoder(M, N, B)
        'compute integrals and PySCF meanfield object'
        stuff, MF= d.integrals(mols, functions, orb_choice)

        naos = np.arange(mols[0].nao)

        #calculate all non-zero contributions of the RDM, neglecting RDM symmetries
        base = fed.Indices(stuff[0], threshold = 10**-12)

        #compute groups (ACM, *right?*)
        compute = fed.RDM_grouper(base)
        #determines the measurement groups
        groups = fed.measurement_group(compute)

        #compute x-string
        ustrings = fed.xfunc(groups, np.identity(M))
        signs = fed.glob_sign(groups, M)

        #Generate histrogram from ground state -- to mimick what we measured from quantum computer
        print('State preparation...')
        v=gs[0][0].copy()
        v+=(0.5-np.random.rand(len(v)))*noise
        v/=np.linalg.norm(v)
        r_fid=(gs[0][0].conj().dot(v))**2
        print('Fid:',r_fid)
        hists = fed.hist_format(v, basis, B, ustrings[-1:])
        CI_s=np.array(list(hists[-1].keys()))
        CI=np.array(list(hists[-1].values()))
        num=np.argsort(-abs(CI))[:R]
        qsci_basis_n=[]
        for j in num:
            v=fed.nn_str_decode(CI_s[j], model_set, B)
            if v != '0'*M:
                qsci_basis_n.append(to10(v))
                # print(CI[j],v)
        print('QSCI...')
        qsci_basis1=np.unique(qsci_basis+qsci_basis_n,axis=0)
        if len(qsci_basis1)>2:
            mat = d.first_quantised(mols, one, two, qsci_basis1)
            u,w=sp.sparse.linalg.eigs(mat,k=1,which='SR')
            if (u<oldu) and abs(u-oldu)>1e-3:
                oldu=u
                qsci_basis=qsci_basis+qsci_basis_n
            qsci_basis1=np.unique(qsci_basis,axis=0)
            print(len(qsci_basis1))
            # print('gs: ', ene[0][0])
            print('qsci:', u+MF[0].energy_nuc()+core)
    return u[0]+MF[0].energy_nuc()+core, r_fid/max_iter,qsci_basis

#generate NN decoder
def NN_decoder(M, N, B, num_epochs=1000):
    excite=np.array(list(range(1,N+1)))
    excite=excite[excite<=(M-Q)]
    return {
        N: fed.nn_parity(
            B, N, hidden=4 * M * N, num_epochs=num_epochs, excite=excite
        )
    }
#%%
histE=[]
for Q in range(10,17):
    E,r_fid,qsci_basis=Energy_farm(50, 0.02, Q)
    histE.append((Q,E,np.sqrt(r_fid),len(np.unique(qsci_basis,axis=0))))

histE=np.array(histE)

#%% 2.0 
g12={}
g12[0.9]=-75.0699288613
g12[1]=-75.30727938196
g12[1.05]=-75.3761444588
g12[1.1]=-75.421718282
g12[1.3]=-75.46208386
g12[1.5]=-75.45408222
g12[1.8]=-75.41324972
g12[2]=-75.383410204
g12[2.3]=-75.34755806
g12[2.5]=-75.3318316
g12[3]=-75.31974497
gse={}
gse[0.9]=-75.086891285
gse[1]=-75.32416555 #1
gse[1.05]=-75.39278962#1.05
gse[1.1]=-75.43801745 #1.1
gse[1.3]=-75.47718491 #1.3
gse[1.5]= -75.46681237 #1.5
gse[1.8]=-75.42931907 #1.8
gse[2]=-75.39745468 #2
gse[2.3]=-75.35927863 #2.3
gse[2.5]=-75.34225663 #2.5
gse[3]=-75.33307285 #3

plt.plot(list(gse.keys()),np.array(list(gse.values()))+1.59e-3,alpha=0.5,color='k',linestyle='--',label='Chemicial Accuracy (20,4)')
plt.plot(list(g12.keys()),np.array(list(g12.values()))+1.59e-3,alpha=0.5,color='b',linestyle='--',label='Chemicial Accuracy (12,4)')

Q=list(range(10,17))
marker=['^','1','s','X','d','p','.']
color=['r', 'g', 'purple', 'b','y', 'orange', 'navy']
for i in [0.9, 1, 1.05, 1.1, 1.3, 1.5, 1.8, 2, 2.3, 2.5, 3]:
    histE = np.loadtxt(f'C2_{str(i)}', dtype=np.complex128)
    if i==1:
        for j in [0,1,2,5,6]:
            plt.scatter(
                i,
                histE[j, 1],
                color=color[j],
                marker=marker[j],
                label=f'Lossy QSCI Qubit={str(Q[j])}',
            )
    else:
        for j in [0,1,2,5,6]:
            plt.scatter(i,histE[j,1],color=color[j],marker=marker[j])

plt.xlabel('Bond Length (Å)',fontsize=15)
plt.ylabel('Energy (Hartree)',fontsize=15)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.3f}'))
plt.legend(ncol=2)