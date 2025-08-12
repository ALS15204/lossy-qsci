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
import scipy as sp
from bitarray import bitarray

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
#generate LiH molecule
dist = np.array([2.5])
atoms = ['C','C']
Rs = np.array([0, 1])
# atoms = ['Li','H']
# Rs = np.array([0, 1])
thetas = [0,0]
basis = '6-31g'
mult = 1
# # Activate orbitals (ACM, *right?*)
orb_choice = list(range(4,14))
hardcore = 0
mols = []
for i in dist:
    R = i*Rs
    mols.append(d.plane_molecule(atoms, R, thetas, basis, mult))

#C2H4
symbols = ["C", "C", "H", "H", "H", "H"]
coordinates = np.array([
    [0.0, 0.0, 0.0], 
    [1.339, 0.0, 0.0], 
    [-0.9341, 0.0, 0.9235], 
    [-0.9341, 0.0, -0.9235], 
    [2.2731, 0.0, 0.9235], 
    [2.2731, 0.0, -0.9235]
])
orb_choice = list(range(5,13))
mols=[d.my_molecule(symbols, coordinates, basis, mult)]


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
stuff, MF = d.integrals(mols, functions, orb_choice)
one = stuff[0][0]
two = stuff[0][1]
core = stuff[0][2]
Q = int(np.log2(comb(M,N)))
print('gs: ', ene[0][0])

def random_lossy(M,N,Q):
    B = r.encoder(M, N, Q, 1)
    return B

#%% Lossy C2H4
R=50
Q=12
stuff, MF = d.integrals(mols, functions, orb_choice)
one = stuff[0][0]
two = stuff[0][1]
core = stuff[0][2]
qsci_basis=[to10('1'*N+'0'*(M-N))]
oldu=0
u=0
r_fid=0

B=random_lossy(M, N, Q)
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
hists = fed.hist_format(gs[0][0], basis, B, ustrings[-1:])
CI_s=np.array(list(hists[-1].keys()))
CI=np.array(list(hists[-1].values()))

num=np.argsort(-CI)[:R]
qsci_basis=[]
for j in num:
    v=basis[j]
    qsci_basis.append(v)
print('QSCI...')
mat = d.first_quantised(mols, one, two, qsci_basis)
u,w=sp.sparse.linalg.eigs(mat,k=1,which='SR')
print(u[0]+MF[0].energy_nuc()+core)
#%%
indexes = np.unique(qsci_basis, return_index=True,axis=0)[1]
qn=[qsci_basis[index] for index in sorted(indexes)]
E_qsci=[]
for i in range(50,70):
    mat = d.first_quantised(mols, one, two, qn[:i])
    u,w=sp.sparse.linalg.eigs(mat,k=1)
    E_qsci.append(u[0]+MF[0].energy_nuc()+core)
#%%
def Energy_farm(R,noise,Q,bias=[]):
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
    for i in range(max_iter):
        B=random_lossy(M, N, Q)        
        print('Decoding...')
        model_set = NN_decoder(M, N, B, bias=bias)
        'compute integrals and PySCF meanfield object'
        stuff, MF= d.integrals(mols, functions, orb_choice)
        #calculate all non-zero contributions of the RDM, neglecting RDM symmetries
        base = fed.Indices(stuff[0], threshold = 10**-12)
        
        #compute groups (ACM, *right?*)
        compute = fed.RDM_grouper(base)
        #determines the measurement groups
        groups = fed.measurement_group(compute)
        
        #compute x-string
        ustrings = fed.xfunc(groups, np.identity(M))
        
        #Generate histrogram from ground state -- to mimick what we measured from quantum computer
        print('State preparation...')
        v=gs[0][0].copy()
        v+=(0.5-np.random.rand(len(v)))*noise
        v/=np.linalg.norm(v)
        r_fid+=(gs[0][0].conj().dot(v))**2
        print(r_fid/(i+1))
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
            if abs(u-oldu)>1e-3:
                oldu=u
                qsci_basis=qsci_basis+qsci_basis_n
            qsci_basis1=np.unique(qsci_basis,axis=0)
            print(len(qsci_basis1))
            print('qsci:', u+MF[0].energy_nuc()+core)
    return u[0]+MF[0].energy_nuc()+core, r_fid/max_iter,qsci_basis

def Energy_farm_random(R,noise,Q):
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
        B=random_lossy(M, N, Q)
        print('Decoding...')
        model_set = NN_decoder(M, N, B)
        'compute integrals and PySCF meanfield object'
        stuff, MF= d.integrals(mols, functions, orb_choice)


        #calculate all non-zero contributions of the RDM, neglecting RDM symmetries
        base = fed.Indices(stuff[0], threshold = 10**-12)

        #compute groups (ACM, *right?*)
        compute = fed.RDM_grouper(base)
        #determines the measurement groups
        groups = fed.measurement_group(compute)

        #compute x-string
        ustrings = fed.xfunc(groups, np.identity(M))
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
            v=fed.nn_str_decode_general(CI_s[j], model_set, B)
            if v != '0'*M:
                qsci_basis_n.append(to10(v))
                # print(CI[j],v)
        print('QSCI...')
        qsci_basis1=np.unique(qsci_basis+qsci_basis_n,axis=0)
        if len(qsci_basis1)>2:
            mat = d.first_quantised(mols, one, two, qsci_basis1)
            u,w=sp.sparse.linalg.eigs(mat,k=1,which='SR')
            if abs(u-oldu)>1e-3:
                oldu=u
                qsci_basis=qsci_basis+qsci_basis_n
            qsci_basis1=np.unique(qsci_basis,axis=0)
            print(len(qsci_basis1))
            print('qsci:', u+MF[0].energy_nuc()+core)
    return u[0]+MF[0].energy_nuc()+core, r_fid/max_iter,qsci_basis

#%%
#generate NN decoder
def NN_decoder(M, N, B, num_epochs=1000, bias=[]):
    excite=np.array(list(range(N+1)))
    excite=excite[excite<=(M-Q)]
    return {
        N: fed.nn_parity(
            B,
            N,
            hidden=4 * M * N,
            num_epochs=num_epochs,
            excite=excite,
            bias=bias,
        )
    }

#%% C2H4_Random
import random
lc=int(np.ceil(Q/2))
if lc%2==1:
    lc-=1
def Bg(M,Q,lc):
    B=[]
    a=[]

    while len(B)<M:
        tmp=np.zeros(Q,dtype=np.int32)
        ls=random.sample(list(range(int(Q))),lc)
        tmp[ls]=1
        for _ in range(len(tmp)):
            atmp="".join(map(str, tmp))
        c=[2]
        c.extend(
            sum(np.array(list(a[i])) != np.array(list(atmp)))
            for i in range(len(a))
        )
        while min(c)<2:
            tmp=np.zeros(Q,dtype=np.int32)
            ls=random.sample(list(range(int(Q))),lc)
            tmp[ls]=1
            for i in range(len(tmp)):
                atmp="".join(map(str, tmp))
            c=[2]
            c.extend(
                sum(np.array(list(a[i])) != np.array(list(atmp)))
                for i in range(len(a))
            )
        B.append(tmp)
        a.append(atmp)

    return np.array(B).T

def random_lossy(M,N,Q):
    return Bg(M,Q,lc)

histE=[]
for Q in [12]*50:
    E,r_fid,qsci_basis=Energy_farm_random(20, 0.013, Q)
    histE.append((Q,E,np.sqrt(r_fid),len(np.unique(qsci_basis,axis=0))))

histE=np.array(histE)
np.savetxt('C2H4_random',histE)
#%% C2H4_Chem
hists = fed.hist_format(gs[0][0], basis, np.eye(M), ustrings[-1:])
CI_s=np.array(list(hists[-1].keys()))
CI=np.array(list(hists[-1].values()))
num=np.argsort(-abs(CI))[:200]
bias=CI_s[num]
def random_lossy(M,N,Q):
    return r.encoder(M, N, Q, 1, bias, bound=0)

histE2=[]
for Q in [12]*50:
    E,r_fid,qsci_basis=Energy_farm(20, 0.013, Q)
    histE2.append((Q,E,np.sqrt(r_fid),len(np.unique(qsci_basis,axis=0))))

histE2=np.array(histE2)
np.savetxt('C2H4_chem',histE2)
#%% C2H4_Chem_bias
hists = fed.hist_format(gs[0][0], basis, np.eye(M), ustrings[-1:])
CI_s=np.array(list(hists[-1].keys()))
CI=np.array(list(hists[-1].values()))
num=np.argsort(-abs(CI))[:200]
bias=CI_s[num]
def random_lossy(M,N,Q):
    return r.encoder(M, N, Q, 200,bias=bias, bound=0.99)

histE3=[]
for Q in [12]*50:
    E,r_fid,qsci_basis=Energy_farm(20, 0.013, Q, basis[num])
    histE3.append((Q,E,r_fid,len(np.unique(qsci_basis,axis=0))))

histE3=np.array(histE3)
np.savetxt('C2H4_chem_bias',histE3)
#%%
histE=np.loadtxt('C2H4_random',dtype=np.complex64)
histE2=np.loadtxt('C2H4_chem',dtype=np.complex64)
histE3=np.loadtxt('C2H4_chem_bias',dtype=np.complex64)
plt.plot(np.sort(histE[:,1]-ene[0][0])[::-1],label='Random Encoding')
plt.plot(np.sort(histE2[:,1]-ene[0][0])[::-1],label='Chemical Encoding')
plt.plot(np.sort(histE3[:,1]-ene[0][0])[::-1],label='Bias Chemical Encoding')
plt.legend()
plt.grid()
plt.xlabel('Sorted Case Number',fontsize=15)
plt.ylabel(r'Energy Difference (Hartree)',fontsize=15)
plt.yscale('log')