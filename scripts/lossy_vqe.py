#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 21:20:55 2024

@author: kesson
"""

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
import numpy as np
from qiskit_aer import AerProvider
from qiskit_aer.backends import AerSimulator
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile
from qiskit.circuit import ParameterVector
import scipy as sp


def to10(s):
    a=[]
    for i in s:
        a.append(int(i,2))
    return np.array(a)

#The encoded qubits

def random_lossy(M,N,Q):
    B = r.encoder(M, N, Q, 1)
    return B

class generator_opt:
    
    def __init__(self, V, num_generations, population_size):
        self.V = V
        self.num_generations = num_generations
        self.population_size = population_size

    def fitness(self, B):
        # Calculate fitness (e.g., minimize overlapping dot products)
        total = len(np.unique(self.V.dot(B.T)%2,axis=0))
        total_overlap = len(self.V)-total # Count overlapping entries
        return total_overlap
    
    def crossover(self, parent1, parent2):
        # Assume parent1 and parent2 are numpy arrays
        crossover_point = np.random.randint(1, len(parent1) - 1)  # Avoid endpoints
        offspring1 = np.vstack((parent1[:crossover_point], parent2[crossover_point:]))
        offspring2 = np.vstack((parent2[:crossover_point], parent1[crossover_point:]))
        return offspring1, offspring2
    
    def mutate(self, matrix, mutation_rate):
        # Flip random positions with probability mutation_rate
        mask = np.random.rand(*matrix.shape) < mutation_rate
        matrix[mask] = 1 - matrix[mask]  # Flip 0s to 1s and vice versa
    
    def genetic_algorithm(self):
        population = [random_lossy(M,N,Q) for _ in range(self.population_size)]
    
        for generation in range(self.num_generations):
            population.sort(key=self.fitness)
            parents = population[:self.population_size // 2]
    
            # Apply crossover and mutation to create new offspring
            new_population = []
            for i in range(self.population_size // 2):
                parent1, parent2 = parents[i], parents[(i + 1)% len(parents)]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                self.mutate(offspring1, mutation_rate=0.01)  # Adjust mutation rate as needed
                self.mutate(offspring2, mutation_rate=0.01)
                new_population.extend([offspring1, offspring2])
    
            # Replace old individuals with offspring
            population = new_population
    
        best_matrix = population[0]
        print('encode score:',self.fitness(best_matrix))
        return best_matrix

def NN_decoder(M, N, B, num_epochs=2000):
    model_set={}
    model_set[N] = fed.nn_parity(B, N, hidden=2*M*N, num_epochs=num_epochs)
    train_list = list(range(1,N+1))
    train_list.remove(N)
    for i in train_list:
        model = fed.nn_parity(B, i, num_epochs=num_epochs)
        print('done')
        model_set[i]=model
    
    # get energy, ground state and the optional meanfield object.
    tableb=fed.look_up_parity(M, N, N, B)
    for i in tableb.keys():
        dd = fed.nn_str_decode(i, model_set, B)
        tableb[i]=dd
    return model_set, tableb

#%%
#prepare backend
backend = AerProvider().backends()[1]
backend = AerSimulator(method = 'density_matrix')
backend.shots = 100000

def sample(backend, circuit, layout, correction, count = 0):
    '''
    sample quantum circuit state distribution
    
    Parameters
    ----------
    circuit: qiskit circuit (no params)
    
    Returns
    -------
    dictionary of probability distribution
    
    '''
    val = backend.shots
    job = backend.run(circuit, shots = int(val))
    counts = job.result().get_counts()
    # evaluate probabilities
    shots = sum(counts.values())
    return {b: c/shots for b, c in counts.items()} 
def clifford_basis(circuit, xstrings):
    '''
    Generate measurement basis
    '''

    circuits = []
    
    for xstring in xstrings:
        cop = circuit.copy()
        count = 0
        cnot = []
        h = []
        for index, i in enumerate(xstring):
            if (i == 1) and count == 0:
                mem = index
                count=1
                h.append(index)
            elif (i == 1) and count:
                #print(mem, index)
                cnot.append((mem, index))
                mem = index
        cnot = cnot[::-1]
        for (i,j) in cnot:
            cop.cx(i, j)
        for i in h:
            cop.h(i)
        cop.measure_all()
        circuits.append(cop)
    return circuits

class input_p:
    '''
    mf: PySCF meanfield object.

    inte: core energy plus electronic integrals
    
    circuit: qiskit circuit.

    params: the parameters to combine. 

    xstrings: the encoded Pauli-x strings for the preparation of Clifford measurement basis.

    model: nn decoder.    

    groups: measurement groups for decoding.

    signs: global fermionic signs
    
    clf_basis: clifford basis
    '''
    def __init__(self, mf, inte, backend, cir,  xstrings, model, groups, signs, B):
        self.mf = mf
        self.i = inte
        self.c = cir
        self.x = xstrings
        self.m = model
        self.g = groups
        self.s = signs
        self.b = B
        self.backend = backend
        
def makecircuit(init, cnot, reps, name = 'gg', HF = 0, sing = 1):

    '''
    Make your hardware efficient quantum circuit.
    
    Input:
    ------
    orb: orbital number of your choice

    e: electron number after freeze

    A: encoder

    cnot: your onfiguration of cnot layer

    rep: the number of circuit repetition
    '''
    #that is the problem. 

    n=init.num_qubits
    param = ParameterVector(name,int(n*reps)+n*sing)
    t=0
    # cnot=rcnot(cnot_n,n)

    for l in range(reps):
        #cnot=rcnot(cnot_n,n)
        for i in range(n):
            init.ry(param[t],i)
            t+=1
        for (i, j) in cnot:
            init.cx(i,j)
        init.barrier()
    if sing == 1:
        for i in range(n):
            init.ry(param[t],i)  
            t+=1
    return init     

def to2(num, M):

    s = []
    while num:
        s.append(num % 2)
        num = num // 2

    additional_zeros = [0] * (M - len(s))
    m = np.array(s + additional_zeros, dtype = int)
    #m = ''.join(np.array(s + additional_zeros, dtype = str))
    return m

backend_s = AerSimulator(method = 'statevector')
def circuit_energy(params, mf, inte, backend, circuit, xstrings, table, groups, signs, A):
    '''
    Circuit energy only via the sampling means.

    Input:
    ------
    mf: PySCF meanfield object.

    inte: core energy plus electronic integrals
    
    circuit: qiskit circuit.

    params: the parameters to combine. 

    xstrings: the encoded Pauli-x strings for the preparation of Clifford measurement basis.

    table: look-up-table decoder replacable by classical decoder.    

    groups: measurement groups for decoding.

    signs: global fermionic signs 

    Output:
    -------
    energy: the energy
    '''

    'First step: you will just need circuit and assign_parameters'
    
    circuit = circuit.assign_parameters(params)

    
    circuits = clifford_basis(circuit, xstrings)

    'Those circuits could be wrong.'


    'In here, you must sample each of them, where we will need our backend information.'

    'prioritize sampling'
    hists = []
    data = []
    'check one stats'
    #count = 0

    'This part takes most of the time.'
    for circuit in circuits:
        stats = sample(backend, circuit, None, correction = 0, count = 0)
        keys = []
        
        for i in stats.keys():
            keys.append(i[::-1])
        stats = dict(map(lambda i,j : (i,j) , keys,stats.values()))

        #if count == 0:
            #stats = dict(sorted(stats.items(), key=lambda item: toint(item[0])))
            #example.append(stats)
            

        hists.append(stats)
        
    'we need to reverse the histogram I think.'
    
    da, ddaa, Vals = fed.rdm_val(hists, table, groups, signs, A, 0)

    
    one = inte[0]
    two = inte[1]
    core = inte[2]
    e = np.tensordot(one, da, axes=((0,1), (0,1)))+1/2*np.tensordot(two, ddaa, axes=((0,1,2,3), (0,1,2,3)))+MF[0].energy_nuc()+core
    return e

def circuit_energy_nn(params, init_input):
    '''
    Circuit energy only via the sampling means.

    Input:
    ------
    params : parameters for PQC
    input_p : initialize input
    Output:
    -------
    energy: the energy
    '''

    'First step: you will just need circuit and assign_parameters'
    circuit = init_input.c
    circuit = circuit.assign_parameters(params)
    circuits = clifford_basis(circuit, init_input.x)

    'In here, you must sample each of them, where we will need our backend information.'

    'prioritize sampling'
    hists = []
    'check one stats'
    #count = 0
    'This part takes most of the time.'
    for circuit in circuits:
        circuit= transpile(circuit,backend)
        stats = sample(backend, circuit, None, correction = 0, count = 0)
        keys = []
        
        for i in stats.keys():
            keys.append(i[::-1])
        stats = dict(map(lambda i,j : (i,j) , keys,stats.values()))
        hists.append(stats)
        
    da, ddaa, Vals = fed.nn_rdm_val_parallel(hists, init_input.m, init_input.g, init_input.s, init_input.b , 0)
    one = init_input.i[0]
    two = init_input.i[1]
    core = init_input.i[2]
    e = (np.tensordot(one, da, axes=((0,1), (0,1)))+1/2*np.tensordot(two, ddaa, axes=((0,1,2,3), (0,1,2,3)))
         +init_input.mf.energy_nuc()+core)
    return e

def get_Vec(params, init_input, table, form, mat, error=0):
    '''
    Vector energy: You will need one table for decoding, and one table for mapping to the matrix format

    In:
    ---
    table: send compressed strings to uncompressed ones

    form: send uncompressed string to the matrix coordinate in the subspace Hamiltonian.
    '''
    circuit = init_input.c
    circuit = circuit.assign_parameters(params)
    
    circuit.save_density_matrix(label="rho", conditional=True)
    circuit = transpile(circuit, backend)
    
    job = backend.run(circuit)
    result = job.result()

    rho = result.data()['rho'][''].data
    basis = []
    for i in range(len(rho)):
        if rho.diagonal()[i]>1e-5:
            key = ''.join(np.array(to2(i, circuit.num_qubits), dtype = str))
            basis.append(fed.str_decode(key, table))
    return basis

def get_shot(params, init_input, R):
    circuit = init_input.c
    circuit = circuit.assign_parameters(params)
    circuit.measure_all()
    stats = sample(backend, circuit, None, correction = 0, count = 0)
    keys = []
    values = []
    for i in stats.keys():
        key = fed.nn_str_decode(i, init_input.m, init_input.b)
        if key!='0'*len(key):
            keys.append(key)
            values.append(stats[i])
    
    return np.array(keys)[np.argsort(-np.array(values))[:R]]

def Simple_Vec(params, init_input, table, form, mat, error=0):
    '''
    Vector energy: You will need one table for decoding, and one table for mapping to the matrix format

    In:
    ---
    table: send compressed strings to uncompressed ones

    form: send uncompressed string to the matrix coordinate in the subspace Hamiltonian.
    '''
    circuit = init_input.c
    circuit = circuit.assign_parameters(params)
    
    circuit.save_density_matrix(label="rho", conditional=True)
    circuit = transpile(circuit, backend)
    
    job = backend.run(circuit)
    result = job.result()

    rho = result.data()['rho'][''].data
    basis = []
    for i in range(len(rho)):
        key = ''.join(np.array(to2(i, circuit.num_qubits), dtype = str))
        basis.append(form[fed.str_decode(key, table)])

    newrho = np.zeros((mat.shape[0], mat.shape[0]))
    for index, i in enumerate(basis):
        for jndex, j in enumerate(basis):
            newrho[i][j] = rho[index][jndex].real
            
    A = np.random.normal(0, error, (mat.shape[0], mat.shape[0]))
    A = (A+A.T)/2
    newrho = newrho+A
    something = newrho@mat
    core = init_input.i[2]
    return np.trace(something)+core+init_input.mf.energy_nuc()

def Simple_Vec_nn(params, init_input, form, mat, error = 0):
    '''
    Vector energy: You will need one table for decoding, and one table for mapping to the matrix format

    In:
    ---
    table: send compressed strings to uncompressed ones

    form: send uncompressed string to the matrix coordinate in the subspace Hamiltonian.
    '''
    circuit = init_input.c
    circuit = circuit.assign_parameters(params)
    circuit.save_density_matrix(label="rho", conditional=True)
    
    circuit = transpile(circuit, backend)
    job = backend.run(circuit)
    result = job.result()

    rho = result.data()['rho'][''].data
    basis = []
    for i in range(len(rho)):
        key = ''.join(np.array(to2(i, circuit.num_qubits), dtype = str))
        basis.append(form[fed.nn_str_decode(key, init_input.m, init_input.b)])
    newrho = np.zeros((mat.shape[0], mat.shape[0]))
    for index, i in enumerate(basis):
        for jndex, j in enumerate(basis):
            newrho[i][j] = rho[index][jndex].real
    A = np.random.normal(0, error, (mat.shape[0], mat.shape[0]))
    A = (A+A.T)/2
    newrho = newrho+A
    something = newrho@mat
    core = init_input.i[2]
    return np.trace(something)+core+init_input.mf.energy_nuc()

def circuit_energy_qsci(params, init_input, R):
    '''
    Circuit energy only via the sampling means.

    Input:
    ------
    params : parameters for PQC
    input_p : initialize input
    Output:
    -------
    energy: the energy
    '''

    'First step: you will just need circuit and assign_parameters'
    circuit = init_input.c
    circuit = circuit.assign_parameters(params)
    circuit.measure_all()
    'Those circuits could be wrong.'


    'In here, you must sample each of them, where we will need our backend information.'
    'check one stats'
    #count = 0
    'This part takes most of the time.'
    Ene=[]
    for i in range(10):
        stats = sample(backend, circuit, None, correction = 0, count = 0)
        keys = []
        
        for i in stats.keys():
            keys.append(fed.str_decode(i[::-1],tableb))
        stats = dict(map(lambda i,j : (i,j) , keys,stats.values()))
    
        mat_basis = sorted(stats, key=stats.get)[:R]
        print(mat_basis)
        opt_basis = []
        for i in mat_basis:
            opt_basis.append(np.array(list(i), dtype = int))
        
        one = init_input.i[0]
        two = init_input.i[1]
        core = init_input.i[2]  
        mat = d.first_quantised(init_input.mf.mol, one, two, opt_basis)
        u,w=sp.sparse.linalg.eigs(mat,k=1)
        Ene.append(min(u))
        print(min(u).real+init_input.mf.energy_nuc()+core)
    
    return min(Ene).real+init_input.mf.energy_nuc()+core

def to2_b(s,coef):
    a=[]
    for i in s:
        a.append(np.binary_repr(i,coef))
    return a

def rcnot(l,Q):
    cnot=[]
    while len(cnot)!=l:
        c=np.random.randint(0,Q)
        x=np.random.randint(0,Q)
        if c!=x:
            cnot.append((c,x))
    return cnot

#generate LiH molecule
dist = np.array([2.5])
atoms = ['Li','H']
Rs = np.array([0, 1])
thetas = [0, 0]
basis = 'sto-3g'
mult = 1
# Activate orbitals (ACM, *right?*)
orb_choice = [1,2,3,4,5]
#optional fermionic to hardcore bosonic encoding
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
basis = d.new_hot(mols[0], orb_choice[0], orb_choice, hardcore)
functions = [[d.compute_integrals, d.get_active_space_integrals, basis]]
ene, gs, MF, mat= d.spectrum(mols, functions, orb_choice[0], orb_choice)
stuff, MF = d.integrals(mols, functions, orb_choice)
one = stuff[0][0]
two = stuff[0][1]
core = stuff[0][2]
Q = 5
print('gs: ', ene[0][0])

odd = []
even = []

'One layer for demonstration'
reps = 4
for i in range(Q//2):
    'odd'
    odd.append((2*i, 2*i+1))
    even.append((2*i+1, 2*i+2))
if Q%2 == 0:
    even.pop()
cnot = odd+even

qr = QuantumRegister(Q, name = 'hea')
init = QuantumCircuit(qr)
cir = makecircuit(init, cnot, reps)
inte, MF = d.integrals(mols, functions, orb_choice)
#%%
stuff, MF = d.integrals(mols, functions, orb_choice)
one = stuff[0][0]
two = stuff[0][1]
core = stuff[0][2]
options = {
    'maxiter': 1000,  # maximum number of iterations
    'disp': False,    # display convergence messages
    'gtol': 1e-20,
}
# R=10   
history_x = []
history_e = []
def callback(xk):
    #store parameters
    history_x.append(xk)
    # history_e.append(circuit_energy_nn(xk, init_input))
    history_e.append(Simple_Vec(xk, init_input, tableb, form, mat))
    if len(history_e)%5==0:
        print(history_e[-1]-ene[0][0])

oldu=ene[1][0]-MF[0].energy_nuc()-core
qsci_basis=[to10('1'*N+'0'*(M-N))]
base = fed.Indices(inte[0], threshold = 10**-12)
E=[]
for j in range(25):
    B=random_lossy(M, N, Q)
    model_set, tableb = NN_decoder(M, N, B, num_epochs=500)
    init = QuantumCircuit(Q)
    init.h(list(range(Q)))
    # init.x(1)
    # cnot= rcnot(10, Q)
    cnot = odd+even
    cir = makecircuit(init, cnot, 2)
    'compute integrals and PySCF meanfield object'    
    #calculate all non-zero contributions of the RDM, neglecting RDM symmetries
    mat_basis = tableb.values()
    opt_basis = []
    for i in mat_basis:
        opt_basis.append(np.array(list(i), dtype = int))
        
    mat = d.first_quantised(MF[0].mol, one, two, opt_basis)
    mat_map = list(zip(np.arange(len(mat_basis)), list(mat_basis)))
    form = dict()
    for (a, b) in mat_map:
        form[b] = a   
    #compute groups (ACM, *right?*)
    compute = fed.RDM_grouper(base)
    #determines the measurement groups
    groups = fed.measurement_group(compute)
    
    #compute x-string
    ustrings = fed.xfunc(groups, np.identity(M))
    signs = fed.glob_sign(groups, M)
    params = np.random.normal(0, 2*np.pi, len(cir.parameters))
    xstrings = fed.xfunc(groups, B)
    init_input = input_p(MF[0], inte[0], backend, cir, xstrings, model_set, groups, signs, B)

    result = sp.optimize.minimize(fun = Simple_Vec, x0 = params, method='L-BFGS-B',
                                            args = (init_input, tableb, form, mat), 
                                            callback = callback, options = options)
    params = result.x
    E.append(result.fun)
    ci = get_shot(params, init_input, 5)
    qsci_basis_n=[]
    for i in ci:
        if sum(to10(i))==N:
            qsci_basis_n.append(to10(i))
    qsci_basis1=np.unique(qsci_basis+qsci_basis_n,axis=0)
    if len(qsci_basis1)>2 and result.fun<oldu:
        mat = d.first_quantised(mols, one, two, qsci_basis1)
        u,w=sp.sparse.linalg.eigs(mat,k=1)
        if u < oldu:
            oldu=u
            qsci_basis=qsci_basis+qsci_basis_n
        print(len(qsci_basis))
        print('gs: ', ene[0][0])
        print('qsci:', oldu+MF[0].energy_nuc()+core)
print(oldu+MF[0].energy_nuc()+core)
np.savetxt('LiH_E',E)
np.savetxt('LiH_qsci',qsci_basis)
E=np.loadtxt('lossy_qsci/LiH_E')
E_qsci=np.loadtxt('lossy_qsci/LiH_Eqsci',dtype=np.complex128)
x=list(range(3,20))
plt.plot(x,E_qsci,'r',label='Lossy QSCI (5 Qubits)',linewidth=2)
plt.plot(x,np.ones(len(x))*E[0],c='g',alpha=0.3,label='Collected VQE Results (5 Qubits)')
for i in E[1:]:
    plt.plot(x,np.ones(len(x))*i,c='g',alpha=0.3)

plt.plot(x,np.ones(len(x))*-7.7742077+1.59e-3,'--k',label='Chemical Accuracy (6,2)')
plt.plot(x,np.ones(len(x))*-7.8234269+1.59e-3,'--b',label='Chemical Accuracy (10,2)')
plt.legend(ncol=2,bbox_to_anchor=(1.05,1.2))
plt.xlabel('Number of QSCI basis (R)',fontsize=15)
plt.ylabel('Energy (Hartree)',fontsize=15)

    