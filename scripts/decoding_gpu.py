#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 16:31:46 2025

@author: kesson
"""
import time
import math
from scipy.special import comb
import numpy as np
# from bitarray import bitarray
import random
import torch
from torch import nn

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class generator:
    '''Generator class

    Input:
    ------
    M: the number of modes

    N: the number of particles

    Q: the guessed compression

    maxiter: the maximal number of trials

    Ouput:
    ------
    multiple functions that generates the generator matrix
    '''


    def __init__(self, M, N, Q, maxiter):
        self.M = M
        self.N = N
        self.Q = Q
        self.hd = N
        if M-Q<N:
            self.bias = list(range(1,M-Q+1))
            self.bnum = [int(comb(M,N)/10)]*(M-Q)
        else:
            self.bias = list(range(1,N+1))
            self.bnum = [int(comb(M,N)/10)]*(N)
        
        def generate_binary_vectors(M,Q,N,excite,sample):
            '''
            Generate Training data
            '''
            binary_vectors = set()
            vector=[]
            for i in range(len(excite)):
                for j in range(sample[i]):
                    vector = ['0'] * M
                    for idx in random.sample(list(range(Q,M)), excite[i]):
                        vector[idx] = '1'
                    for idx in random.sample(list(range(Q)), N-excite[i]):
                        vector[idx] = '1'
                    binary_vectors.add(''.join(vector))
        
            return list(binary_vectors)
        
        s=generate_binary_vectors(self.M,self.Q,self.N,self.bias,self.bnum)

        lc=int(np.ceil(Q/2))
        if lc%2==1:
            lc-=1
        # lc = 2
        self.s = s
        self.lc = lc


        self.maxiter = maxiter

        #self.items = 0

    def modtest2(self):
        '''generator matrix with substantially simpler check
        '''
    #    s10=np.array(to10(s))
        aa=0
        for _ in range(self.maxiter):
            B,a=self.Bg()
            ps=[]
            for i in self.s:
                v=self.to10(list(i))
                v=B.dot(v)%2
                vs = ''.join('0' if j==0 else '1' for j in v)
                ps.append(vs)
            aa+=len(self.s)

            g = len(set(ps))
            f = len(self.s)
            print('Injectivity: ', g, f, g/f)
            if g/f>0.999:
                return B

    def Bg(self):
        # while 1:
        B=list(np.eye(self.Q,dtype=np.int32))
        # B=[]
        a=[]


        while len(B)<self.M:
            tmp=np.zeros(self.Q,dtype=np.int32)
            ls=random.sample(list(range(int(self.Q))),self.lc)
            tmp[ls]=1
            for _ in range(len(tmp)):
                atmp="".join(map(str, tmp))
            c=[self.hd]
            c.extend(
                sum(np.array(list(a[i])) != np.array(list(atmp)))
                for i in range(len(a))
            )
            while min(c)<self.hd:
                tmp=np.zeros(self.Q,dtype=np.int32)
                ls=random.sample(list(range(int(self.Q))),self.lc)
                tmp[ls]=1
                for i in range(len(tmp)):
                    atmp="".join(map(str, tmp))
                c=[self.hd]
                c.extend(
                    sum(np.array(list(a[i])) != np.array(list(atmp)))
                    for i in range(len(a))
                )
            B.append(tmp)
            a.append(atmp)

        B=np.array(B).T
        return B,a

    def to10(self, s):
        a=[]
        a.extend(int(i,2) for i in s)
        return np.array(a)

def encoder(orb, e, Q, maxiter=5000):
    '''Compute the compression matrix with the molecule information.

    Input:
    ------
    mol: molecule object.

    Q: the guessed qubit number.

    freeze: freezing core orbitals.

    RHF: restricted Hartree-Fock.

    Output:
    -------
    A: a (Q, M) matrix
    '''
    
    g = generator(orb, e, Q, maxiter)
    return g.modtest2()

def recover_v(B, e, M, N, Q, num_iterations=1000, initial_temp=1.0, final_temp=0.01, cooling_rate=0.95):
    '''
    Recover v from e using simulated annealing
    
    Input:
    ------
    B: compression matrix (Q x M)
    e: compressed Q-bit array
    M: number of modes
    N: number of particles
    Q: compressed bits
    
    Output:
    ------
    best_v: recovered M-bit array
    '''
    def calculate_error(v):
        compressed = B.dot(v) % 2
        return np.sum(np.abs(compressed - e))

    # Generate random initial v with N 1's
    current_v = np.zeros(M, dtype=np.int32)
    indices = random.sample(range(M), N)
    current_v[indices] = 1
    current_error = calculate_error(current_v)
    
    best_v = current_v.copy()
    best_error = current_error
    
    temp = initial_temp
    
    for i in range(num_iterations):
        # Generate neighbor by swapping a 1 and 0
        neighbor = current_v.copy()
        ones = np.where(neighbor == 1)[0]
        zeros = np.where(neighbor == 0)[0]
        
        if len(ones) > 0 and len(zeros) > 0:
            pos1 = np.random.choice(ones)
            pos2 = np.random.choice(zeros)
            neighbor[pos1], neighbor[pos2] = neighbor[pos2], neighbor[pos1]
            
            neighbor_error = calculate_error(neighbor)
            
            # Accept if better or probabilistically if worse
            delta = neighbor_error - current_error
            if delta < 0 or np.random.random() < np.exp(-delta / temp):
                current_v = neighbor
                current_error = neighbor_error
                
                if current_error < best_error:
                    best_v = current_v.copy()
                    best_error = current_error
        
        # Cool temperature
        temp = max(temp * cooling_rate, final_temp)
        
        # Early stopping if perfect solution found
        if best_error == 0:
            break
            
    return best_v

def recover_v_genetic(B, e, M, N, Q, population_size=100, generations=100):
    '''
    Recover v using genetic algorithm approach
    
    Input:
    ------
    B: compression matrix (Q x M) 
    e: compressed Q-bit array
    M: number of modes
    N: number of particles
    Q: compressed bits
    population_size: size of population in genetic algorithm
    generations: number of generations to evolve
    
    Output:
    ------
    best_v: recovered M-bit array
    '''
    import numpy as np
    
    # Helper function to calculate fitness
    def calculate_fitness(v):
        compressed = B.dot(v) % 2
        error = -np.sum(np.abs(compressed - e)) # Negative since we want to maximize
        return error
    
    # Initialize population - each individual has exactly N ones
    population = []
    for _ in range(population_size):
        v = np.zeros(M, dtype=np.int32)
        ones = np.random.choice(M, size=N, replace=False)
        v[ones] = 1
        population.append(v)
    
    # Evolution loop
    for gen in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = [calculate_fitness(v) for v in population]
        
        # Select parents using tournament selection
        def tournament_select():
            idx1, idx2 = np.random.randint(0, population_size, 2)
            if fitness_scores[idx1] > fitness_scores[idx2]:
                return population[idx1].copy()
            return population[idx2].copy()
        
        # Create new population
        new_population = []
        elite = population[np.argmax(fitness_scores)].copy() # Keep best solution
        new_population.append(elite)
        
        while len(new_population) < population_size:
            # Select parents
            parent1 = tournament_select()
            parent2 = tournament_select()
            
            # Crossover
            if np.random.random() < 0.8:  # 80% crossover rate
                # Ensure offspring maintains exactly N ones
                crossover_point = np.random.randint(1, M-1)
                child = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
                
                # Fix number of ones if needed
                num_ones = np.sum(child)
                if num_ones != N:
                    if num_ones > N:
                        ones = np.where(child == 1)[0]
                        to_zero = np.random.choice(ones, size=int(num_ones-N), replace=False)
                        child[to_zero] = 0
                    else:
                        zeros = np.where(child == 0)[0]
                        to_one = np.random.choice(zeros, size=int(N-num_ones), replace=False)
                        child[to_one] = 1
            else:
                child = parent1.copy()
            
            # Mutation
            if np.random.random() < 0.1:  # 10% mutation rate
                # Swap two positions (one 1 and one 0) to maintain N ones
                ones = np.where(child == 1)[0]
                zeros = np.where(child == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    pos1 = np.random.choice(ones)
                    pos2 = np.random.choice(zeros)
                    child[pos1], child[pos2] = child[pos2], child[pos1]
            
            new_population.append(child)
            
        population = new_population
        
    # Return best solution
    final_fitness = [calculate_fitness(v) for v in population]
    return population[np.argmax(final_fitness)]

def applyB(s,B):
    a=[]
    for v in s:
        # Move tensor to CPU before converting to numpy
        v_np = v.cpu().numpy() 
        v_compressed = B.dot(v_np)%2
        a.append(v_compressed)
    return torch.tensor(a, dtype=torch.float).to(device)

def generate_binary_vectors(M,Q,N,excite,sample):
    '''
    Generate Training data
    '''
    binary_vectors = set()
    vector=[]
    for i in range(len(excite)):
        for _ in range(sample[i]):
            vector = np.zeros(M)
            vector[random.sample(list(range(Q,M)), excite[i])]=1
            vector[random.sample(list(range(Q)), N-excite[i])]=1
            binary_vectors.add(tuple(vector))

    return torch.tensor(list(binary_vectors), dtype=torch.float).to(device)

class Autodecoder(nn.Module):
    def __init__(self, M, hidden, Q, depth, N):
        super(Autodecoder, self).__init__()
        self.N = N
        self.M = M
        self.Q = Q
        self.dummy = '0'*(N)+'0'*(M-N)
        # Decoder
        bc=[nn.Linear(Q, hidden),nn.PReLU()]
        for _ in range(depth):
            bc.append(nn.Linear(hidden, hidden))
            bc.append(nn.PReLU())

        bc.append(nn.Linear(hidden, M))
        bc.append(nn.Sigmoid())
        self.decode = nn.Sequential(*bc)

        # Move model to GPU
        self.to(device)
    
    def forward(self, x):
        Q = x.shape[1]
        z = x.view(-1, Q)
        # Decode
        recon_x = self.decode(z)
        return recon_x, z
    
    
def loss_function(recon_x, x, Q):
    return nn.functional.binary_cross_entropy(
        recon_x, x.view(-1, Q), reduction='sum'
    )

def nn_parity(B, N, depth=1, hidden=0, model=None, num_epochs=1000, lr=1e-2, excite=[], sample=[]):
    '''
    Neural Network decoders: Almost exactly decode all encoder's bitstrings
    '''
    (Q, M) = B.shape
    if len(excite)==0:        
        excite=np.array(list(range(1,N+1)))
        excite=excite[excite<=(M-Q)]
    if len(sample)==0:
        sample=[400]*(len(excite))
    if hidden ==0:
        hidden = M*N
    if model is None:
        model = Autodecoder(M, hidden, Q, depth, N)
    losses = []
    # Training loop
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        # Generate the dataset
        dataset = generate_binary_vectors(M,Q,N,excite,sample)
        edata=applyB(dataset,B)
        optimizer.zero_grad()
        recon_batch, z = model(edata)
        loss = loss_function(recon_batch, dataset, M)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())
        if epoch%100==0:
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss}')
        if epoch%200==0:
            lr=lr*0.95
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    return model.eval()
#%%
results = []
test_configs = [
   (30, 4), (40, 4), (50, 4), (60, 4), (70, 4)
]
num_tests = 1000

for M, N in test_configs:
    # Calculate Q based on information theory bound
    Q = math.ceil(math.log2(comb(M, 2*N)))
    print(f"\nTesting M={M}, N={N}, Q={Q}")
    
    B = encoder(M, N, Q)
    
    # Initialize metrics
    annealing_times = []
    genetic_times = []
    nn_times = []
    annealing_matches = 0
    genetic_matches = 0 
    nn_matches = 0

    # Train neural network model
    print("Training neural network model...")
    start_train_time = time.time()
    model_o = nn_parity(B, N,  hidden=M*M, num_epochs=M*90,lr=2e-3,
                        excite=list(range(1,N+1,2))) 
    model_e = nn_parity(B, N,  hidden=M*M, num_epochs=M*90,lr=2e-3,
                        excite=list(range(2,N+1,2)))
    train_time = time.time() - start_train_time
    print(f"Neural network training time: {train_time:.4f} seconds")

    for test in range(num_tests):
        # Generate random test vector with ones both inside and outside v[:Q]
        test_v = np.zeros(M, dtype=np.int32)
        n_in_Q = np.random.randint(1, min(N, Q))  # At least 1 in Q region
        n_after_Q = N - n_in_Q
        
        # Verify n_after_Q doesn't exceed available positions after Q
        if n_after_Q > (M - Q):
            n_in_Q = N - (M - Q)  # Adjust n_in_Q to fit remaining positions
            n_after_Q = M - Q     # Ta
            
        indices_in_Q = random.sample(range(Q), n_in_Q)
        indices_after_Q = random.sample(range(Q, M), n_after_Q)
        indices = indices_in_Q + indices_after_Q
        
        test_v[indices] = 1
        test_e = B.dot(test_v) % 2
        
        # Test annealing method
        start_time = time.time()
        recovered_annealing = recover_v(B, test_e, M, N, Q, 
                                      num_iterations=10**4,  # Increased from default
                                      initial_temp=20.0,     # Higher initial temperature
                                      final_temp=0.001,       # Lower final temperature
                                      cooling_rate=0.95) 
        annealing_time = time.time() - start_time
        annealing_times.append(annealing_time)
        if np.array_equal(test_v, recovered_annealing):
            annealing_matches += 1
            
        # Test genetic method
        start_time = time.time()
        recovered_genetic = recover_v_genetic(B, test_e, M, N, Q, 500, 500)
        genetic_time = time.time() - start_time
        genetic_times.append(genetic_time)
        if np.array_equal(test_v, recovered_genetic):
            genetic_matches += 1

        # Test neural network method
        start_time = time.time()
        test_e_tensor = torch.tensor(test_e, dtype=torch.float32).unsqueeze(0).to(device)
        if sum(test_e)%2:
            recovered_nn = model_o.decode(test_e_tensor)
        else:
            recovered_nn = model_e.decode(test_e_tensor)
        # recovered_nn = model.decode(test_e_tensor)
        recovered_nn = torch.round(recovered_nn).detach().cpu().numpy().astype(np.int32)
        recovered_nn = recovered_nn.squeeze()
        compressed = B.dot(recovered_nn) % 2
        if np.array_equal(compressed, test_e) and np.sum(recovered_nn) == N:
            nn_matches += 1
        nn_time = time.time() - start_time
        nn_times.append(nn_time)

    # Store results for this configuration
    results.append({
        'M': M,
        'N': N,
        'Q': Q,
        'train_time': train_time,
        'avg_annealing_time': np.mean(annealing_times),
        'avg_genetic_time': np.mean(genetic_times),
        'avg_nn_time': np.mean(nn_times),
        'annealing_success': annealing_matches/num_tests*100,
        'genetic_success': genetic_matches/num_tests*100,
        'nn_success': nn_matches/num_tests*100
    })
    print(results[-1])

# Print summary of all results

print("\nScaling Results Summary:")
print("------------------------")
# Save results to a text file and print to console
with open('decoding_results4.txt', 'w') as f:
    for result in results:
        output = (
            f"\nM={result['M']}, N={result['N']}, Q={result['Q']}\n"
            f"Neural Network Training Time: {result['train_time']:.4f} seconds\n"
            f"Average Decoding Times:\n"
            f"  Annealing: {result['avg_annealing_time']:.4f} seconds\n"
            f"  Genetic:   {result['avg_genetic_time']:.4f} seconds\n"
            f"  Neural Net: {result['avg_nn_time']:.4f} seconds\n"
            f"Success Rates:\n"
            f"  Annealing: {result['annealing_success']:.1f}%\n"
            f"  Genetic:   {result['genetic_success']:.1f}%\n"
            f"  Neural Net: {result['nn_success']:.1f}%\n"
        )
        f.write(output)
        print(output, end='')
#%%
results=[]
def parse_result_file(filename):
       results = []
       current_result = {}
       
       with open(filename, 'r') as f:
           lines = f.readlines()
           
       for line in lines:
           line = line.strip()
           if line.startswith('M='):
               # Start of new result block
               if current_result:
                   results.append(current_result)
               current_result = {}
               # Parse M, N, Q
               params = line.split(', ')
               current_result['M'] = int(params[0].split('=')[1])
               current_result['N'] = int(params[1].split('=')[1])
               current_result['Q'] = int(params[2].split('=')[1])
           elif 'Training Time:' in line:
               current_result['train_time'] = float(line.split(': ')[1].split()[0])
           elif 'Annealing:' in line:
               if 'seconds' in line:
                   current_result['avg_annealing_time'] = float(line.split(': ')[1].split()[0])
               else:
                   current_result['annealing_success'] = float(line.split(': ')[1].rstrip('%'))
           elif 'Genetic:' in line:
               if 'seconds' in line:
                   current_result['avg_genetic_time'] = float(line.split(': ')[1].split()[0])
               else:
                   current_result['genetic_success'] = float(line.split(': ')[1].rstrip('%'))
           elif 'Neural Net:' in line:
               if 'seconds' in line:
                   current_result['avg_nn_time'] = float(line.split(': ')[1].split()[0])
               else:
                   current_result['nn_success'] = float(line.split(': ')[1].rstrip('%'))
       
       # Add the last result
       if current_result:
           results.append(current_result)
           
       return results

# Read existing results
try:
    previous_results = parse_result_file('decoding_results4.txt')
except FileNotFoundError:
    previous_results = []

# Append new results
results.extend(previous_results)
#%%
import matplotlib.pyplot as plt

# Extract data for plotting
Ms = [r['M'] for r in results]
times_annealing = [r['avg_annealing_time'] for r in results]
times_genetic = [r['avg_genetic_time'] for r in results]
times_nn = [r['avg_nn_time'] for r in results]
success_annealing = [r['annealing_success'] for r in results]
success_genetic = [r['genetic_success'] for r in results]
success_nn = [r['nn_success'] for r in results]

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot average times
ax1.plot(Ms, times_annealing, 'o-', label='Annealing')
ax1.plot(Ms, times_genetic, 's-', label='Genetic')
ax1.plot(Ms, times_nn, '^-', label='Neural Net')
ax1.set_xlabel('Number of Modes (M)',fontsize=15)
ax1.set_ylabel('Average Time (seconds)',fontsize=15)
ax1.set_title('Average Decoding Times(per Sample)',fontsize=15)
ax1.set_yscale('log')  # Set y-axis to logarithmic scale
ax1.legend()
ax1.grid(True)

# Plot success rates
ax2.plot(Ms, success_annealing, 'o-', label='Annealing')
ax2.plot(Ms, success_genetic, 's-', label='Genetic')
ax2.plot(Ms, success_nn, '^-', label='Neural Net')
ax2.set_xlabel('Number of Modes (M)',fontsize=15)
ax2.set_ylabel('Success Rate (%)',fontsize=15)
ax2.set_title('Decoding Success Rates',fontsize=15)
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()