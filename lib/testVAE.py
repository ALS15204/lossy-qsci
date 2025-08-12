#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 19:17:55 2024

@author: kesson
"""
import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__)))
import random
import torch
import numpy as np
import torch.nn as nn
import RLE as r

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def applyB(s,B):
    a=[]
    for v in s:
        v=B.dot(v)%2
        a.append(v)
    return torch.tensor(a, dtype=torch.float)

def generate_binary_vectors(M,Q,N,excite,sample):
    '''
    Generate Training data
    '''
    binary_vectors = set()
    vector=[]
    for i in range(len(excite)):
        for j in range(sample[i]):
            vector = np.zeros(M)
            vector[random.sample(list(range(Q,M)), excite[i])]=1
            vector[random.sample(list(range(Q)), N-excite[i])]=1
            binary_vectors.add(tuple(vector))
            
    return torch.tensor(list(binary_vectors), dtype=torch.float)

# class Autodecoder(nn.Module):
#     def __init__(self, M, hidden, Q, depth, N):
#         super(Autodecoder, self).__init__()
#         self.N = N
#         self.M = M
#         self.Q = Q
#         self.dummy = '0'*(M-N)+'0'*N
#         # Decoder
#         bc=[nn.Linear(Q, hidden),nn.PReLU()]
#         for i in range(depth):
#             bc.append(nn.Linear(hidden, hidden))
#             bc.append(nn.PReLU())
            
#         bc.append(nn.Linear(hidden, M))
#         bc.append(nn.Sigmoid())
#         self.decode = nn.Sequential(*bc)
        
    
#     def forward(self, x):
#         Q = x.shape[1]
#         z = x.view(-1, Q)
#         # Decode
#         recon_x = self.decode(z)
#         return recon_x, z
    
class Autoencoder(nn.Module):
    def __init__(self, M, hidden, Q, depth, data):
        super(Autoencoder, self).__init__()
        self.M = M
        self.Q = Q
        # Decoder
        bc=[nn.Linear(M, hidden),nn.PReLU()]
        for i in range(depth):
            bc.append(nn.Linear(hidden, hidden))
            bc.append(nn.PReLU())
            
        bc.append(nn.Linear(hidden, Q))
        bc.append(nn.Sigmoid())
        self.decode = nn.Sequential(*bc)
        
    
    def forward(self, x):
        # Decode
        B = self.decode(x.T)
        return B
    
M=12
N=2
Q=7
nQ=5
excite=[1]
sample=[10]
dataset = generate_binary_vectors(M,Q,N,excite,sample)
hidden = M*N
B = r.encoder(M, N, Q, 1)
B = torch.tensor(B,dtype=torch.float)

model = Autoencoder(Q, hidden, nQ, 1, dataset)
losses = []
# Training loop
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F

class SSCLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.03):
        super().__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, emb_i, emb_j):
        z_i = F.normalize(emb_i, dim=1)
        z_j = F.normalize(emb_j, dim=1)
        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, self.batch_size)
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([sim_ij, sim_ji], dim=0)

        nominator = torch.exp(positives / self.temperature)
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)

        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

B = r.encoder(M, N, Q, 1)
B = torch.tensor(B,dtype=torch.float)
criterion = SSCLoss(batch_size=M)
for epoch in range(300):
    # Generate the dataset
    optimizer.zero_grad()
    B_cut = model(B)
    loss = criterion(B_cut, B_cut)
    loss.backward()
    optimizer.step()
    losses.append(loss)
    if epoch%10==0:
        print(f'Epoch {epoch+1}, Loss: {loss}')