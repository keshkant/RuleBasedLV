"""
kurtz.py

Milstein-scheme simulation of Kurtz's diffusion approximation to the
rule-based stochastic Lotka-Volterra model (see Eq. 9 in the accompanying
paper):

    dx_i = x_i (1 - x_i + sum_j J_ij x_j) dt
           + sqrt( x_i (1 + x_i + sum_j |J_ij| x_j) / V ) dW_i

This sits between the exact rule-based simulation (tau_leaping.py, which
simulates individual birth/death/interaction events) and the deterministic
grLV limit (grLV.py, V -> infinity): it keeps demographic noise but
replaces the discrete-event dynamics with a continuous SDE, integrated
here via the Milstein scheme (drift + diffusion + the O(dt) Milstein
correction term for multiplicative, state-dependent noise).

The interaction matrix J is drawn once per config_idx from a Gaussian
ensemble (mean mu/K, std sigma/sqrt(K)) diluted on an Erdos-Renyi random
graph with average degree K, and cached to disk so repeated runs with the
same parameters reuse the same disorder realisation (consistent with the
convention used in tau_leaping.py and grLV.py).

Usage:
    python kurtz.py S K V mu sigma config_idx

Arguments:
    S          number of species
    K          average number of interaction partners per species
    V          system size (controls the strength of demographic noise)
    mu         mean of interaction strengths, scaled as mu/K
    sigma      std of interaction strengths, scaled as sigma/sqrt(K)
    config_idx index labelling this disorder realisation / run, used for
               caching J and naming output files
"""

import numpy as np, cupy as cp, networkx as nx
import sys, os

if len(sys.argv) != 7:
    print('Usage: S K V mu sigma config_idx')
    exit()

S = int(sys.argv[1])          # number of species
K = int(sys.argv[2])          # average connectivity
V = int(sys.argv[3])          # system size (sets demographic noise scale)
mu = float(sys.argv[4])       # mean interaction strength (before /K scaling)
sigma = float(sys.argv[5])    # interaction heterogeneity (before /sqrt(K) scaling)
config_idx = int(sys.argv[6]) # realisation index, used for caching/output naming

filename_J = 'results/J_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx)

# --- Build (or load a cached) interaction matrix J ---
# J_ij ~ N(mu/K, sigma^2/K), then diluted so each species interacts with
# only ~K others on average (Erdos-Renyi random graph with edge prob K/S).
if os.path.exists(filename_J):
    J = cp.load(filename_J)
else:
    J = cp.random.normal(mu/K,sigma/cp.sqrt(K),size=(S,S))
    G = nx.erdos_renyi_graph(S,K/S)
    A = cp.asarray(nx.to_numpy_array(G))
    J = J*A
    cp.save(filename_J,J)
    
# Initial condition: every species starts at abundance x_i(0) = 1
x = cp.ones(S)

def kurtz_drift(x):
    """Deterministic (grLV) drift term a(x) = x*(1 - x + J x)."""
    return x*(1 - x + cp.matmul(J,x))

def kurtz_diffusion(x,vecZ):
    """Diffusion term b(x)*dW, with state-dependent noise amplitude
    b_i(x) = sqrt( x_i (1 + x_i + sum_j |J_ij| x_j) / V )."""
    return cp.sqrt(x*(1+x+cp.matmul(cp.abs(J),x))/V)*vecZ

def kurtz_correction(x,vecZ):
    """Milstein correction term (1/2) b(x) b'(x) (dW^2 - dt), required
    because the diffusion coefficient b(x) above depends on the state x
    (multiplicative noise). Simplifies to
    (1/4) * (1 + 2x_i + sum_j |J_ij| x_j) / V * (dW_i^2 - dt).
    The (x>0) mask suppresses this term for already-extinct species."""
    return (x>0)*(1/4)*((1+2*x+cp.matmul(cp.abs(J),x))/V)*vecZ

t = 0
dt = 1e-3      # fixed integration time step
tu = 1e-1      # interval at which abundances are recorded into X
tmax = 1e+3    # total simulated time
tn = int(tmax/tu)

X = cp.zeros((tn+1,S))    # recorded abundance trajectory at intervals of tu
Taue = cp.inf*cp.ones(S)  # extinction time of each species (+inf if never
                           # crosses 0 by tmax)

X[0,:] = x.copy()
tdx = 1  # index into X for the next recorded snapshot
td = 0   # time accumulated since the last recorded snapshot

while t < tmax:        
    # Increment of the driving Wiener process for this step, one
    # independent component per species.
    dW = cp.sqrt(dt)*cp.random.normal(0,1,size=S)
            
    drift = kurtz_drift(x)*dt
    diffusion = kurtz_diffusion(x,dW)
    correction = kurtz_correction(x,dW*dW-dt)
    
    # Milstein update: x_{n+1} = x_n + drift + diffusion + Milstein correction
    dx = drift + diffusion + correction

    x += dx

    # Record the extinction time the first time a species' abundance
    # drops to or below zero (guarded so it is only set once, the first
    # time this happens, per species).
    idx = (Taue==cp.inf) & (x<=0)
    Taue[idx] = t
    x *= x>0  # clip: extinct species remain at exactly 0

    t += dt
    td += dt
   
    # Periodically snapshot the abundances into X
    if td > tu:
        X[tdx,:] = x
        tdx += 1

        td -= tu

# Save the abundance trajectory and per-species extinction times
cp.save('results/kurtz_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx),X)
cp.save('results/kurtz_taue_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx),Taue)
