"""
tau_leaping.py

Tau-leaping simulation of the rule-based stochastic Lotka-Volterra model
(see Eq. 6 in the accompanying paper). Each species i undergoes:

    - intrinsic birth:      X_i  -> 2X_i          (rate lambda_b * n_i)
    - intrinsic death:      X_i  -> 0              (rate lambda_d * n_i)
    - self-regulation:      2X_i -> X_i            (rate n_i(n_i-1)/V)
    - amensalism (J < 0):   X_i + X_j -> X_j       (rate |J_ij| n_i n_j / V)
    - commensalism (J > 0): X_i + X_j -> 2X_i + X_j (rate J_ij n_i n_j / V)

Species abundances are represented as raw population counts n_i, related
to the rescaled abundance used in the paper via x_i = n_i / V.

The interaction matrix J is drawn once per config_idx from a Gaussian
ensemble (mean mu/K, std sigma/sqrt(K)) diluted on an Erdos-Renyi random
graph with average degree K, and cached to disk so repeated runs with the
same parameters reuse the same disorder realisation.

Usage:
    python tau_leaping.py S K V mu sigma config_idx

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
# only ~K others on average (Erdos-Renyi random graph with edge prob K/N).
if os.path.exists(filename_J):
    J = cp.load(filename_J)
else:
    J = cp.random.normal(mu/K,sigma/cp.sqrt(K),size=(S,S))
    G = nx.erdos_renyi_graph(S,K/S)
    A = cp.asarray(nx.to_numpy_array(G))
    J = J*A
    cp.save(filename_J,J)
    print('New matrix has been created')

# Birth/death rate constants
lambb = cp.ones(S)
lambd = cp.ones(S)*1e-5

# Initial condition: every species starts at abundance x_i(0) = 1,
# i.e. raw population count n_i(0) = V.
n = V*cp.ones(S)

def event_num(rate,tau):
    """Sample the number of reaction events occurring in a window tau,
    given a (vector of) per-event rate(s), via a Poisson draw."""
    return cp.random.poisson(rate*tau)

def birth(n):
    """Diagonal rate matrix for intrinsic birth events X_i -> 2X_i."""
    rate = lambb*n
    return cp.diag(rate)

def death(n):
    """Diagonal rate matrix for intrinsic death events X_i -> 0."""
    rate = lambd*n
    return cp.diag(rate)

def self_regulation(n):
    """Diagonal rate matrix for self-regulation events 2X_i -> X_i."""
    rate = n*(n-1)/V
    return cp.diag(rate)

def competition(n,J=J):
    """Rate matrix for amensalistic interactions X_i + X_j -> X_j,
    driven by the negative part of J (i.e. species i is harmed by j)."""
    J = J*(J<0)
    rate = cp.outer(n,n)*(-J)/V * (1-cp.eye(S))
    return rate

def cooperation(n,J=J):
    """Rate matrix for commensalistic interactions X_i + X_j -> 2X_i + X_j,
    driven by the positive part of J (i.e. species i benefits from j)."""
    J = J*(J>0)
    rate = cp.outer(n,n)*J/V * (1-cp.eye(S))
    return rate

# --- Simulation setup ---
t = 0
dt = 1e-3      # fixed tau-leaping time step
tu = 1e-1      # interval at which abundances are recorded into X
tmax = 1e+3    # total simulated time

eps = 0.2      # (currently unused) tolerance parameter for adaptive step-size tau-leaping
tn = int(tmax/tu)

# X stores the recorded abundance trajectory (rescaled by V) at intervals of tu
X = cp.zeros((tn+1,S))
X[0,:] = n.copy()/V

# Taue records the extinction time of each species (first time n_i hits 0);
# stays at +inf for species that survive to tmax.
Taue = cp.inf*cp.ones(S)

tdx = 1  # index into X for the next recorded snapshot
td = 0   # time accumulated since the last recorded snapshot

while t < tmax:
    # Total production rate matrix (birth + commensalistic gain)
    # and total loss rate matrix (death + self-regulation + amensalistic loss),
    # combining Eq. (6)'s reaction channels.
    Rp = birth(n) + cooperation(n)
    Rn = death(n) + self_regulation(n) + competition(n)

    # --- Adaptive step-size tau-leaping (disabled; fixed dt is used instead) ---
#    M = cp.matmul(Rp-Rn,cp.ones(S))
#    Q = cp.matmul(Rp+Rn,cp.ones(S))
#    g = 2 + 1/(n-1+1e-10)
#    g = 2

#    dt1 = cp.amin(cp.maximum(eps*n/g,1)/cp.abs(M))
#    dt2 = cp.amin(cp.maximum(eps*n/g,1)**2/Q)
#    dt = min([dt1,dt2])

    # Draw the number of production/loss events for each reaction pair (i,j)
    # over the interval dt (tau-leaping approximation).
    Ep = event_num(Rp, dt)
    En = event_num(Rn, dt)

    # Net change in each species' count: sum event contributions over all
    # interaction partners j (row-sum of the event-count matrices).
    dn = cp.matmul(Ep-En,cp.ones(S))

    n += dn

    # Record the extinction time the first time a species' count drops to
    # zero or below; only set once per species (idx guards against overwrite).
    idx = (Taue==cp.inf) & (n<=0)
    Taue[idx] = t
    n *= n>0  # clip: extinct species remain at exactly 0

    t += dt
    td += dt

    # Periodically snapshot the (rescaled) abundances into X
    if td > tu:

        X[tdx,:] = n/V
        tdx +=  1

        td -= tu

    # Early stop if the entire community has gone extinct
    if cp.sum(n) == 0:
        print('All species went extinct')
        break

# Print final time reached and the fraction of surviving species
print('%.4f %.4f'%(t,cp.mean(n>0)))

# Save the abundance trajectory and per-species extinction times
cp.save('results/qbd_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx),X)
cp.save('results/qbd_taue_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx),Taue)
