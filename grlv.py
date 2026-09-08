"""
grLV.py

Deterministic generalised random Lotka-Volterra (grLV) simulation
(see Eq. 10 in the accompanying paper):

    dx_i/dt = x_i * (1 - x_i + sum_j J_ij x_j)

This is the V -> infinity (noiseless) limit of the stochastic rule-based
model in tau_leaping.py, integrated here with an adaptive-step
Runge-Kutta-Fehlberg 4(5) (RKF45) scheme instead of simulated via
demographic noise.

The interaction matrix J is drawn once per config_idx from a Gaussian
ensemble (mean mu/K, std sigma/sqrt(K)) diluted on an Erdos-Renyi random
graph with average degree K, and cached to disk so repeated runs with the
same parameters reuse the same disorder realisation (consistent with the
convention used in tau_leaping.py).

Usage:
    python grLV.py S K V mu sigma config_idx

Arguments:
    S          number of species
    K          average number of interaction partners per species
    V          system size (kept for filename/parameter consistency with
               the stochastic model; V does not enter the deterministic
               dynamics itself)
    mu         mean of interaction strengths, scaled as mu/K
    sigma      std of interaction strengths, scaled as sigma/sqrt(K)
    config_idx index labelling this disorder realisation / run, used for
               caching J and naming output files
"""

import numpy as np, cupy as cp, networkx as nx
import sys, os
from scipy import integrate

if len(sys.argv)!=7:
    print('Usage: S K V mu sigma config_idx')
    exit()

S = int(sys.argv[1])          # number of species
K = int(sys.argv[2])          # average connectivity
V = int(sys.argv[3])          # system size (for filename consistency only)
mu = float(sys.argv[4])       # mean interaction strength (before /K scaling)
sigma = float(sys.argv[5])    # interaction heterogeneity (before /sqrt(K) scaling)
config_idx = int(sys.argv[6]) # realisation index, used for caching/output naming

filename_J = 'results/qbd_J_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx)

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

# Growth rate constant (lambda_b - lambda_d = 1 in the paper's normalisation)
lamb = cp.ones(S)
# Initial condition: every species starts at abundance x_i(0) = 1
x = cp.ones(S)

def LV(x):
    """Right-hand side of the deterministic grLV equation,
    dx/dt = x * (lamb - x + J x)."""
    return x*(lamb-x+cp.matmul(J,x))

# --- Adaptive-step RKF45 integrator settings ---
tol = 5*1e-14   # error tolerance used both for step-size control and as an
                # early-stopping criterion once the state has converged
t = 0
dt = 1e-2       # initial/maximum time step
tu = 1e-1       # interval at which abundances are recorded into X
tmax = 1e+3     # total simulated time

tn = int(tmax/tu)
X = cp.zeros((tn+1,S))   # recorded abundance trajectory at intervals of tu

xth = 1e-12               # abundance threshold at or below which a species
                           # is considered extinct
Taue = cp.inf*cp.ones(S)  # extinction time of each species (+inf if never
                           # crosses xth by tmax)

X[0,:] = x.copy()
tdx = 1  # index into X for the next recorded snapshot
td = 0   # time accumulated since the last recorded snapshot

while t < tmax:
    # --- RKF45 (Runge-Kutta-Fehlberg) stage evaluations ---
    k1 = dt*LV(x)
    k2 = dt*LV(x + (1/4)*k1)
    k3 = dt*LV(x + (3/32)*k1 + (9/32)*k2)
    k4 = dt*LV(x + (1932/2197)*k1 - (7200/2197)*k2 + (7296/2197)*k3)
    k5 = dt*LV(x + (439/216)*k1 - 8*k2 + (3680/513)*k3 - (845/4104)*k4)
    k6 = dt*LV(x - (8/27)*k1 + 2*k2 - (3544/2565)*k3 + (1859/4104)*k4 - (11/40)*k5)

    # 5th-order solution increment (used as the accepted step, i.e. "local
    # extrapolation") and the embedded 4th/5th-order error estimate
    dx = (16/135)*k1 + (6656/12825)*k3 + (28561/56430)*k4 - (9/50)*k5 + (2/55)*k6
    err = (1/360)*k1 - (128/4275)*k3 - (2197/75240)*k4 + (1/50)*k5 + (2/55)*k6

    # Step-size scale factor: h >= 1 means the error is within tolerance
    # and the step can be accepted; h < 1 means it should be rejected and
    # retried with a smaller dt.
    h = (tol/(2*cp.linalg.norm(err)))**(1/4)

    if cp.max(cp.abs(dx)) <= tol:
        # State has stopped changing (system has reached its fixed point):
        # stop integrating early.
        break

    elif h >= 1:
        # Step accepted: advance time and state.
        t = t + dt
        td = td + dt
        x = x + dx
        x = x*(x>0)
        # Record the extinction time the first time a species' abundance
        # drops to or below the threshold (guarded so it is only set once,
        # the first time this happens, per species).
        idx = (Taue==cp.inf) & (x<=xth)
        Taue[idx] = t
    
    else:
        # Step rejected: do not advance t or x; dt will be shrunk below
        # and the step retried.
        pass

    # Update the step size (capped at the initial dt), based on the
    # error-based scale factor h.
    dt = min(1e-2,h * dt)
    
    # Periodically snapshot the abundances into X (species below the
    # extinction threshold are recorded as exactly zero).
    if td > tu:
        X[tdx,:] = x*(x>xth)
        tdx += 1

        td -= tu
        print('%.6f %.6f'%(t,cp.mean(x>xth)))

# Save the abundance trajectory and per-species extinction times
cp.save('results/qLV_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx),X)
cp.save('results/qLV_taue_%d_%d_%d_%.2f_%.2f_%d.npy'%(S,K,V,mu,sigma,config_idx),Taue)
