# Extinction drives emergent metastability in complex ecosystems
Code accompanying the paper:
[Extinction drives emergent metastability in complex ecosystems](https://arxiv.org/abs/2608.14416)

## Requirements
The code is written in Python 3.10 and requires the following packages:
- numpy (v2.2.6 is used for the paper)
- networkx (v3.4.2 is used for the paper)
- cupy (recommended, v14.0.1 with CUDA 12.2 is used for the paper)

> If GPU computing is not available, replace `cupy` with `numpy` in the code (`cp` -> `np`)

## Usage
Create a `results/` folder first. Each script takes the same command-line arguments:

```bash
python <script>.py S K V mu sigma config_idx
```

- `grlv.py` corresponds to the deterministic grLV model (Eq. 10)
- `kurtz.py` corresponds to the approximate Langevin equation (Eq. 9)
- `tau_leaping.py` corresponds to the rule-based model (Eq. 6)

Example:
```bash
python tau_leaping.py 1024 1024 1000 -1 1 0
```

### Parameters
| Argument | Description |
|---|---|
| `S` | number of species |
| `K` | average number of interaction partners per species |
| `V` | system size (controls the strength of demographic noise) |
| `mu` | mean interaction strength |
| `sigma` | interaction heterogeneity |
| `config_idx` | index labelling the disorder realisation / run, used for caching the interaction matrix and naming output files |

### Output format
Each script produces two `.npy` files per run, with filenames encoding the parameters above in the order `S_K_V_mu_sigma_config_idx`:
- `..._<params>.npy`: abundance trajectory
- `..._taue_<params>.npy`: per-species extinction time with `inf` for species that survive to maximum simulation time

## License
This code is released under the MIT License. See `LICENSE` for details.
