# Resource-Constrained Spatial Public Goods Game

Core dynamical implementation accompanying the manuscript **Resource-Constrained Spatial Public Goods Games with Effective Cooperation and Frozen Dynamics**.

## Scope

This repository provides the baseline model on a periodic square lattice with resource-dependent group activation and a freezing neighborhood of Manhattan distance at most 2 (13 sites). Parameter-scan orchestration, manuscript datasets, bootstrap analyses, and figure-generation pipelines are maintained separately.

## Files

- `core_model.py`: payoff calculation, resource accumulation, freezing, pairwise payoff comparison with a Fermi imitation probability, and a simulation driver.
- `requirements.txt`: NumPy and Numba versions used for this package.

## Installation

Use Python 3.12. Install the dependencies from this directory:

```bash
python -m pip install -r requirements.txt
```

The first simulation call compiles the numerical functions with Numba and may take longer than later calls.

## Small example

Run the following in Python from this directory:

```python
from core_model import run_simulation

params = {
    "r": 4.5,
    "c": 1.0,
    "alpha": 0.1,
    "theta": 0.1,
    "mu": 0.5,
    "K_fermi": 0.5,
}

result = run_simulation(L=20, steps=200, mc=1, rho=1.0, params=params)
cooperation, effective_cooperation, frozen_fraction, active_cooperation = result[:4]
print("Final overall cooperation:", cooperation[-1])
print("Final effective cooperation:", effective_cooperation[-1])
print("Final frozen fraction:", frozen_fraction[-1])
print("Final active cooperation:", active_cooperation[-1])
```

This is a small usage example. Manuscript simulation sizes, run lengths, independent repetitions, and averaging windows are specified in the paper. Simulation trajectories are stochastic; this example does not prescribe a random seed.

## Model conventions

- `S == 0` denotes cooperation; `S == 1` denotes defection. The manuscript setting uses full occupancy (`rho=1.0`).
- The accumulated resource account `P_accum` determines whether a player can organize its own group at the start of a round. A nonnegative balance permits organization. A player with a negative balance may still participate in an activated neighboring group.
- Round net payoffs are added to the resource accounts before freezing is assessed.
- A player is frozen when all updated resource balances in its extended neighborhood are negative. Freezing suspends its own strategy updating.
- Imitation is synchronous and uses the old strategy field and the same round's net payoffs. Non-frozen players may imitate frozen nearest neighbors.
- The effective cooperation ratio is the number of non-frozen cooperators divided by the population size. The active cooperation ratio is the cooperator fraction within the non-frozen population. These observables characterize strategy composition and updating eligibility; realized game participation is a separate quantity.
- When the non-frozen population is empty, active cooperation is recorded as zero and included in averages over all runs.

## Parameters

| Parameter | Meaning |
| --- | --- |
| `r` | Public-goods multiplication factor |
| `c` | Cooperative contribution per activated group |
| `alpha` | Organizer commission ratio |
| `theta` | Organizational cost per group position; an activated five-player group costs its organizer `5 * theta` |
| `mu` | Recurring individual cost per round |
| `K_fermi` | Noise parameter in the Fermi imitation probability; use a positive value |
| `L` | Side length of the periodic lattice |
| `steps` | Number of simulated rounds |
| `mc` | Number of independent realizations |

## Relation to the manuscript

This release shares the core dynamical implementation. Reproducing a particular manuscript figure additionally requires its experiment configuration, simulation records, and analysis procedure.
