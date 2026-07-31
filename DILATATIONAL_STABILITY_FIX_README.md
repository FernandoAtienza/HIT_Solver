# Stabilized dilatational HIT runs

## What the fix changes

The original nested controllers reacted at every numerical step:

1. the outer Mach controller multiplied the requested power by the instantaneous Mach error;
2. the forcing controller divided that power by the instantaneous velocity–forcing correlation.

For purely dilatational forcing, the response is acoustic and delayed, while the instantaneous correlation can become small or change sign. The combination produces controller windup and large alternating peaks in injected power, kinetic energy, and turbulent Mach number.

The patch:

- preserves the accepted solenoidal forcing behavior;
- orients only the dilatational realization to prevent sign chatter while retaining a curl-free field;
- relaxes the dilatational force amplitude in physical time;
- filters both turbulent Mach number and kinetic energy;
- uses bounded logarithmic power changes with a dead band and rate limit;
- applies explicit anti-windup power bounds;
- stores filtered controller quantities in `diagnostic_history.npz`;
- creates backups before modifying the repository.

## Apply the patch

From any terminal:

```bash
python3 /path/to/stabilize_dilatational_hit.py ~/github/HIT_Solver
```

The script modifies:

```text
~/github/HIT_Solver/OOP/forcing.py
~/github/HIT_Solver/OOP/hit2d.py
```

Backups are created with:

```text
.before_stable_dilatational
```

The modified files are syntax-checked automatically.

## First validation at 128 x 128

Copy the stable campaign script into `~/github/HIT_Solver/scripts/`, then run:

```bash
cd ~/github/HIT_Solver
NX=128 SNAPSHOT_EVERY=5000 ./scripts/run_dilatational_hit_campaign_stable.sh
```

Do not launch the 512 x 512 production runs until the 128 x 128 histories show stationary behavior.

## Production runs at 512 x 512

```bash
cd ~/github/HIT_Solver
NX=512 SNAPSHOT_EVERY=10000 ./scripts/run_dilatational_hit_campaign_stable.sh
```

Use `SNAPSHOT_EVERY=5000` instead when disk space permits and smoother post-processing histories are desired.

## Expected stationary values

Because the initial state uses `rho = 1` and `p = 1/gamma`, its reference sound speed is one. The consistent kinetic-energy targets are therefore:

```text
Mt = 0.25 -> K_target = 0.5 * Mt^2 = 0.03125
Mt = 0.50 -> K_target = 0.5 * Mt^2 = 0.12500
```

Evaluate stationarity over `4 <= N_eddy <= 16`.

Useful checks:

- no sustained alternating peaks in `K`, `Mt`, or requested forcing power;
- mean pressure remains close to `1/gamma = 0.7142857`;
- time-averaged `Mt` and `K` remain near their targets;
- mean component anisotropy `A_K` and covariance `C_uv` remain small;
- directional longitudinal/transverse correlations agree within their sampling uncertainty;
- mass error remains negligible.

Purely dilatational forcing will still exhibit stronger divergence and acoustic fluctuations than solenoidal forcing. The goal is statistical stationarity, homogeneity, and isotropy—not a perfectly constant instantaneous signal.
