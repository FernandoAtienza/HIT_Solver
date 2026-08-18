# Next thesis campaign — stationary Reynolds-number calibration

The previous thesis campaign is already useful for the thesis:

- N=256 and N=512 agree closely in bulk statistics at Mt=0.25.
- N=512 removes the residual high-k artifacts present at lower resolution.
- Two independent Mt=0.25 realizations give almost identical spectra.
- The integrated dilatational energy fraction and WENO activity increase strongly with Mt.

However, the Mach pilot also showed a systematic decrease in the statistically
stationary Taylor Reynolds number:

- Mt=0.10: Re_lambda ≈ 131.6
- Mt=0.25: Re_lambda ≈ 130.6
- Mt=0.50: Re_lambda ≈ 113.1
- Mt=0.60: Re_lambda ≈ 101.3

Therefore the final Mach comparison must not yet be presented as a fixed-Reynolds
study.

This campaign calibrates explicit viscosity at Mt=0.40, 0.50 and 0.60 to target
a stationary Re_lambda,2D ≈ 130 while keeping the frozen numerical method.

## Launch

```bash
cd ~/github/HIT_Solver

unzip -o ~/Downloads/hit_reynolds_calibration_campaign.zip -d .

chmod +x \
  scripts/run_hit_reynolds_calibration.sh \
  scripts/summarize_reynolds_calibration.py

nohup ./scripts/run_hit_reynolds_calibration.sh \
  > reynolds_calibration_launcher.log 2>&1 &

echo "PID: $!"
```

Monitor:

```bash
tail -f \
  ~/github/HIT_Solver/results/hit2d/reynolds_calibration_ReL130_N256/campaign.log
```

The next step after this calibration is the final 512² Mach-number campaign.
