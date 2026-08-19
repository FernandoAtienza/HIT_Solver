# Final thesis HIT campaign

This is the intended final simulation campaign before freezing the solver and
moving exclusively to post-processing and thesis writing.

## Frozen numerical method

- hybrid compact/WENO7;
- legacy shock sensor;
- local Lax-Friedrichs WENO splitting;
- conservative compact-region hyperviscosity;
- \(m_n=0.005\);
- hyperviscosity every 5 complete RK steps;
- CFL = 0.10;
- solenoidal OU forcing in \(3 \le k_f \le 5\);
- Pr = 0.72.

## Campaign structure

1. Confirm the corrected viscosities at \(N=256\) for \(M_t=0.40\) and 0.60.
2. If each is within 5% of stationary \(Re_{\lambda,2D}=130\), run its final
   \(512^2\) case.
3. Run the already accepted \(M_t=0.10\), 0.25 and 0.50 cases at \(512^2\).
4. Automatically post-process every successful simulation over
   approximately \(4 \le N_{eddy}\le16\).

The code is intentionally not tuned further in this campaign.

## Install

```bash
cd ~/github/HIT_Solver

unzip -o ~/Downloads/final_thesis_hit_campaign.zip -d .

chmod +x \
  scripts/run_hit_final_thesis_campaign.sh \
  scripts/check_stationary_re.py \
  scripts/summarize_final_thesis_campaign.py
```

## Launch

```bash
cd ~/github/HIT_Solver

nohup ./scripts/run_hit_final_thesis_campaign.sh \
  > final_thesis_campaign_launcher.log 2>&1 &

echo "PID: $!"
```

Monitor:

```bash
tail -f \
  ~/github/HIT_Solver/results/hit2d/final_thesis_Mach_ReL130_N512/campaign.log
```

GPU:

```bash
watch -n 2 nvidia-smi
```
