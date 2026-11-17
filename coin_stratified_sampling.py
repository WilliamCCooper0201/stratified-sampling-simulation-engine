"""
STA4222 Project 3: Stratified Sampling and Regression Estimation

Research Question:
Do physical differences between US coin types produce statistically detectable
biases in flip outcomes, and does stratifying by coin type improve the precision
and stability of population mean estimation compared to simple random sampling?
"""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

np.random.seed(42)

# ── Coin Properties (US Mint specs) ──────────────────────────────────────────
COINS = {
    'penny':   {'weight': 2.500, 'diameter': 19.05, 'p': 0.502},
    'nickel':  {'weight': 5.000, 'diameter': 21.21, 'p': 0.500},
    'dime':    {'weight': 2.268, 'diameter': 17.91, 'p': 0.501},
    'quarter': {'weight': 5.670, 'diameter': 24.26, 'p': 0.503},
}
N_PER_STRATUM = 10_000
N_SAMPLE      = 500      # per stratum
N_SRS         = 2_000
N_BOOTSTRAP   = 1_000
N_FLIPS       = 100
FAIR_P        = 0.500
SAVE_DIR      = ''  # plots save to the current directory

coin_order = list(COINS.keys())
colors     = ['#4C72B0', '#DD8452', '#55A868', '#C44E52']

# ── 1. Simulate Population ───────────────────────────────────────────────────
rows = []
for coin, props in COINS.items():
    heads = np.random.binomial(N_FLIPS, props['p'], N_PER_STRATUM)
    for h in heads:
        rows.append({
            'coin_type':   coin,
            'weight':      props['weight'],
            'heads_count': h,
            'deviation':   h - (N_FLIPS * FAIR_P),   # observed - expected
        })

population = pd.DataFrame(rows)
print(f"Population created: {len(population):,} trials")

# ── 2. Stratified Sample (500 per coin, equal allocation) ────────────────────
strat_parts = []
for coin in COINS:
    subset = population[population['coin_type'] == coin]
    strat_parts.append(subset.sample(N_SAMPLE, random_state=42))
strat_sample = pd.concat(strat_parts).reset_index(drop=True)
print(f"Stratified sample: {len(strat_sample):,} trials (500 per coin)")

# ── 3. Simple Random Sample ───────────────────────────────────────────────────
srs_sample = population.sample(N_SRS, random_state=42).reset_index(drop=True)
print(f"SRS sample: {len(srs_sample):,} trials\n")

# ── 4. Regression — separate model per stratum with coin dummies ─────────────
# Single model with coin-type dummy variables (stratum indicators)
# Y = heads_count, X = weight + coin_type dummies
# This satisfies Section 5: "single regression model including stratum indicators"

dummies     = pd.get_dummies(strat_sample['coin_type'], drop_first=True).astype(float)
X_reg       = sm.add_constant(dummies)
model       = sm.OLS(strat_sample['heads_count'], X_reg).fit()

print("Regression: heads_count ~ coin_type dummies (weight dropped: collinear with coin identity)")
print(model.summary().tables[1])
print(f"\nR² = {model.rsquared:.4f}")

b_weight = 0.0  # weight dropped (collinear with coin dummies); regression adjustment term is ~0 by design

# ── 5. Regression Estimator (Stratified) ─────────────────────────────────────
def regression_estimate(sample_df):
    W     = 1 / len(COINS)
    total = 0.0
    for coin, props in COINS.items():
        data_h  = sample_df[sample_df['coin_type'] == coin]
        if len(data_h) < 2:
            return np.nan
        y_bar_s  = data_h['heads_count'].mean()
        x_bar_s  = data_h['weight'].mean()
        x_bar_U  = props['weight']
        mu_reg_h = y_bar_s + (x_bar_U - x_bar_s) * b_weight
        total   += W * mu_reg_h
    return total

mu_strat = regression_estimate(strat_sample)

# ── 6. Bootstrap SE ───────────────────────────────────────────────────────────
boot_estimates = []
for _ in range(N_BOOTSTRAP):
    boot_parts = []
    for coin in COINS:
        data_h = strat_sample[strat_sample['coin_type'] == coin]
        boot_parts.append(data_h.sample(len(data_h), replace=True))
    est = regression_estimate(pd.concat(boot_parts))
    if not np.isnan(est):
        boot_estimates.append(est)

se_strat  = np.std(boot_estimates, ddof=1)
ci_lo_str = mu_strat - 1.96 * se_strat
ci_hi_str = mu_strat + 1.96 * se_strat

# ── 7. Poststratification ─────────────────────────────────────────────────────
W     = 1 / len(COINS)
mu_ps = sum(W * srs_sample[srs_sample['coin_type'] == c]['heads_count'].mean()
            for c in COINS)

boot_ps = []
for _ in range(N_BOOTSTRAP):
    bs  = srs_sample.sample(len(srs_sample), replace=True)
    boot_ps.append(sum(W * bs[bs['coin_type'] == c]['heads_count'].mean()
                       for c in COINS if c in bs['coin_type'].values))
se_ps    = np.std(boot_ps, ddof=1)
ci_lo_ps = mu_ps - 1.96 * se_ps
ci_hi_ps = mu_ps + 1.96 * se_ps

# ── 8. SRS Estimate ───────────────────────────────────────────────────────────
mu_srs   = srs_sample['heads_count'].mean()
se_srs   = srs_sample['heads_count'].std(ddof=1) / np.sqrt(len(srs_sample))
ci_lo_sr = mu_srs - 1.96 * se_srs
ci_hi_sr = mu_srs + 1.96 * se_srs

# ── 9. True Population Mean ───────────────────────────────────────────────────
true_mean = population['heads_count'].mean()

# ── 10. Comparison Table ──────────────────────────────────────────────────────
results = pd.DataFrame({
    'Method':   ['Stratified Regression', 'Poststratification', 'Simple Random Sample'],
    'Estimate': [mu_strat,  mu_ps,   mu_srs],
    'SE':       [se_strat,  se_ps,   se_srs],
    'CI_Lower': [ci_lo_str, ci_lo_ps, ci_lo_sr],
    'CI_Upper': [ci_hi_str, ci_hi_ps, ci_hi_sr],
    'CI_Width': [ci_hi_str - ci_lo_str,
                 ci_hi_ps  - ci_lo_ps,
                 ci_hi_sr  - ci_lo_sr],
})
results[['Estimate','SE','CI_Lower','CI_Upper','CI_Width']] = \
    results[['Estimate','SE','CI_Lower','CI_Upper','CI_Width']].round(4)

print("\n" + "="*70)
print("COMPARISON TABLE")
print("="*70)
print(results.to_string(index=False))
print(f"\nTrue population mean: {true_mean:.4f}")

print("\nPer-stratum population means & deviation from fair (p=0.500):")
for coin in coin_order:
    m = population[population['coin_type'] == coin]['heads_count'].mean()
    print(f"  {coin}: mean={m:.4f}, deviation={m - 50:.4f}")

# ── 11. SRS Instability Demonstration ────────────────────────────────────────
# Repeat sampling 200 times, record estimate each time for both methods
N_REPS = 200
srs_reps   = []
strat_reps = []

for i in range(N_REPS):
    # SRS
    s = population.sample(N_SRS, random_state=i)
    srs_reps.append(s['heads_count'].mean())
    # Stratified
    parts = [population[population['coin_type'] == c].sample(N_SAMPLE, random_state=i)
             for c in COINS]
    strat_reps.append(pd.concat(parts)['heads_count'].mean())

print(f"\nInstability — SD of estimates over {N_REPS} repeated samples:")
print(f"  SRS:        {np.std(srs_reps):.4f}")
print(f"  Stratified: {np.std(strat_reps):.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# ── Plot 1: Boxplot — heads distribution by coin type ────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
data_by_coin = [population[population['coin_type'] == c]['heads_count'].values
                for c in coin_order]
bp = ax.boxplot(data_by_coin, patch_artist=True, tick_labels=coin_order)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.axhline(50, color='gray', linestyle='--', alpha=0.7, label='Fair coin (50)')
ax.set_title('Distribution of Heads Count by Coin Type', fontweight='bold')
ax.set_xlabel('Coin Type')
ax.set_ylabel('Heads Count (out of 100 flips)')
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR + 'boxplot_by_stratum.png', dpi=150)
plt.show()

# ── Plot 2: Separate regression per stratum (observed vs expected) ────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True)
axes = axes.flatten()
expected = N_FLIPS * FAIR_P  # 50.0

for i, (coin, color) in enumerate(zip(coin_order, colors)):
    ax   = axes[i]
    data = strat_sample[strat_sample['coin_type'] == coin]
    X_c  = sm.add_constant(data['weight'])
    fit  = sm.OLS(data['heads_count'], X_c).fit()

    ax.scatter(data['weight'], data['heads_count'],
               color=color, alpha=0.2, s=10, label='Observed')
    ax.axhline(expected, color='black', linestyle='--', linewidth=1.2,
               label='Expected (50)')
    ax.axhline(data['heads_count'].mean(), color=color, linestyle='-',
               linewidth=1.5, label=f'Observed mean ({data["heads_count"].mean():.2f})')
    ax.set_title(f'{coin.capitalize()}  (p={COINS[coin]["p"]})', fontweight='bold')
    ax.set_xlabel('Weight (g)')
    ax.set_ylabel('Heads Count')
    ax.legend(fontsize=7)

fig.suptitle('Observed vs Expected Heads by Coin Type (Stratified Sample)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(SAVE_DIR + 'regression_by_stratum.png', dpi=150)
plt.show()

# ── Plot 3: SRS instability vs Stratified ────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(srs_reps,   color='#C44E52', alpha=0.7, linewidth=0.8, label='SRS')
ax.plot(strat_reps, color='#4C72B0', alpha=0.7, linewidth=0.8, label='Stratified')
ax.axhline(true_mean, color='black', linestyle='--', linewidth=1.2,
           label=f'True mean ({true_mean:.3f})')
ax.set_title('Estimation Stability: SRS vs Stratified Sampling\n(200 repeated samples)',
             fontweight='bold')
ax.set_xlabel('Sample Replicate')
ax.set_ylabel('Estimated Mean Heads Count')
ax.legend()
plt.tight_layout()
plt.savefig(SAVE_DIR + 'instability_srs_vs_stratified.png', dpi=150)
plt.show()

# ── Plot 4: CI width comparison ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
methods   = ['Stratified\nRegression', 'Poststrat-\nification', 'Simple Random\nSample']
ci_widths = [ci_hi_str - ci_lo_str, ci_hi_ps - ci_lo_ps, ci_hi_sr - ci_lo_sr]
bar_colors = ['#4C72B0', '#55A868', '#C44E52']
bars = ax.bar(methods, ci_widths, color=bar_colors, alpha=0.8, edgecolor='black')
for bar, w in zip(bars, ci_widths):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f'{w:.4f}', ha='center', va='bottom', fontsize=10)
ax.set_title('95% Confidence Interval Width by Estimation Method', fontweight='bold')
ax.set_ylabel('CI Width')
plt.tight_layout()
plt.savefig(SAVE_DIR + 'ci_width_comparison.png', dpi=150)
plt.show()

print("\nDone. All plots saved to", SAVE_DIR)


