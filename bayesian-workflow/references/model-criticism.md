# Model Criticism

Model criticism answers: "Is this model any good?" Convergence diagnostics (reference/diagnostics.md) only tell you the sampler worked -- they say nothing about whether the model is appropriate for the data.

## Contents
- Posterior predictive checks (PPC)
- Leave-one-out cross-validation (LOO-CV)
- Calibration assessment
- Simulation-based calibration (SBC)
- Residual analysis
- Decision workflow

## Posterior predictive checks (PPC)

The most important model criticism tool. Simulate data from the fitted model and compare to observed data.

```python
with model:
    idata.extend(pm.sample_posterior_predictive(idata, random_seed=rng))

# Visual check: do simulated datasets resemble the real data?
az.plot_ppc(idata)
```

**What to look for**:
- Posterior predictive distribution should envelop the observed data
- Check shape, spread, and key features (skewness, multimodality, tails)
- Systematic departures indicate model misspecification

**Targeted PPCs** — Check specific data features the model should capture, using `az.plot_ppc` 
to isolate parameters of interest (e.g the posterior standard deviation, to check if model captures the observed variance).

Choose test statistics relevant to your problem: mean, variance, skewness, max, proportion above threshold, autocorrelation (for time series), etc.

## Leave-one-out cross-validation (LOO-CV)

Estimates out-of-sample predictive accuracy using Pareto-smoothed importance sampling (PSIS-LOO). This is the primary tool for model comparison but also useful for single-model criticism.

```python
loo = az.loo(idata, pointwise=True)
print(loo)
```

**Key outputs**:
- `elpd_loo`: Expected log pointwise predictive density. Higher (less negative) is better.
- `p_loo`: Effective number of parameters. If p_loo >> actual parameter count, the model may be misspecified or the priors are too weak.
- `pareto_k`: Per-observation diagnostic. Flags influential or poorly-fit observations.

**Pareto k diagnostic** — Critical for trusting LOO results:

| Pareto k | Interpretation | Action |
|---|---|---|
| < 0.5 | Reliable | Trust LOO estimate |
| 0.5–0.7 | Marginally reliable | Investigate flagged observations |
| > 0.7 | Unreliable for that observation | Use K-fold CV or moment matching |

```python
# Find problematic observations
pareto_k = loo.pareto_k.values
bad_obs = np.where(pareto_k > 0.7)[0]
print(f"Observations with high Pareto k: {bad_obs}")

# Visualize
az.plot_khat(loo)
```

High Pareto k observations are often outliers or observations the model fits poorly. Investigate them — they may reveal model misspecification.

## Calibration assessment

Calibration is mandatory for every model, not optional. A well-calibrated model's X% credible intervals should contain the true value about X% of the time. Run this even for binary and count data — ArviZ handles all data types correctly.

### How to run calibration

Always use ArviZ for calibration plots. Don't write custom calibration code — ArviZ's `plot_ppc_pit` handles continuous, binary, and count data correctly out of the box:

```python
# ArviZ 1.0+ (arviz_plots)
import arviz_plots as azp

# PPC-PIT: compares posterior predictive to observed
azp.plot_ppc_pit(idata)

# LOO-PIT: leave-one-out calibration (more robust, preferred when LOO is available)
azp.plot_ppc_pit(idata, loo_pit=True)
```

Refer to [this guide](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#coverage) for detailed coverage interpretation — it's a treasure trove for the whole Bayesian workflow.

### Coverage calibration

**Interpretation**:
- If empirical coverage ≈ nominal → well-calibrated
- If the difference is positive, the model is under-confident: the predictions have a wider spread than the data – they are too uncertain.
- If the difference is negative, the model is over-confident: the predictions have a narrower spread than the data – they are too certain.

### PIT histograms (probability integral transform)

A sharper calibration check. If the model is calibrated, PIT values should be uniform. Refer to [this section](https://arviz-devs.github.io/EABM/Chapters/Prior_posterior_predictive_checks.html#pit-ecdfs) for how to do it, using the new ArviZ.

**Patterns**:
- U-shaped → underdispersed (intervals too narrow)
- Inverted-U → overdispersed (intervals too wide)
- Skewed → systematic bias in location
- Uniform → well-calibrated

## Simulation-based calibration (SBC)

SBC validates that the entire inference pipeline is correct — priors, likelihood, sampler, and code. It simulates data from the prior, fits the model, and checks that posterior rank statistics are uniform.

This is the gold standard for validating a new model implementation. Run it once per model specification, if you have doubts about the model, since SBC is computationally expensive.

Use the [simuk package](https://github.com/arviz-devs/simuk), either directly, or as inspiration to adapt your own code.

**Interpretation**:
- Uniform ranks → inference pipeline is correct
- Systematic patterns → implementation bug, wrong prior, sampler failure
- SBC failures mean the model code has a bug — fix before interpreting results

**When to run SBC**:
- Developing a new model you'll reuse
- Complex hierarchical models where bugs are easy to introduce
- Custom likelihoods or potentials
- Not necessary for routine analyses with standard model families

## Residual analysis

For regression-style models, check residuals for patterns:

```python
# Posterior predictive mean
pp_mean = idata.posterior_predictive["obs"].mean(dim=["chain", "draw"])
residuals = y_obs - pp_mean

# Residuals vs. fitted
plt.scatter(pp_mean, residuals, alpha=0.5)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Fitted values")
plt.ylabel("Residuals")
plt.title("Residuals vs. Fitted")

# Residuals vs. predictors (check for missed nonlinearity)
for j, name in enumerate(predictor_names):
    plt.figure()
    plt.scatter(X[:, j], residuals, alpha=0.5)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel(name)
    plt.ylabel("Residuals")
```

Look for: trends (missed nonlinearity), fans (heteroscedasticity), clusters (missing grouping variable).

## Decision workflow

After running diagnostics:

```
1. Convergence OK?  (reference/diagnostics.md)
   NO  → Fix sampler issues first. Do NOT proceed.
   YES ↓

2. Posterior predictive check pass?
   NO  → Model misspecification. Revise likelihood or add complexity.
   YES ↓

3. LOO-CV: any high Pareto k?
   YES → Investigate flagged observations. Consider K-fold CV.
   NO  ↓

4. Calibration OK?  (coverage + PIT)
   NO  → Model is mis-calibrated. Check priors, likelihood, missing predictors.
   YES ↓

5. Residual patterns?
   YES → Missing structure. Add predictors, nonlinearity, or hierarchical effects.
   NO  ↓

→ Model is ready for interpretation and reporting.
```
