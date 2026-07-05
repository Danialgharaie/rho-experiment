# rho-experiment

Code for https://danialgharaie.github.io/notebook/2026/the-price-of-a-wrong-prior/

Extends the cost-of-precision model from
[lambda-experiment](https://github.com/Danialgharaie/lambda-experiment) /
"The Price of Precision": the price-aware split's cheap/precise allocation
is chosen using an *assumed* correlation `rho_hat`, then evaluated on data
generated with a *true* `rho`. Everything downstream of the allocation
(interpreting the resulting reads into a posterior mean) uses the true rho
-- the mistake being isolated is mispricing precision, not also misreading
your own instruments afterwards.

- `sim.py` -- shared primitives (world generation, posterior variance,
  the allocation optimizer).
- `misspec_exp.py` -- the (rho_hat x rho_true) grid experiment and both
  figures.
- `paired.py` -- paired-replicate comparison isolating the cost of
  confidently assuming `rho_hat=0`.

```bash
uv sync
uv run python misspec_exp.py
uv run python paired.py
```
