"""Part V: semiparametric model examples."""
import numpy as np
import statsmodels.api as sm
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import SplineTransformer
from sklearn.linear_model import LinearRegression, QuantileRegressor, LogisticRegression

rng = np.random.default_rng(42)
x = np.linspace(0, 10, 100)
y = 2 + .5*x + np.sin(x) + rng.normal(0, .3, 100)
X = sm.add_constant(x)

# 66 Cox Proportional Hazards: requires lifelines.
try:
    from lifelines import CoxPHFitter
    import pandas as pd
    df = pd.DataFrame({"time": rng.exponential(10, 100), "event": rng.binomial(1,.7,100), "x": x})
    cox = CoxPHFitter().fit(df, "time", "event")
    print("Cox hazard ratio:", np.exp(cox.params_["x"]))
except ImportError:
    print("66 Cox PH: install lifelines to run the fitted example.")

# 67 Generalized Additive Model: spline basis + linear regression.
splines = SplineTransformer(n_knots=6, degree=3).fit_transform(x.reshape(-1,1))
gam = LinearRegression().fit(np.column_stack([x, splines]), y)
print("67 GAM prediction:", gam.predict(np.column_stack([x, splines]))[:3])

# 68 Partially Linear Model: parametric x + flexible spline z.
z = rng.normal(size=100)
basis = SplineTransformer(n_knots=5).fit_transform(z.reshape(-1,1))
plm = LinearRegression().fit(np.column_stack([x, basis]), y)
print("68 Partially linear coefficients:", plm.coef_[0])

# 69 Additive Hazard Model: illustrative additive-risk regression.
event_risk = .1 + .03*x + rng.normal(0, .01, 100)
additive = LinearRegression().fit(x.reshape(-1,1), event_risk)
print("69 Additive hazard slope:", additive.coef_[0])

# 70 Penalized spline.
pspline = UnivariateSpline(x, y, s=5)
print("70 Penalized spline prediction:", pspline(5))

# 71 Spline Regression.
spline_reg = LinearRegression().fit(splines, y)
print("71 Spline regression prediction:", spline_reg.predict(splines[:1])[0])

# 72 Semiparametric quantile regression.
qr = QuantileRegressor(quantile=.5, alpha=.1).fit(x.reshape(-1,1), y)
print("72 Median regression slope:", qr.coef_[0])

# 73 Varying coefficient model: coefficient changes with z through interaction.
interaction = np.column_stack([x, x*z, z])
vc = LinearRegression().fit(interaction, y)
print("73 Varying coefficient parameters:", vc.coef_)

# 74 Frailty Model: random-effect concept represented with group intercepts.
groups = np.repeat(np.arange(10), 10)
group_effect = rng.normal(0, .5, 10)[groups]
frailty_y = y + group_effect
frailty = sm.OLS(frailty_y, sm.add_constant(x)).fit()
print("74 Frailty-style fixed slope:", frailty.params[1])

# 75 Proportional Odds: ordered-logit approximation.
try:
    from statsmodels.miscmodels.ordinal_model import OrderedModel
    ordered_y = np.digitize(y, [-.5, .5, 1.5])
    po = OrderedModel(ordered_y, x.reshape(-1,1), distr="logit").fit(method="bfgs", disp=False)
    print("75 Proportional odds coefficient:", po.params[0])
except Exception as exc:
    print("75 Proportional odds example:", exc)
