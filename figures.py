import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)

TEAL = "#1f7a85"
DARK = "#0f4c52"  
RED = "#c0563b"

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 200,
    "font.size": 12,
    "axes.grid": True, "grid.alpha": 0.25, "axes.axisbelow": True,
})

LABELS = {
    "subgroup_median_leadtime": "Subgroup Median Lead Time",
    "carrier_median_leadtime": "Carrier Median Lead Time",
    "packing_type": "Packing Type",
    "experience_level": "Experience Level",
    "teammate_experience": "Teammate Experience",
    "nb_item": "Number of Items",
    "size_cat": "Size Category",
    "subgroup": "Subgroup",
    "carrier": "Carrier",
    "item_ratio": "Item Ratio",
    "item_86": "Item 86", "item_90": "Item 90",
    "picked_hour": "Picked Hour",
    "picked_shift": "Picked Shift",
    "nb_item_x_experience_level": "Items x Experience",
}
def pretty(col):
    return LABELS.get(col, col.replace("_", " ").title())

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

target = "total_leadtime"
drop_cols = ["total_leadtime", "time_picked", "picking_to_processing"]

y_train, y_test = train_df[target], test_df[target]
X_train = train_df.drop(columns=drop_cols, errors="ignore").copy()
X_test = test_df.drop(columns=drop_cols, errors="ignore").copy()

numeric_cols = X_train.select_dtypes(include="number").columns.tolist()
categorical_cols = X_train.select_dtypes(exclude="number").columns.tolist()

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical_cols),
])

models = {
    "Linear Regression": LinearRegression(),
    "Ridge": Ridge(alpha=10.0),
    "Random Forest": RandomForestRegressor(
        n_estimators=300, max_depth=20, min_samples_leaf=5, random_state=SEED),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8,
        colsample_bytree=0.8, reg_lambda=1.0, random_state=SEED,
        objective="reg:squarederror", tree_method="hist"),
    "HistGradientBoosting": HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.1, max_depth=None,
        min_samples_leaf=20, random_state=SEED),
}
order = list(models.keys())
cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(y_train.values, bins=60, color=TEAL, edgecolor="white", linewidth=0.4)
median = float(np.median(y_train))
ax.axvline(median, color=RED, ls="--", lw=2, label=f"Median = {median:.0f} s")
ax.set_xlabel("Total Lead Time (seconds)")
ax.set_ylabel("Frequency")
ax.legend(frameon=True)
fig.tight_layout()
fig.savefig("fig2_histogram.png")
plt.close(fig)

mae_mean, mae_std = [], []
for name in order:
    pipe = Pipeline([("preprocessor", preprocessor), ("model", models[name])])
    scores = cross_validate(pipe, X_train, y_train, cv=cv,
                            scoring="neg_mean_absolute_error")
    mae_mean.append(-scores["test_score"].mean())
    mae_std.append(scores["test_score"].std())

fig, ax = plt.subplots(figsize=(8.5, 5))
colors = [DARK if n == "HistGradientBoosting" else TEAL for n in order]
bars = ax.bar([n.replace(" ", "\n") for n in order], mae_mean,
              yerr=mae_std, capsize=5, color=colors, edgecolor="white")
ax.set_ylabel("Cross-Validated MAE (seconds)")
ax.set_ylim(0, max(mae_mean) * 1.15)
for b, v in zip(bars, mae_mean):
    ax.text(b.get_x() + b.get_width() / 2, v + max(mae_mean) * 0.01,
            f"{v:.2f}s", ha="center", va="bottom", fontsize=10)
fig.tight_layout()
fig.savefig("fig3_cv_mae.png")
plt.close(fig)

best = Pipeline([("preprocessor", preprocessor),
                 ("model", models["HistGradientBoosting"])]).fit(X_train, y_train)
pred = best.predict(X_test)

test_mae = mean_absolute_error(y_test, pred)
test_rmse = np.sqrt(mean_squared_error(y_test, pred))
test_r2 = r2_score(y_test, pred)

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(y_test, pred, s=8, alpha=0.25, color=TEAL, edgecolor="none")
lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
ax.plot(lims, lims, "--", color=RED, lw=1.8, label="Perfect prediction")
ax.set_xlabel("Actual Lead Time (seconds)")
ax.set_ylabel("Predicted Lead Time (seconds)")
ax.text(0.04, 0.96,
        f"MAE = {test_mae:.2f}s\nRMSE = {test_rmse:.2f}s\nR\u00b2 = {test_r2:.3f}",
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))
ax.legend(loc="lower right", frameon=True)
ax.set_aspect("equal", "box")
fig.tight_layout()
fig.savefig("fig4_actual_vs_pred.png")
plt.close(fig)

perm = permutation_importance(best, X_test, y_test, n_repeats=10,
                              random_state=SEED,
                              scoring="neg_root_mean_squared_error")
imp = (pd.DataFrame({"feature": X_test.columns,
                     "mean": perm.importances_mean,
                     "std": perm.importances_std})
       .sort_values("mean", ascending=False)
       .head(10).iloc[::-1])

fig, ax = plt.subplots(figsize=(8.5, 5.5))
ax.barh([pretty(f) for f in imp["feature"]], imp["mean"],
        xerr=imp["std"], capsize=3, color=TEAL, edgecolor="white")
ax.set_xlabel("Mean Increase in RMSE when Permuted (seconds)")
fig.tight_layout()
fig.savefig("fig5_importance.png")
plt.close(fig)

print("Saved fig2_histogram.png, fig3_cv_mae.png, "
      "fig4_actual_vs_pred.png, fig5_importance.png")
