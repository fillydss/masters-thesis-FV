import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from xgboost import XGBRegressor

SEED = 42
np.random.seed(SEED)

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

target = "total_leadtime"
y_train = train_df[target]
y_test = test_df[target]

drop_cols = ["total_leadtime", "time_picked", "picking_to_processing"]
X_train = train_df.drop(columns=drop_cols, errors="ignore").copy()
X_test = test_df.drop(columns=drop_cols, errors="ignore").copy()

numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                      ("scaler", StandardScaler())]), numeric_cols),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))]), categorical_cols),
])

search_space = {
    "linear_regression": (LinearRegression(), {}),
    "ridge": (Ridge(random_state=SEED),
              {"model__alpha": [0.1, 1.0, 10.0]}),
    "random_forest": (RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1),
              {"model__max_depth": [None, 20],
               "model__min_samples_leaf": [2, 5]}),
    "hist_gradient_boosting": (HistGradientBoostingRegressor(max_iter=300, random_state=SEED),
              {"model__learning_rate": [0.05, 0.1],
               "model__max_depth": [None, 8]}),
    "xgboost": (XGBRegressor(n_estimators=500, subsample=0.8, colsample_bytree=0.8,
                             reg_lambda=1.0, random_state=SEED, n_jobs=1,
                             objective="reg:squarederror", tree_method="hist"),
              {"model__learning_rate": [0.05, 0.1],
               "model__max_depth": [4, 6]}),
}

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"mae": "neg_mean_absolute_error",
           "rmse": "neg_root_mean_squared_error",
           "r2": "r2"}

fitted, rows = {}, []
for name, (estimator, grid) in search_space.items():
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    gs = GridSearchCV(pipe, grid, scoring=scoring, refit="mae",
                      cv=cv, n_jobs=-1, return_train_score=False)
    gs.fit(X_train, y_train)
    bi = gs.best_index_
    fitted[name] = gs
    rows.append({
        "model": name,
        "best_params": {k.replace("model__", ""): v for k, v in gs.best_params_.items()},
        "cv_mae": -gs.cv_results_["mean_test_mae"][bi],
        "cv_mae_std": gs.cv_results_["std_test_mae"][bi],
        "cv_rmse": -gs.cv_results_["mean_test_rmse"][bi],
        "cv_r2": gs.cv_results_["mean_test_r2"][bi],
    })
    print(f"[done] {name:22s} best MAE={-gs.cv_results_['mean_test_mae'][bi]:.4f}  "
          f"params={rows[-1]['best_params']}")

cv_df = pd.DataFrame(rows).sort_values("cv_mae").reset_index(drop=True)

print("\n=== CHOSEN HYPERPARAMETERS ===")
for r in rows:
    print(f"  {r['model']:22s} {r['best_params']}")

print("\n=== CROSS-VALIDATION RESULTS ===")
print(cv_df[["model", "cv_mae", "cv_rmse", "cv_r2"]].round(4).to_string(index=False))


best_name = cv_df.loc[0, "model"]
best_pipeline = fitted[best_name].best_estimator_
y_pred = best_pipeline.predict(X_test)

print(f"\n=== BEST MODEL: {best_name} ===")
print("Test MAE :", round(mean_absolute_error(y_test, y_pred), 4))
print("Test RMSE:", round(np.sqrt(mean_squared_error(y_test, y_pred)), 4))
print("Test R2  :", round(r2_score(y_test, y_pred), 4))

perm = permutation_importance(best_pipeline, X_test, y_test, n_repeats=10,
                              random_state=SEED, scoring="neg_root_mean_squared_error", n_jobs=-1)
imp = pd.DataFrame({"feature": X_test.columns,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std}).sort_values("importance_mean", ascending=False)
print("\n=== PERMUTATION IMPORTANCE (top 20) ===")
print(imp.head(20).round(3).to_string(index=False))
