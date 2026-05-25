import numpy as np
import pandas as pd

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

train_df = pd.read_csv("train_data.csv")
test_df = pd.read_csv("test_data.csv")

target = "total_leadtime"
y_train = train_df[target]
y_test = test_df[target]

drop_cols = [
    "total_leadtime",
    "time_picked",
     "picking_to_processing"
]

X_train = train_df.drop(columns=drop_cols, errors="ignore").copy()
X_test = test_df.drop(columns=drop_cols, errors="ignore").copy()

numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

numeric_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_preprocessor, numeric_cols),
    ("cat", categorical_preprocessor, categorical_cols)
])

models = {
    "linear_regression": LinearRegression(),
    "ridge": Ridge(alpha=1.0, random_state=SEED),
    "random_forest": RandomForestRegressor(
        n_estimators=300,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=SEED,
        n_jobs=1
    ),
    "hist_gradient_boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=20,
        random_state=SEED
    ),
    "xgboost": XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=1,
        objective="reg:squarederror"
    )
}

cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

scoring = {
    "mae": "neg_mean_absolute_error",
    "rmse": "neg_root_mean_squared_error",
    "r2": "r2"
}

cv_results = []
for name, model in models.items():
    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
        return_train_score=False
    )

    cv_results.append({
        "model": name,
        "cv_mae_mean": -scores["test_mae"].mean(),
        "cv_mae_std": scores["test_mae"].std(),
        "cv_rmse_mean": -scores["test_rmse"].mean(),
        "cv_rmse_std": scores["test_rmse"].std(),
        "cv_r2_mean": scores["test_r2"].mean(),
        "cv_r2_std": scores["test_r2"].std()
    })

cv_results_df = pd.DataFrame(cv_results).sort_values(by="cv_rmse_mean").reset_index(drop=True)
print("\nCross-validation results:")
print(cv_results_df.to_string(index=False))

best_model_name = cv_results_df.loc[0, "model"]
best_model = models[best_model_name]

best_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", best_model)
])

best_pipeline.fit(X_train, y_train)
y_pred = best_pipeline.predict(X_test)

test_mae = mean_absolute_error(y_test, y_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
test_r2 = r2_score(y_test, y_pred)

print("\nBest model:", best_model_name)
print("Test MAE:", round(test_mae, 4))
print("Test RMSE:", round(test_rmse, 4))
print("Test R2:", round(test_r2, 4))

predictions_df = pd.DataFrame({
    "actual_total_leadtime": y_test.values,
    "predicted_total_leadtime": y_pred
})

print("\nPrediction sample:")
print(predictions_df.head(10).to_string(index=False))

perm = permutation_importance(
    best_pipeline,
    X_test,
    y_test,
    n_repeats=10,
    random_state=SEED,
    scoring="neg_root_mean_squared_error",
    n_jobs=1
)

importance_df = pd.DataFrame({
    "feature": X_test.columns,
    "importance_mean": perm.importances_mean,
    "importance_std": perm.importances_std
}).sort_values(by="importance_mean", ascending=False)

print("\nTop 20 features by permutation importance:")
print(importance_df.head(20).to_string(index=False))