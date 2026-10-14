# Model notebooks

All four notebooks reuse the same NIT Patna/SIDSS + mandatory NASA POWER project data, target, feature order, and seeded split. They are saved with actual executed outputs.

| Notebook | Role |
|---|---|
| [01_linear_regression.ipynb](01_linear_regression.ipynb) | Linear Regression |
| [02_decision_tree_regressor.ipynb](02_decision_tree_regressor.ipynb) | Decision Tree Regressor |
| [03_random_forest_regressor.ipynb](03_random_forest_regressor.ipynb) | Random Forest Regressor |
| [04_gradient_boosting_regressor.ipynb](04_gradient_boosting_regressor.ipynb) | XGBoost Regressor |

Linear Regression is the baseline; Decision Tree is a single-tree nonlinear model; Random Forest is a bagging ensemble; XGBoost is the boosted-tree ensemble.

From the repository root, install `pip install -r requirements-notebooks.txt`, then open a notebook in Jupyter. Run cells top to bottom. Existing processed data and saved tuning JSON must be present; if missing, run `python main.py` once. No notebook makes a NASA API call or overwrites trained model artifacts. The notebooks expose saved-run metric differences and scientific limitations rather than hiding them.

To execute all four and refresh static final summaries, run `python scripts/execute_notebooks.py`.

The [combined output report](../reports/model_outputs.html) contains all four models’ saved tables, metrics, sample predictions and plots. Download and open the HTML in a browser. Regenerate it with `python scripts/export_model_outputs.py` after executing the notebooks.
