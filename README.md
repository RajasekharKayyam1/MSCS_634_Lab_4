MSCS 634 – Lab 4: Regression Analysis and Regularization
Student: Rajasekhar Reddy Kayyam
Course: MSCS 634 – Advanced Big Data and Data Mining
Purpose
Explore regression modeling on the built-in Scikit-learn Diabetes dataset (442 observations, 10 numeric baseline features) to predict a quantitative measure of disease progression. Compare simple linear, multiple linear, polynomial, Ridge, and Lasso regression using MAE, MSE, RMSE, and R², supported by visualizations.
Contents
- MSCS_634_Lab_4.ipynb — Executed notebook with code, outputs, plots, and analysis.
- README.md — Setup, methods, findings, and challenges.
Requirements and execution
Install Python 3.10+ and Jupyter, then run:
python -m pip install notebook numpy pandas matplotlib scikit-learn
jupyter notebook MSCS_634_Lab_4.ipynb
Run Kernel → Restart & Run All to reproduce the outputs. No external dataset download is required.
Method
1. Inspect dataset features, summary statistics, correlations, target distribution, and missing values.
2. Use a reproducible 80/20 train/test split (random_state=42).
3. Fit simple linear regression using BMI and multiple linear regression using all ten predictors.
4. Fit polynomial regression on BMI with degrees 1, 2, 3, 5, and 10; compare training and testing R².
5. Fit Ridge and Lasso on all ten features using standardized inputs. Compare alpha values 0.01, 0.1, 1, 10, and 100.
6. Compare test MAE, MSE, RMSE, and R² and plot predicted vs actual outcomes.
Key insights
- In this fixed-split experiment, Lasso with alpha=1.0 achieved the lowest test RMSE among the listed model configurations: approximately 53.15, with test R² approximately 0.467.
- The degree-10 polynomial had the largest training-versus-testing R² gap among the polynomial degrees examined, illustrating overfitting risk.
- At alpha=1.0, Lasso set one standardized feature coefficient to zero; Ridge generally shrinks coefficients without making them exactly zero.
- Increasing regularization strength changes prediction error and coefficient sparsity; very large alpha values may underfit.
- Correlations in this observational dataset do not establish causation. These results are for coursework, not medical decision-making.
Challenges and design decisions
- Used the same train/test partition to make held-out comparisons consistent.
- Used Pipeline for standardization so scaling is learned from the training data only.
- Kept polynomial regression univariate (BMI) to avoid a combinatorial explosion in polynomial features on a small dataset.
- Included a high polynomial degree to make the bias–variance tradeoff visible.
- The alpha sweep is exploratory, not hyperparameter tuning on a separate validation set. A production study should use cross-validation within the training data and a final untouched test set.
