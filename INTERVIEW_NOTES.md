# scikit-learn interview notes

Notes for machine-learning interviews that use scikit-learn 1.9. Answers are written the way you would say them out loud. Add your own examples under each question as you practice.

## What scikit-learn is

**What is scikit-learn?**  
A Python library for classical machine learning. You import it as `sklearn`. It covers supervised models, clustering, dimensionality reduction, preprocessing, model selection, and metrics. It expects tabular data: `X` with shape `(n_samples, n_features)`, and `y` with shape `(n_samples,)`.

**What does it leave to other libraries?**  
Deep learning, GPU training, and large-scale distributed training. scikit-learn is the usual choice for structured data and models that fit in memory.

**What is an estimator?**  
Any object that learns from data through `fit`. After fitting, it can `predict` or `transform`.

## The estimator API

**What methods should you know?**

| Method | Role |
| --- | --- |
| `fit(X, y)` | Learn from data. Unsupervised estimators take only `X`. |
| `predict(X)` | Class labels or regression values. |
| `predict_proba(X)` | Class probabilities, when the model supports them. |
| `decision_function(X)` | A signed score. Higher means more confidence in the positive class. |
| `transform(X)` | A new representation of `X`. Used by scalers, encoders, and PCA. |
| `fit_transform(X)` | Fit on `X`, then transform it. |
| `score(X, y)` | A default metric. Accuracy for classifiers, R² for regressors. |

**Where do parameters live?**  
You pass hyperparameters to the constructor: `C`, `max_depth`, `n_estimators`. Values learned during `fit` are stored on attributes that end with `_`: `coef_`, `intercept_`, `feature_importances_`, `classes_`.

**Why does every model share this API?**  
So models, preprocessors, cross-validation, and pipelines can be swapped without rewriting the training loop. `get_params` and `set_params` are what `GridSearchCV` and `Pipeline` use to clone an estimator and set nested parameters.

## Data and leakage

**What is the first split you make?**  
Hold out a test set with `train_test_split` before any fitting. Fit every scaler, encoder, imputer, and model on the training split only. Transform the test split with those already fitted objects.

**What is data leakage?**  
Information from the test set, or from the future, reaching the model during training. The score then looks better than production performance. Typical causes:

- Fitting a scaler on the full dataset, then splitting.
- Choosing features by looking at the whole dataset, including the test rows.
- Tuning hyperparameters on the test set.
- Filling missing values with a statistic computed from all rows.
- Time-series data split at random, so later rows train a model that is tested on earlier rows.

**How do you split time-series data?**  
Cut by time. Train on the past, test on a later window. Use `TimeSeriesSplit` for cross-validation. A random split lets the model see the future.

**Why pass `stratify=y`?**  
So each split keeps the same class proportions as the full dataset. This matters when one class is rare. A random split can put almost all of the rare class into one side.

**Why set `random_state`?**  
So the split and any stochastic model can be reproduced. Interviews often expect you to mention it when you talk about comparing two models.

## Preprocessing

**Which models need scaled features?**  
Models that use distances or a linear penalty on coefficients: linear and logistic regression, SVM, k-nearest neighbors, k-means, and PCA. A feature on a scale of thousands otherwise dominates one on a scale of 0 to 1.

Tree models split on one feature at a time, so `StandardScaler` does not change their splits. Random forests and gradient boosting do not need scaled numeric inputs.

**StandardScaler vs MinMaxScaler?**  
`StandardScaler` subtracts the mean and divides by the standard deviation. Use it as the default for linear models and SVMs. `MinMaxScaler` maps each feature to a fixed range, usually 0 to 1. Use it when you need bounded inputs. Both learn their statistics in `fit`, so fit them on the training data only.

**How do you encode categories?**  
`OneHotEncoder` makes one binary column per category. Use it for nominal categories, where no order exists. Set `handle_unknown="ignore"` if the test set may contain a new category. `OrdinalEncoder` maps categories to integers. Use it when the order is real, or when a tree model can split on the codes. An arbitrary integer coding fed to a linear model invents a fake order.

**What is the dummy-variable issue?**  
One-hot columns sum to 1, so they are perfectly collinear with an intercept. For a linear model, drop one category (`drop="first"`) or remove the intercept. Tree models can keep every column.

**How do you impute missing values?**  
`SimpleImputer` fills with the mean, median, or most frequent value learned from the training data. Put the imputer inside a pipeline so each cross-validation fold computes its own fill value. The median is the safer default for numeric features that contain outliers.

## Supervised models

**How do you choose a model in an interview?**  
Start from the data and the constraint, then name the model.

| Situation | A solid default |
| --- | --- |
| Numeric features, roughly linear relationship, need coefficients | Linear or logistic regression |
| Many correlated features, still want a linear model | Ridge (L2) |
| Want a sparse linear model, automatic feature dropping | Lasso (L1) |
| Mixed feature types, nonlinear, little tuning time | `HistGradientBoostingClassifier` or a random forest |
| Small dataset, clear margin between classes, features scaled | SVM with an RBF kernel |
| Need a simple baseline or local decisions | k-nearest neighbors |
| Need calibrated probabilities | Logistic regression, or any classifier wrapped in `CalibratedClassifierCV` |

Always mention a baseline: majority-class accuracy for classification, or predicting the mean for regression.

### Linear and logistic regression

**What does linear regression learn?**  
Weights `coef_` that minimize squared error between `X @ coef_ + intercept_` and `y`. `LinearRegression.score` returns R².

**What do Ridge and Lasso add?**  
Both add a penalty so large weights are discouraged.

- Ridge (`Ridge`, and logistic regression with `l1_ratio=0`) adds the sum of squared weights. Weights shrink toward zero and stay nonzero. It handles correlated features by sharing weight among them.
- Lasso (`Lasso`, and logistic regression with `l1_ratio=1`) adds the sum of absolute weights. Some weights become exactly zero, so it also selects features.
- Elastic net mixes the two. In logistic regression that mix is `l1_ratio` between 0 and 1.

In scikit-learn 1.9, `LogisticRegression` no longer takes `penalty` as the way to choose this. `l1_ratio=0` is L2, `l1_ratio=1` is L1, and `C` is the inverse of regularization strength. Smaller `C` means a stronger penalty. `C=np.inf` turns the penalty off.

**What does logistic regression predict?**  
It models the log-odds of a class as a linear function of the features, then converts that to a probability with the logistic function. `predict` applies a threshold, default 0.5, to `predict_proba`. For more than two classes the default is multinomial: one linear function per class, then a softmax.

**Can you interpret `coef_`?**  
In linear regression, a coefficient is the change in the prediction for a one-unit change in that feature, with the others held fixed. That reading depends on the feature's scale, so interpret coefficients after you state how the column was scaled. In logistic regression the coefficient is a change in log-odds, not in probability. Correlated features make individual coefficients unstable even when predictions stay good.

### Trees and ensembles

**How does a decision tree predict?**  
It asks a sequence of threshold questions on single features and sends each sample to a leaf. The leaf predicts the majority class, or the mean of the training targets in that leaf. An unpruned tree memorizes the training set. Limit it with `max_depth`, `min_samples_leaf`, or `ccp_alpha`.

**Bagging vs boosting?**  
Bagging trains many models on bootstrap samples and averages them. The trees are independent and can train in parallel. Random forests are bagged trees that also pick a random subset of features at each split (`max_features="sqrt"` for classifiers). Averaging reduces variance.

Boosting trains trees one after another. Each new tree focuses on the residual error of the ensemble so far. Gradient boosting shrinks each tree by `learning_rate` and adds it to the running prediction. Use enough trees (`n_estimators` or `max_iter`) together with a small learning rate. Boosting reduces bias and can overfit if you add too many trees or make each tree too deep.

**Random forest vs gradient boosting, in practice?**

- A random forest is a strong default, is hard to break, and trains trees in parallel. Important knobs are `n_estimators`, `max_depth`, and `min_samples_leaf`.
- Gradient boosting often reaches a higher score after tuning. `HistGradientBoostingClassifier` is the faster implementation in current scikit-learn. It bins numeric features and can take categorical features when the column dtype is categorical. Important knobs are `learning_rate`, `max_iter`, and `max_depth` or `max_leaf_nodes`.

**What is out-of-bag score?**  
Each random-forest tree leaves out about a third of the rows. Predicting those left-out rows gives an internal validation score without a separate holdout. Set `oob_score=True`. It is a useful check. A final reported number still comes from a test set the forest never saw.

**Is `feature_importances_` reliable?**  
It is a useful ranking, and it is biased toward features with more possible split points, including high-cardinality numeric columns and categories that were not grouped. For a claim you would defend, use permutation importance: shuffle one column and measure how much the score drops. That works on any fitted model and respects feature interactions better than a single coefficient.

### SVM and neighbors

**What is a support vector machine doing?**  
It looks for the boundary with the widest margin between classes. Points on the margin are the support vectors. `C` trades a wider margin for fewer margin violations. A small `C` allows a smoother boundary. The RBF kernel measures similarity by distance, so scale the features first. `gamma` controls how far a single training point influences the boundary. A large `gamma` can overfit.

**How do you get probabilities from an SVM in 1.9?**  
`SVC(probability=True)` still works and is deprecated. It will be removed in 1.11. The current approach is `CalibratedClassifierCV(SVC(), ensemble=False)`, which fits the SVM and then calibrates scores on held-out predictions. Say this if the interviewer asks for probabilities rather than labels.

**When is k-nearest neighbors a bad fit?**  
Prediction scans neighbors, so it is slow at query time on large data. Distance is dominated by irrelevant features and by unscaled columns. It also stores the training set. It is a good baseline and a good choice when the decision really is local.

## Unsupervised learning

**What is k-means assuming?**  
You choose `n_clusters`. It alternates between assigning each point to the nearest centroid and moving each centroid to the mean of its points. Clusters are treated as spherical and of similar size. Scale the features first, because the algorithm uses Euclidean distance. The result depends on the initial centroids, so run it with several initializations (`n_init`) and keep the lowest inertia. Inertia always falls as `k` grows, so the elbow is only a heuristic. Check cluster stability and whether the groups are useful.

**What is PCA for?**  
It finds orthogonal directions of highest variance and projects the data onto the first `k` of them. Use it to compress features, to visualize, or to remove linear correlation before a distance-based model. Scale features first, or a large-range column dominates the variance. PCA does not look at `y`. A direction with low variance can still be the one that separates classes. For supervised reduction, mention that you would select features with the target, rather than keeping the top principal components only because they explain variance.

## Metrics

**Which classification metric do you report?**  
Match it to the cost of errors.

- Accuracy is the fraction of correct labels. It hides failure on a rare class. If 99% of rows are negative, a model that always predicts negative scores 0.99.
- Precision is the fraction of positive predictions that are truly positive. Use it when a false positive is expensive.
- Recall is the fraction of actual positives that you caught. Use it when a false negative is expensive.
- F1 is the harmonic mean of precision and recall. Use it when you care about both and the classes are uneven.
- ROC-AUC measures how well the score ranks positives above negatives, across thresholds. It needs scores or probabilities, not hard labels. It can look strong when the negative class is huge, because false positives are diluted.
- Precision-recall AUC is the ranking metric to prefer when positives are rare.
- A confusion matrix is what you show when someone asks which classes are being mixed up.
- Log loss punishes confident wrong probabilities. Use it when the probabilities themselves matter.

**What threshold do you use?**  
`predict` uses 0.5. If false negatives and false positives have different costs, sweep the threshold on a validation set using `predict_proba`, pick the threshold there, and apply that same threshold to the test set once.

**Which regression metric do you report?**

- MAE is the average absolute error, in the units of `y`. It is easy to explain and resists outliers.
- MSE and RMSE punish large errors more. RMSE is in the units of `y`.
- R² is the fraction of variance explained relative to predicting the mean. R² of 0 matches the mean baseline. Negative R² is worse than predicting the mean.

## Validation and tuning

**Why cross-validate?**  
One split can be lucky. `cross_val_score` refits the model on several folds and reports the mean and spread. Use stratified folds for classification. The test set stays untouched until the model and its hyperparameters are frozen.

**Where does the test set enter?**  
Once. After you have chosen the model, the preprocessing, and the hyperparameters using training data and cross-validation. If you look at the test score and then change the model, that set has become another validation set.

**GridSearchCV vs RandomizedSearchCV?**  
`GridSearchCV` tries every combination you list. Use it when the grid is small. `RandomizedSearchCV` samples combinations. Use it when you have many hyperparameters, and spend the budget on the ones that matter most: `C`, `max_depth`, `learning_rate`, `n_estimators`. Both refit the best parameters on the full training set by default.

**What must sit inside the search?**  
The entire pipeline, including scaling and imputation. If you scale first and then cross-validate the model only, every fold's scaler has seen the validation rows.

**How do you tell overfitting from underfitting?**  
Compare training score and validation score.

- Both scores are poor: the model is too simple, the features are weak, or the target is noisy. Add features, use a more flexible model, or reduce regularization.
- Training score is high and validation score is much lower: the model memorized the training set. Constrain it with `max_depth`, `min_samples_leaf`, stronger `C` regularization, fewer boosting iterations, or more data.

A learning curve plots score against training-set size. A gap between train and validation that shrinks as data grows points to collecting more data. A gap that stays open points to simplifying the model.

## Pipelines

**Why use a pipeline?**  
`Pipeline` chains preprocessing and a final estimator. `fit` runs each step in order. `predict` runs every transformer, then the model. Cross-validation clones the whole pipeline per fold, so preprocessing cannot see the validation rows. It is also the clean object to save and deploy, because the scaler travels with the model.

A column transformer applies different steps to numeric and categorical columns, then concatenates the results. The usual pattern is impute and scale the numeric columns, and impute and one-hot encode the categorical columns.

**How do you name parameters in a grid?**  
Use the step name, two underscores, then the parameter: `clf__C`, `prep__num__scaler__with_mean`. That double underscore is how `set_params` addresses nested estimators.

## Imbalance, calibration, and saving

**The positive class is 1% of the data. What do you change?**  
Keep a stratified split. Stop using accuracy as the headline metric. Look at recall, precision, and the precision-recall curve. Practical levers inside scikit-learn are `class_weight="balanced"`, which up-weights the rare class in the loss, and a threshold chosen for the recall you need. Getting more positive examples, or cleaner features, usually beats a clever sampler.

**What is probability calibration?**  
A model can rank well and still produce probabilities that do not match observed frequencies. A random forest's probabilities are often pushed away from 0 and 1. An SVM score is not a probability at all. `CalibratedClassifierCV` fits a mapping from scores to probabilities with Platt scaling or isotonic regression, using out-of-fold predictions so the mapping is not fit on the same rows as the model.

**How do you save a model?**  
`skops` or the versioned pickle path documented by scikit-learn. Loading a pickle is equivalent to running code, so load only files you trust. Record the scikit-learn version with the file. A pipeline saved after `fit` includes the fitted scaler and encoder.

## Questions interviewers use as follow-ups

**Why can cross-validation still be biased?**  
If you used the validation scores to pick features or a model family, those scores are optimistic. Nested cross-validation puts the search in the inner loop and the evaluation in the outer loop. Mention it when the interviewer asks for an unbiased estimate. For a single final number, a held-out test set that was locked before selection is the simpler answer.

**Two models score 0.91 and 0.92. Which do you ship?**  
Ask for the metric, the variance across folds, the latency, and the cost of a mistake. A 0.01 gap inside the fold-to-fold noise is not a reason to take the more complicated model. If the scores are stable and the metric matches the business cost, take the higher one and confirm once on the test set.

**A categorical column has a new level in production. What happens?**  
With the default `OneHotEncoder(handle_unknown="error")`, `transform` fails. Set `handle_unknown="ignore"` so the new level becomes an all-zero row, and monitor how often that happens. An all-zero row is a silent fallback, so it deserves an alert.

**Why might a tree beat a tuned logistic regression on the same table?**  
The target depends on thresholds and interactions, which a tree represents directly. A linear model needs you to build those interactions as columns. If a linear model wins, the relationship is close to linear and the coefficients are easier to explain.

**What do you check before you trust a high score?**  
That the split was done before fitting. That the metric matches the decision. That a simple baseline was beaten by enough to matter. That the training and validation scores are both healthy. That duplicate rows or a leaked identifier column are not carrying the score. A column that is a copy of the target, or a unique id, can make any model look perfect.
