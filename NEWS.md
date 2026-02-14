# xpect 1.0

* Initial CRAN submission.

# xpect 1.1
Switched XGBoost params construction to xgb.params() to align with the R migration guide and reduce deprecation risk.

Replaced alias params with “descriptive” names: eta→learning_rate, gamma→min_split_loss, alpha→reg_alpha, lambda→reg_lambda.

Added seed inside the XGBoost params (not just set.seed()) for more reproducible training/prediction.

In sampler(), preserved integer hyperparameters by generating integer sequences for integer range (critical for past, max_depth).
