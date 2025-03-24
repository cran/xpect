#' xpect
#'
#' This function implements probabilistic time series forecasting by combining gradient-boosted regression (gbtree booster)
#' with conformal inference approach. It produces predictive distributions capturing uncertainty and optimizes
#' hyper parameters through Bayesian, coarse-to-fine, or random search methods. The approach leverages historical
#' observations from predictor series to estimate the future values of a specified target series. Users can customize
#' the forecasting model extensively by setting parameters for model complexity, regularization, and conformal
#' calibration.
#'
#' @param predictors A data frame containing multiple time series predictors and the target series to forecast.
#' @param target Character string specifying the name of the target series to forecast within the predictors dataset.
#' @param future Integer specifying the number of future time steps to forecast.
#' @param past Integer or numeric vector specifying past observations used as input features. Single value sets fixed value (default: 1). NULL sets standard range (1L-30L), while two values define custom range.
#' @param coverage Numeric or numeric vector for fraction of total variance preserved during SVD. Single value sets fixed value (default: 0.5). NULL sets standard range (0.05-0.95), while two values define custom range.
#' @param max_depth Integer or numeric vector for max depth of XGBoost trees. Single value sets fixed value (default: 3). NULL sets standard range (3L-10L), while two values define custom range.
#' @param eta Numeric or numeric vector for learning rate in XGBoost. Single value sets fixed value (default: 0.1). NULL sets standard range (0.01-0.3), while two values define custom range.
#' @param gamma Numeric or numeric vector for minimum loss reduction to split a leaf node. Single value sets fixed value (default: 0). NULL sets standard range (0-5), while two values define custom range.
#' @param alpha Numeric or numeric vector for L1 regularization strength. Single value sets fixed value (default: 0). NULL sets standard range (0-1), while two values define custom range.
#' @param lambda Numeric or numeric vector for L2 regularization strength. Single value sets fixed value (default: 1). NULL sets standard range (0-1), while two values define custom range.
#' @param subsample Numeric or numeric vector (0-1) for instance subsampling ratio per tree.  Single value sets fixed value (default: 0.8). NULL sets standard range (0-1), while two values define custom range.
#' @param colsample_bytree Numeric or numeric vector (0-1) for column subsampling ratio per tree.  Single value sets fixed value (default: 0.8). NULL sets standard range (0-1), while two values define custom range.
#' @param search Character string specifying the hyper parameter search method to employ. Options include: "none" (default), "random_search", "bayesian", "coarse_to_fine".
#' @param calib_rate Numeric fraction (default: 0.5) of observations allocated for conformal calibration, influencing the uncertainty estimation.
#' @param n_sim Integer (default: 1000) determining the number of simulated calibration error samples used during conformal inference.
#' @param nrounds Integer (default: 200) specifying the maximum number of boosting iterations allowed during model training.
#' @param n_samples Integer specifying the number of parameter configurations evaluated during random search or initial Bayesian sampling.
#' @param n_exploration Integer specifying the number of exploratory evaluations during Bayesian optimization to balance exploration-exploitation.
#' @param n_phases Integer specifying how many iterative refinement phases are performed in coarse-to-fine optimization.
#' @param top_k Integer (default: 3) indicating how many top-performing parameter configurations are retained in each coarse-to-fine optimization iteration.
#' @param seed Integer setting the random seed for reproducibility.
#'
#' @author Giancarlo Vercellino \email{giancarlo.vercellino@gmail.com}
#'
#' @return A list containing:
#' \describe{
#'   \item{history}{A data frame logging each evaluated hyperparameter configuration and its associated cross-entropy performance against the selected benchmark.}
#'   \item{best_model}{The optimal forecasting model, including probability density functions (pdf), cumulative distribution functions (cdf), inverse cumulative distribution functions (icdf), and random sampling functions (sampler) for each point in the forecasted horizon.}
#'   \item{best_params}{A named vector detailing the selected hyper parameters of the best-performing forecasting model.}
#'   \item{plot}{A visualization displaying the optimal forecasts alongside confidence bands derived from conformal intervals, facilitating intuitive uncertainty interpretation.}
#'   \item{time_log}{Duration tracking the computational time required for the complete optimization and model-building process.}
#' }
#'
#'
#'
#' @examples
#' \donttest{
#'
#'   dummy_data <- data.frame(target_series = cumsum(rnorm(100)), predictor1 = cumsum(rnorm(100)))
#'
#'   result <- xpect(predictors = dummy_data,
#'                           target = "target_series",
#'                           future = 3,
#'                           past = c(5L, 20L),#CUSTOM RANGE
#'                           coverage = 0.9,
#'                           max_depth = c(3L, 8L),#CUSTOM RANGE
#'                           eta = c(0.01, 0.05),
#'                           gamma = NULL,#STANDARD RANGE
#'                           alpha = NULL,#STANDARD RANGE
#'                           lambda = NULL,#STANDARD RANGE
#'                           subsample = 0.8,
#'                           colsample_bytree = 0.8,
#'                           search = "random_search",
#'                           n_samples = 3,
#'                           seed = 123)
#' }
#'
#' @import purrr
#' @import xgboost
#' @import furrr
#' @import future
#' @import rBayesianOptimization
#' @import edfun
#' @importFrom utils tail head
#' @importFrom cubature adaptIntegrate
#' @importFrom lubridate seconds_to_period
#' @import glogis
#' @import normalp
#' @import evd
#' @import ald
#' @import gld
#' @import GeneralizedHyperbolic
#' @import ggplot2
#' @importFrom scales number
#' @importFrom stats coef median predict
#'
#' @export
xpect <- function(predictors, target, future,
                  past = 1L, coverage = 0.5, max_depth = 3L,
                  eta = 0.1, gamma = 0, alpha = 0, lambda = 1,
                  subsample = 0.8, colsample_bytree = 0.8,
                  search = "none",
                  calib_rate = 0.5, n_sim = 1000, nrounds = 200,
                  n_samples = 10, n_exploration = 10, n_phases = 3, top_k = 3, seed = 42)
{
  set.seed(seed)

  start_time <- Sys.time()

  search <- match.arg(search, choices = c("none", "bayesian", "coarse_to_fine", "random_search"))

  results <- switch(search, "bayesian" = bayesian_search(predictors, target, future, past, coverage,
                                                         max_depth, eta, gamma, alpha, lambda, subsample, colsample_bytree,
                                                         n_samples, n_exploration, calib_rate, n_sim, nrounds, seed),
                      "coarse_to_fine" = coarse_to_fine_search(predictors, target, future, past, coverage,
                                                         max_depth, eta, gamma, alpha, lambda, subsample, colsample_bytree,
                                                         n_samples, n_phases, top_k, calib_rate, n_sim, nrounds, seed),
                       "random_search" = random_search(predictors, target, future, past, coverage,
                                                         max_depth, eta, gamma, alpha, lambda, subsample, colsample_bytree,
                                                         n_samples, calib_rate, n_sim, nrounds, seed),
                                "none" = engine(predictors, target, future, past, coverage, calib_rate,
                                                         n_sim, nrounds, max_depth, eta, gamma, alpha, lambda, subsample,
                                                         colsample_bytree, seed))

  if(search %in% c("bayesian", "coarse_to_fine", "random_search"))
  {
    history = results$history
    best_model = results$best_model
    best_params = results$best_params
  }

  if(search == "none")
  {
    history = NA
    best_model = results
    best_params = NA
  }

  plot <- plot_graph(predictors[[target]], best_model, alpha = 0.05, dates = NULL)

  end_time <- Sys.time()
  time_log <- seconds_to_period(round(difftime(end_time, start_time, units = "secs"), 0))

  outcome <- list(history = history, best_model = best_model,
                  best_params = best_params, plot = plot, time_log = time_log)

  return(outcome)
}




###############
engine <- function(predictors, target, future, past,
                   coverage, calib_rate, n_sim, nrounds, max_depth, eta,
                   gamma, alpha, lambda, subsample, colsample_bytree, seed)
{
  set.seed(seed)

  ts <- predictors[[target]]
  scaled_ts <- dts(ts, 1)
  labels <- paste0("t", past + 1:future)

  feat_model <- transformation(predictors, future, past, coverage)
  predictors <- feat_model$set
  new <- feat_model$new

  target <- as.data.frame(smart_reframer(scaled_ts, past + future, 1))[, past + 1:future, drop = FALSE]

  preds <- as.data.frame(list_flatten(map(1:future,
                                          ~ conformal_predictor(cbind(target[,.x, drop = FALSE], predictors),
                                                                target = labels[.x], newdata =  new, calib_rate,
                                                                n_sim, nrounds, max_depth, eta, gamma, alpha,
                                                                lambda, subsample, colsample_bytree, seed))))

  proj_space <- as.data.frame(t(apply(preds, 1, function(x) tail(ts, 1) * cumprod(1+x))))
  if(all(ts > 0)){proj_space <- proj_space[apply(proj_space, 1, function(x) all(x > 0)), , drop = FALSE]}
  if(all(ts <= 0)){proj_space <- proj_space[apply(proj_space, 1, function(x) all(x <= 0)), , drop = FALSE]}

  pred_funs <- map(proj_space, ~ suppressWarnings(mixture(.x)))
  names(pred_funs) <- paste0("t", 1:future)

  return(pred_funs)
}

####
transformation <- function(predictors, future, past, coverage)
{

  multi_ts <- map(predictors, ~ dts(.x, 1))
  multi_set <- map(multi_ts, ~ smart_reframer(.x, past + future, 1))
  min_set <- min(map_dbl(multi_set, ~ nrow(.x)))
  multiplex <- as.data.frame(Reduce(cbind, map(multi_set, ~ tail(.x, min_set)[, 1:past, drop = F])))
  rownames(multiplex) <- NULL
  colnames(multiplex) <- paste0("feat_", 1:ncol(multiplex))
  new <- as.data.frame(Reduce(cbind, map(multi_ts, ~ matrix(tail(.x, past), nrow = 1))))
  colnames(new) <- colnames(multiplex)

  redux <- svd_coverage(rbind(multiplex, new), coverage)
  new <- tail(redux, 1)
  set <- head(redux, -1)

  out <- list(set = set, new = new)
  return(out)
}




#####
conformal_predictor <- function(data, target, newdata, calib_rate, n_sim,
                                nrounds, max_depth, eta, gamma, alpha,
                                lambda, subsample, colsample_bytree, seed)
{

  set.seed(seed)

  # Split data into training and calibration sets
  n <- nrow(data)
  calib_size <- floor(calib_rate * n)
  calib_indices <- sample(1:n, size = calib_size, replace = FALSE)
  train_indices <- setdiff(1:n, calib_indices)

  train_data <- data[train_indices, , drop = FALSE]
  calib_data <- data[calib_indices, , drop = FALSE]

  # Train the model on the training set
  train_y <- train_data[[target]]
  train_X <- train_data[, setdiff(names(train_data), target), drop = FALSE]
  dtrain <- xgb.DMatrix(data = as.matrix(train_X), label = train_y)

  params <- list(
    objective = "reg:squarederror",
    max_depth = max_depth,
    eta = eta,
    gamma = gamma,
    alpha = alpha,
    lambda = lambda,
    subsample = subsample,
    colsample_bytree = colsample_bytree)

  model <- xgb.train(params = params,
                     data = dtrain,
                     nrounds = nrounds,
                     verbose = 0)

  # Compute calibration residuals on the calibration set
  calib_y <- calib_data[[target]]
  calib_X <- calib_data[, setdiff(names(calib_data), target), drop = FALSE]
  dcalib <- xgb.DMatrix(data = as.matrix(calib_X))
  calib_pred <- predict(model, dcalib)


  # Get point predictions for the new data
  dnew <- xgb.DMatrix(data = as.matrix(newdata))
  new_seeds <- predict(model, dnew)

  calib_errors <- calib_y - calib_pred
  new_preds <- purrr::map(new_seeds, ~ .x + sample(calib_errors, size = n_sim, replace = TRUE))

  return(new_preds)
}


###
bayesian_search <- function(predictors, target, future,
                            past, coverage, max_depth, eta, gamma,
                            alpha, lambda, subsample, colsample_bytree,
                            n_samples, n_exploration, calib_rate, n_sim, nrounds, seed)
{
  set.seed(seed)

  params <- list(past = past, coverage = coverage, max_depth = max_depth, eta = eta, gamma = gamma,
                 alpha = alpha, lambda = lambda, subsample = subsample, colsample_bytree = colsample_bytree)

  default <- list(past = c(1L, 30L), coverage = c(0.05, 0.95), max_depth = c(3L, 10L), eta = c(0.01, 0.3),
                  gamma = c(0, 5), alpha = c(0, 1), lambda = c(0, 1), subsample = c(0, 1), colsample_bytree = c(0, 1))

  bounds <- map2(params, default, ~ {if(is.null(.x)){out <- .y} else {out <- .x}; return(out)})
  tuning_params <- keep(bounds, ~ length(.x) == 2)

  make_alist <- function(args) {
    res <- replicate(length(args), substitute())
    names(res) <- args
    res}

  make_function <- function(args, body, env = parent.frame()) {
    args <- as.pairlist(args)
    eval(call("function", args, body), env)}

  body_expr <- quote(
    {
      pred_funs <- engine(predictors, target, future, past,
                          coverage, calib_rate = calib_rate, n_sim = n_sim, nrounds = nrounds, max_depth, eta,
                          gamma, alpha, lambda, subsample, colsample_bytree, seed)

      entropy <- seq_ent(pred_funs)
      score <- mean(entropy[is.finite(entropy)])

      out <- list(Score = -score, Pred = list(pred_funs))
      return(out)
    })

  args <- make_alist(names(tuning_params))
  wrapper <- make_function(args, body_expr)

  opt <- BayesianOptimization(FUN = wrapper, tuning_params, init_points = n_samples, n_iter = n_exploration, verbose = FALSE)

  best_idx <- which.min(-opt$History$Value)
  best_model <- list_flatten(opt$Pred[[best_idx]])

  history <- round(as.data.frame(opt$History), 4)
  history$entropy <- -history$Value
  history$Value <- NULL
  best_params <- history[best_idx,, drop = FALSE]

  out <- list(history = history, best_model = best_model, best_params = best_params)

  return(out)
}

#####
seq_ent <- function(pred_funs){map_dbl(pred_funs, ~ entropy(.x))}

entropy <- function(pred_fun)
{
  pdf <- pred_fun$dfun
  p_range <- pred_fun$qfun(c(0, 1))
  integrand <- function(x) abs(pdf(x) * log(pdf(x)))
  out <- adaptIntegrate(integrand, p_range[1],  p_range[2])$integral
  return(out)
}



##############

random_search <- function(predictors, target, future,
                          past, coverage, max_depth,
                          eta, gamma, alpha, lambda,
                          subsample, colsample_bytree,
                          n_samples, calib_rate,
                          n_sim, nrounds, seed)
{
  set.seed(seed)

  params <- list(past = past, coverage = coverage, max_depth = max_depth, eta = eta, gamma = gamma,
                 alpha = alpha, lambda = lambda, subsample = subsample, colsample_bytree = colsample_bytree)

  default <- list(past = c(1L, 30L), coverage = c(0.05, 0.95), max_depth = c(3L, 10L), eta = c(0.01, 0.3),
                  gamma = c(0, 5), alpha = c(0, 1), lambda = c(0, 1), subsample = c(0, 1), colsample_bytree = c(0, 1))

  params <- map2(params, default, ~ sampler(.x, n_samples, range = .y))

  plan(multisession, workers = availableCores() - 1)


  models <- future_pmap(params, ~ tryCatch(engine(predictors, target, future,
                                                  past = ..1, coverage = ..2,
                                                  max_depth = ..3, eta = ..4,
                                                  gamma = ..5, alpha = ..6,
                                                  lambda = ..7, subsample = ..8,
                                                  colsample_bytree = ..9,
                                                  calib_rate, n_sim, nrounds, seed), error = function(e) NA), .options = furrr_options(seed = TRUE))

  entropy <- map(models, ~ seq_ent(.x))
  score <- map_dbl(entropy, ~ mean(.x[is.finite(.x)]))

  history <- round(cbind(as.data.frame(params), entropy = score), 4)
  best_idx <- which.min(score)
  best_model <- models[[best_idx]]
  best_params = history[best_idx,, drop = FALSE]

  out <- list(history = history, best_model = best_model, best_params = best_params)
  return(out)
}

# Coarse-to-Fine Optimizer using the random search utility
coarse_to_fine_search <- function(predictors, target, future,
                                  past, coverage, max_depth,
                                  eta, gamma, alpha, lambda,
                                  subsample, colsample_bytree,
                                  n_samples, n_phases, top_k,
                                  calib_rate, n_sim, nrounds, seed)
{
  set.seed(seed)

  exploration_list <- list()
  history_list <- list()
  best_model_list <- list()
  best_params_list <- list()

  if(top_k > n_samples){stop("top_k greater than n_samples")}

  for(s in 1:n_phases)
  {
    exploration <- random_search(predictors, target, future,
                                 past, coverage, max_depth,
                                 eta, gamma, alpha, lambda,
                                 subsample, colsample_bytree,
                                 n_samples, calib_rate,
                                 n_sim, nrounds, seed)

    best_model <- exploration$best_model
    best_params <- exploration$best_params

    history <- exploration$history
    history <- history[order(history$entropy),]

    past <- min(history$past[1:top_k]):max(history$past[1:top_k])
    coverage <- seq(min(history$coverage[1:top_k]), max(history$coverage[1:top_k]), length.out = 100)
    max_depth <- min(history$max_depth[1:top_k]):max(history$max_depth[1:top_k])
    eta <- seq(min(history$eta[1:top_k]), max(history$eta[1:top_k]), length.out = 100)
    gamma <- seq(min(history$gamma[1:top_k]), max(history$gamma[1:top_k]), length.out = 100)
    alpha <- seq(min(history$alpha[1:top_k]), max(history$alpha[1:top_k]), length.out = 100)
    lambda <- seq(min(history$lambda[1:top_k]), max(history$lambda[1:top_k]), length.out = 100)
    subsample <- seq(min(history$subsample[1:top_k]), max(history$subsample[1:top_k]), length.out = 100)
    colsample_bytree <- seq(min(history$colsample_bytree[1:top_k]), max(history$colsample_bytree[1:top_k]), length.out = 100)

    exploration_list[[s]] <- exploration
    history_list[[s]] <- history
    best_model_list[[s]] <- best_model
    best_params_list[[s]] <- best_params
  }

  history <- as.data.frame(cbind(phase = base::rep(1:n_phases, each = n_samples), Reduce(rbind, history_list)))

  best_idx <- which.min(as.data.frame(Reduce(rbind, best_params_list))[, "entropy"])

  best_model <- best_model_list[[best_idx]]
  best_params <- best_params_list[[best_idx]]

  outcome <- list(history = history, best_model = best_model, best_params = best_params)

  return(outcome)
}



####
sampler <- function(vect, n_samp, range = NULL)
{
  if(is.null(vect))
  {
    if(is.numeric(range)){vect <- seq(min(range), max(range), length.out = 1000)}
    if(is.integer(range)){vect <- min(range):max(range)}
    if(is.character(range)|is.logical(range)){vect <- range}
  }

  if(length(vect)==1){samp <- rep(vect, n_samp)}
  if(length(vect) > 1){samp <- sample(vect, n_samp, replace = TRUE)}

  return(samp)
}

###
smart_reframer <- function(ts, seq_len, stride)
{
  n_length <- length(ts)
  if(seq_len > n_length | stride > n_length){stop("vector too short for sequence length or stride")}
  if(n_length%%seq_len > 0){ts <- tail(ts, - (n_length%%seq_len))}
  n_length <- length(ts)
  idx <- seq(from = 1, to = (n_length - seq_len + 1), by = 1)
  reframed <- t(sapply(idx, function(x) ts[x:(x+seq_len-1)]))
  if(seq_len == 1){reframed <- t(reframed)}
  idx <- rev(seq(nrow(reframed), 1, - stride))
  reframed <- reframed[idx,,drop = FALSE]
  colnames(reframed) <- paste0("t", 1:seq_len)
  return(reframed)
}


####
svd_coverage <- function(df, coverage = 0.95)
{
  model <- svd(df)
  if(ncol(df) > 1){rdim <- sum(cumsum(model$d)/sum(model$d) <= coverage)} else {rdim <- 1}
  if(rdim < 1){rdim <- 1}
  redux <- as.data.frame(model$u[, 1:rdim] %*% diag(model$d[1:rdim], rdim, rdim))
  n_feats <- ncol(redux)
  colnames(redux) <- paste0("r", 1:n_feats)
  return(redux)
}

####
mixture <- function(x)
{
  gnd_params <- tryCatch(unlist(paramp(x)[c(2, 4, 5)]), error = function(e) NA)
  if(is.numeric(gnd_params)){gnd_set <- dnormp(x, gnd_params[1], gnd_params[2], gnd_params[3])} else {gnd_set <- NA}

  glogis_params <- tryCatch(glogisfit(x)$parameters, error = function(e) NA)
  if(is.numeric(glogis_params)){glogis_set <- dglogis(x, glogis_params[1], glogis_params[2], glogis_params[3])} else {glogis_set <- NA}

  gev_params <- tryCatch(fgev(x, std.err = FALSE)$estimate, error = function(e) NA)
  if(is.numeric(gev_params)){gev_set <- dgev(x, gev_params[1], gev_params[2], gev_params[3])} else {gev_set <- NA}

  gl_params <- tryCatch(fit.fkml(x)$lambda, error = function(e) NA)
  if(is.numeric(gl_params)){gl_set <- dgl(x, gl_params[1],  gl_params[2], gl_params[3], gl_params[4])} else {gl_set <- NA}

  ald_params <- tryCatch(mleALD(x)$par, error = function(e) NA)
  if(is.numeric(ald_params)){ald_set <- dALD(x, ald_params[1], ald_params[2], ald_params[3])} else {ald_set <- NA}

  hyp_params <- tryCatch(coef(hyperbFit(x)), error = function(e) NA)
  if(is.numeric(hyp_params)){hyp_set <- tryCatch(dhyperb(x, hyp_params[1], hyp_params[2], hyp_params[3], hyp_params[4]), error = function(e) NA)} else {hyp_set <- NA}

  dens <- gnd_set + glogis_set + gev_set + gl_set + ald_set + hyp_set
  finite_idx <- is.finite(dens)
  values <- sample(x[finite_idx], size = 1000, replace = TRUE, prob = dens[finite_idx])
  if(all(x >= 0)){values <- values[values >= 0]}
  if(all(x < 0)){values <- values[values < 0]}

  pred_fun <- edfun::edfun(values, cut = 0)

  return(pred_fun)
}

###
plot_graph <- function(ts, pred_funs, alpha = 0.05, dates = NULL, line_size = 1.3, label_size = 11,
                       forcat_band = "seagreen2", forcat_line = "seagreen4", hist_line = "gray43",
                       label_x = "Horizon", label_y= "Forecasted Var", date_format = "%b-%Y")
{
  preds <- Reduce(rbind, map(pred_funs, ~ quantile(.x$rfun(1000), probs = c(alpha, 0.5, (1-alpha)))))
  colnames(preds) <- c("lower", "median", "upper")
  future <- nrow(preds)

  if(is.null(dates)){x_hist <- 1:length(ts)} else {x_hist <- as.Date(as.character(dates))}
  if(is.null(dates)){x_forcat <- length(ts) + 1:nrow(preds)} else {x_forcat <- as.Date(as.character(tail(dates, 1)))+ 1:future}

  forecast_data <- data.frame(x_forcat = x_forcat, preds)
  historical_data <- data.frame(x_all = as.Date(c(x_hist, x_forcat)), y_all = c(ts = ts, pred = preds[, "median"]))

  plot <- ggplot()+ geom_line(data = historical_data, aes(x = .data$x_all, y = .data$y_all), color = hist_line, linewidth = line_size)
  plot <- plot + geom_ribbon(data = forecast_data, aes(x = x_forcat, ymin = .data$lower, ymax = .data$upper), alpha = 0.3, fill = forcat_band)
  plot <- plot + geom_line(data = forecast_data, aes(x = x_forcat, y = median), color = forcat_line, linewidth = line_size)
  if(!is.null(dates)){plot <- plot + scale_x_date(name = paste0("\n", label_x), date_labels = date_format)}
  if(is.null(dates)){plot <- plot + scale_x_continuous(name = paste0("\n", label_x))}
  plot <- plot + scale_y_continuous(name = paste0(label_y, "\n"), labels = number)
  plot <- plot + ylab(label_y) + theme_bw()
  plot <- plot + theme(axis.text=element_text(size=label_size), axis.title=element_text(size=label_size + 2))

  return(plot)
}


dts <- function(ts, lag = 1) tail(ts, -lag)/head(ts, -lag)-1



