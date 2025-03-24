library(testthat)

# Create a simple dummy predictors dataset for testing
set.seed(42)
n_obs <- 100
df <- data.frame(
  target_series = cumsum(rnorm(n_obs)),
  predictor1   = cumsum(rnorm(n_obs)),
  predictor2   = cumsum(rnorm(n_obs))
)

# Test for search method "none"
test_that("xpect with 'none' search returns expected structure", {
  result <- xpect(predictors = df,
                  target = "target_series",
                  future = 2,
                  search = "none",
                  seed = 123)

  # Expect the result to be a list with the required components.
  expect_true(is.list(result))
  expect_true(is.na(result$history))
  expect_true(is.na(result$best_params))
  expect_true(is.list(result$best_model))
  expect_s3_class(result$plot, "ggplot")
  expect_true(!is.null(result$time_log))
})

# Test for search method "random_search"
test_that("xpect with 'random_search' returns expected structure", {
  result <- xpect(predictors = df,
                  target = "target_series",
                  future = 2,
                  past = 5L,
                  lambda = NULL,
                  eta = c(0.01, 0.05),
                  search = "random_search",
                  n_samples = 3,  # small sample size for test speed
                  seed = 123)

  expect_true(is.list(result))
  expect_true(is.data.frame(result$history))
  expect_true(all(!is.na(result$best_params)))
  expect_true(is.list(result$best_model))
  expect_s3_class(result$plot, "ggplot")
})

# Test for search method "bayesian"
test_that("xpect with 'bayesian' search returns expected structure", {
  result <- xpect(predictors = df,
                  target = "target_series",
                  future = 2,
                  past = 5,
                  lambda = NULL,
                  eta = c(0.01, 0.05),
                  search = "bayesian",
                  n_samples = 5,       # initial sampling points
                  n_exploration = 1,   # number of exploratory evaluations
                  seed = 123)

  expect_true(is.list(result))
  expect_true(is.data.frame(result$history))
  expect_true(all(!is.na(result$best_params)))
  expect_true(is.list(result$best_model))
  expect_s3_class(result$plot, "ggplot")
})

test_that("xpect with 'coarse_to_fine' search returns expected structure", {
  # Note: Ensure top_k is less than or equal to n_samples.
  result <- xpect(predictors = df,
                  target = "target_series",
                  future = 2,
                  past = 5L,
                  lambda = NULL,
                  eta = c(0.01, 0.05),
                  search = "coarse_to_fine",
                  n_samples = 3,
                  n_phases = 2,
                  top_k = 2,
                  seed = 123)

  expect_true(is.list(result))
  expect_true(is.data.frame(result$history))
  expect_true(all(!is.na(result$best_params)))
  expect_true(is.list(result$best_model))
  expect_s3_class(result$plot, "ggplot")
})
