skip_on_cran()
skip_on_ci()
skip("This test is for manual testing only")
test_that("dlo_dynamics returns the correct output", {
  # Test with example values
  expect_equal(dlo_dynamics(0, 0, 0.1, 0.01, 0.5, 0.1), c(0, 0.1))
})

# Test the generate_q function
test_that("generate_q returns the correct output", {
  # Test with example values
  expect_equal(dim(generate_q(100, 0.5)), c(100, 3))
})

# Test the calculate_moving_average function
test_that("calculate_moving_average returns the correct output", {
  # Test with example values
  expect_equal(calculate_moving_average(c(1,2,3,4,5), 2), c(1.5, 2, 3, 4, 4.5))
})

# Test the generate_observables function
test_that("generate_observables returns the correct output", {
  # Test with example values
  X <- matrix(c(runif(10,0.5,0.75),runif(10,0.5,0.75)), ncol=2)
  expect_equal(dim(generate_observables(X, 10, 4, 0.05)), c(10, 8))
})

# Test the simulate_video function
test_that("simulate_video returns the correct output", {
  # Test with example values
  expect_equal(dim(simulate_video(dt = 0.01, num_steps = 50, num_observables = 4, eta_n = 0.5, zeta_n = 0.5, eta = 0.5, zeta = 0.5, sigma_q = 0.1, sd_observable = 0.1, loadings = 0.8, window_size = 10)), c(50, 14))
})
