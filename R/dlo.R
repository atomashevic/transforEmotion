# Simulation of emotion data based on Damped Linear Oscillator (DLO) model

#' @importFrom Matrix bdiag
#' @title Dynamics function of the DLO model
#' @description This function calculates the dynamics of a system using the DLO (Damped Linear Oscillator) model based on Equation 1 (Ollero et al., 2023).
#' The DLO model is a second-order differential equation that describes the behavior of a damped harmonic oscillator.
#' The function takes in the current state of the system, the derivative of the state, the damping coefficient, the time step,
#' and the values of the eta and zeta parameters. It returns the updated derivative of the state.
#'
#' @param x Numeric.
#' The current state of the system (value of the latent score).
#' @param dxdt Numeric.
#' The derivative of the state (rate of change of the latent score).
#' @param q Numeric. 
#' The damping coefficient.
#' @param dt Numeric.
#' The time step.
#' @param eta Numeric. 
#' The eta parameter of the DLO model.
#' @param zeta Numeric. 
#' The zeta parameter of the DLO model.
#' 
#' @export
#'
#' @return A numeric vector containing the updated derivative of the state.
#'
#' @references
#' Ollero, M. J. F., Estrada, E., Hunter, M. D., & Cáncer, P. F. (2023).
#'  Characterizing affect dynamics with a damped linear oscillator model: Theoretical considerations and recommendations for individual-level applications. 
#'  \emph{Psychological Methods}. 
#' \url{https://doi.org/10.1037/met0000615}
# Updated: 13.11.2023.
dlo_dynamics <- function(x, dxdt, q, dt, eta, zeta){
  dxdt_new <- dxdt
  dx2dt <- eta * x + zeta * dxdt + q
  return(c(dxdt_new, dx2dt))
}

# mvrnorm function from MASS package to remove dependency on MASS package
# Updated: 26.10.2023.
MASS_mvrnorm <- function(n = 1, mu, Sigma, tol = 1e-06, empirical = FALSE, EISPACK = FALSE){
  p <- length(mu)
  if (!all(dim(Sigma) == c(p, p))) 
    stop("incompatible arguments")
  if (EISPACK) 
    stop("'EISPACK' is no longer supported by R", domain = NA)
  eS <- eigen(Sigma, symmetric = TRUE)
  ev <- eS$values
  if (!all(ev >= -tol * abs(ev[1]))) 
    stop("'Sigma' is not positive definite")
  X <- matrix(rnorm(p * n), n)
  if (empirical) {
    X <- scale(X, TRUE, FALSE)
    X <- X %*% svd(X, nu = 0)$v
    X <- scale(X, FALSE, TRUE)
  }
  X <- drop(mu) + eS$vectors %*% diag(sqrt(pmax(ev, 0)), p) %*% 
    t(X)
  nm <- names(mu)
  if (is.null(nm) && !is.null(dn <- dimnames(Sigma))) 
    nm <- dn[[1]]
  dimnames(X) <- list(nm, NULL)
  if (n == 1) 
    drop(X)
  else t(X)
  
}

#' @title  Generate a matrix of Dynamic Error values for the DLO simulation
#' @description This function generates a matrix of Dynamic Error values (q) for the DLO simulation.
#' @param num_steps Numeric integer.
#' The number of time steps used in the simulation.
#' @param sigma_q Numeric. 
#' Standard deviation of the Dynamic Error/
#' @return A (num_steps X 3) matrix of Dynamic Error values for neutral, negative and positive emotion latent score.
# Updated: 26.10.2023.
generate_q <- function(num_steps, sigma_q) {
  q <- matrix(rnorm(3 * num_steps, mean = c(0, 0, 0), sd = sigma_q), ncol = 3)
  return(q)
}

#' @title  Calculate the moving average for a time series
#' @description This function calculates the moving average for a time series.
#' @param data Matrix or Data frame.
#' The time series data
#' @param window_size Numeric integer.
#' The size of the moving average window.
#' @return Matrix or Data frame containing the moving average values.
# Updated: 13.11.2023.
calculate_moving_average <- function(data, window_size) {
  n <- length(data)
  ma <- numeric(n)
  for (i in (window_size):(n - window_size)) {
    ma[i] <- mean(data[(i - window_size + 1):(i + window_size - 1)])
  }
  # Extend the moving average to the edges of the time series
  for (i in 1:(window_size - 1)) {
    ma[i] <- mean(data[1:(i + window_size - 1)])
  }
  for (i in (n - window_size + 1):n) {
    ma[i] <- mean(data[(i - window_size + 1):n])
  }
  return(ma)
}

#' @title Generate observable emotion scores data from latent variables
#' @description Function to generate observable data from 2 latent variables (negative and positive affect).
#' The function takes in the latent variable scores, the number of time steps, the number of observable variables per latent factor,
#' and the measurement error variance. It returns a matrix of observable data.
#' The factor loadings are not the same for all observable variables. They have uniform random noise added to them (between -0.15 and 0.15).
#' The loadings are scaled so that the sum of the loadings for each latent factor is 2, to introduce a ceiling effect and to differentiate the dynamics of specific emotions. This is further empahsized by adding small noise to the measurement error variance for each observed variable (between -0.01 and 0.01).
#' 
#' @param X Matrix or Data frame.
#' The (num_steps X 2) matrix of latent variable scores.
#' @param num_steps Numeric integer.
#' Number of time steps.
#' @param num_obs Numeric integer.
#' The number of observable variables per latent factor.
#' @param error Numeric.
#' Measurement error variance.
#' @param loadings Numeric (default = 0.8).
#' The default initial loading of the latent variable on the observable variable.
#' @return A (num_steps X num_obs) Matrix or Data frame containing the observable variables.
# Updated: 13.11.2023.
generate_observables <- function(X, num_steps, num_obs, error, loadings=0.8){
  loads <- lapply(rep(loadings, 2), rep, num_obs)
  loads <- lapply(loads, function(x){x + runif(length(x), -0.15, 0.15)}) # add noise to loadings
  LoadMat <- as.matrix(Matrix::bdiag(loads))
  LoadMat <- LoadMat / (colSums(LoadMat) + runif(1, -0.05, 0.05))*2 # scale loadings so that sum of loadings for each latent factor is 2
  var <- rep(error, num_obs*2) + runif(num_obs*2, -0.01, 0.01) # add noise to measurement error variance
  Q <- diag(var,num_obs*2,num_obs*2)
  e <- t(MASS_mvrnorm(num_steps, matrix(0,num_obs*2,1),Q))
  obs.data <- LoadMat %*% t(X) + e
  obs.data <- as.data.frame(t(obs.data))
  obs.data[obs.data < 0] <- 0 # negative values of observed variables are not possible
  obs.data <- obs.data/2 # scale observed variables so that the sum of the observed variables for each latent factor is roughly 1
  return(obs.data)
}

#' @title Simulate latent and observed emotion scores for a single "video"
#' @description This function simulates emotions in a video using the DLO model implemented as continuous time state space model. The function takes in several parameters, including the time step, number of steps, number of observables, and various model parameters. It returns a data frame containing the simulated emotions and their derivatives, as well as smoothed versions of the observables.
#' The initial state of the video is always the same. Neutral score is 0.5 and both positive and negative emotion score is 0.25. 
#' To simulate more realistic time series, there is an option of including a sudden jump in the emotion scores. This is done by emphasizing the effect of the dominant emotion during the period where the derivative of the latent variable is high. The observable value of the strongest emotion from the positive or negative group will spike in the next k time step (emph.dur). The probability of this happening is p at each time step in which the derivative of the latent variable is greater than 0.2. The jump is proportionate to the derivative of the latent variable and the sum of the observable values of the other emotions.
#'
#' @param dt Numeric real.
#' The time step for the simulation (in minutes).
#' @param num_steps Numeric real.
#' Total length of the video (in minutes).
#' @param num_observables  Numeric integer. 
#' The number of observables to generate per factor. Total number of observables generated is 2 x num_observables.
#' @param eta_n Numeric.
#' The eta parameter for the neutral state.
#' @param zeta_n Numeric.
#' The zeta parameter for the neutral state.
#' @param eta Numeric.
#' The eta parameter for the positive and negative emotions.
#' @param zeta Numeric.
#' The zeta parameter for the positive and negative emotions.
#' @param sigma_q Numeric.
#' The standard deviation of Dynamic Error of the q(t) function.
#' @param sd_observable Numeric.
#' The standard deviation of the measurement error.
#' @param loadings Numeric (default = 0.8).
#' The default initial loading of the latent variable on the observable variable.
#' @param window_size Numeric integer.
#' The window size for smoothing the observables.
#' @param emph Logical.
#' Whether to emphasize the effect of dominant emotion (default is FALSE).
#' @param emph.dur Numeric integer.
#' The duration of the emphasis (default is 10).
#' @param emph.prob Numeric.
#' The probability of the dominant emotion being emphasized (default is 0.5).
#' @return A data frame (num_steps X (6 + num_observables)) containing the latent scores for neutral score, positive emotions, negative emotions and their derivatives, as well as smoothed versions of the observables.
#' @export
#' @examples
#' simulate_video(dt = 0.01, num_steps = 50, num_observables = 4, eta_n = 0.5, zeta_n = 0.5, eta = 0.5, zeta = 0.5, sigma_q = 0.1, sd_observable = 0.1, loadings = 0.8, window_size = 10)
# Updated: 13.11.2023.

simulate_video <- function(dt, num_steps, num_observables, eta_n, zeta_n, eta, zeta, sigma_q, sd_observable, loadings, window_size, emph = FALSE, emph.dur = 10, emph.prob = 0.5){
  # Initial condition for neutral state
  n_0 <- 0.5
  dndt_0 <- 4
  # Initial conditions for negative emotion
  x_0 <- 0.25
  dxdt_0 <- 2
  # Initial conditions for positive emotion
  y_0 <- 0.25
  dydt_0 <- 2
  
  # Initialize arrays to store time series data

  n <- numeric(num_steps)
  dndt <- numeric(num_steps)
  x <- numeric(num_steps)
  dxdt <- numeric(num_steps)
  y <- numeric(num_steps)
  dydt <- numeric(num_steps)
  
  # Initial values for x1 and y1
  n[1] <- n_0
  dndt[1] <- dndt_0
  x[1] <- x_0
  dxdt[1] <- dxdt_0
  y[1] <- y_0
  dydt[1] <- dydt_0
  
  # Generate q(t): dynamic error of the DLO model
  q_t <- generate_q(num_steps, sigma_q)
  
  # Euler method integration for x1, y1
  # Loop for every step/frame of the simulation

  for (i in 2:num_steps) {
    # Calculate derivatives for n
    derivatives_n <- dlo_dynamics(n[i - 1], dndt[i - 1], q_t[i, 1], dt, eta_n, zeta_n)
    n[i] <- n[i - 1] + derivatives_n[1] * dt
    dndt[i] <- derivatives_n[2] * dt
    derivatives_x <- dlo_dynamics(x[i - 1], dxdt[i - 1], q_t[i, 2], dt, eta, zeta)
    # Calculate derivatives for X/Negative emotions
    x[i] <- x[i - 1] + derivatives_x[1] * dt
    dxdt[i] <- derivatives_x[2] * dt
    derivatives_y <- dlo_dynamics(y[i - 1], dydt[i - 1], q_t[i, 3], dt, eta, zeta)
    # Calculate derivatives for Y/Positive emotions
    y[i] <- y[i - 1] + derivatives_y[1] * dt
    dydt[i] <- derivatives_y[2] * dt
    # Data cleaning and normalization
    if (!is.na(n[i])){
    if (n[i] < 0) {
      n[i] <- 0
    }
    if (x[i] < 0) {
      x[i] <- 0
    }
    if (y[i] < 0) {
      y[i] <- 0
    }
    total_sum <-  n[i] + x[i] + y[i]
    if (total_sum != 1) {
      scaling_factor <- 1 / total_sum
      n[i] <- n[i] * scaling_factor
      x[i] <- x[i] * scaling_factor
      y[i] <- y[i] * scaling_factor
    }
    }
    else{
      n[i] <- 0
      x[i] <- 0
      y[i] <- 0
    }
  } #end loop here
  # Calculate factors and observables
  factors <- cbind(x,y)
  observables <- generate_observables(factors, num_steps, num_observables, sd_observable, loadings)

  # Calculate moving averages for observables
  smoothed_x <- apply(observables[,1:num_observables], 2, calculate_moving_average, window_size)
  smoothed_y <- apply(observables[,(num_observables+1):(2*num_observables)], 2, calculate_moving_average, window_size)
  # Aggregate results into a data frame
  results <- as.data.frame(cbind(n,dndt, x, dxdt, y, dydt, smoothed_x, smoothed_y))
  # If emphasis is TRUE, emphasize the effect of dominent emotions:
  if (emph){
    results <- emphasize(results, num_observables, num_steps, emph.dur, emph.prob)
  }
  colnames(results) <- c("Nt","dNt","Ng","dNg", "Ps","dPs", paste0("N",1:num_observables), paste0("P",1:num_observables))
  return(results)
}

#' @title Generate and emphasize sudden jumps in emotion scores
#' @description This function generates and emphasizes the effect of strong emotions expressions during the period where the derivative of the latent variable is high. The observable value of the strongest emotion from the positive or negative group will spike in the next k time steps. The probability of this happening is p at each time step in which the derivative of the latent variable is greater than 0.2. The jump is proportionate to the derivative of the latent variable and the sum of the observable values of the other emotions.
#' @param data Data frame.
#' The data frame containing the latent and observable variables created by the \code{simulate_video} function.
#' @param num_observables Numeric integer.
#' The number of observable variables per latent factor.
#' @param num_steps Numeric integer.
#' The number of time steps used in the simulation.
#' @param k Numeric integer.
#' The mumber of time steps to emphasize the effect of strong emotions on future emotions (default is 10). Alternatively: the length of a strong emotional episode.
#' @param p Numeric.
#' The probability of the strongest emotion being emphasized in the next k time steps (default is 0.5).
#' @return A data frame containing the updated observable variables.
# Updated: 26.10.2023.
emphasize <- function(data, num_observables, num_steps, k = 10, p = 0.5){
  # emphasize negative emotions
  neg_col <- 6+num_observables
  dn.inc <- which(data[,4] > quantile(data[,4], 0.8))
  for (i in dn.inc){
    if (i>20 & i<(num_steps-k)){
    dr.ratio <- data[i,4]/0.2
    q = runif(1)
    if (q < p){
      for (j in 1:k){
        sum.p <- sum(data[i+j,7:neg_col])
        w = which(data[i+j,7:neg_col]==max(data[i+j,7:neg_col])) + 6
        r = which(7:neg_col!=w)+6
        data[i+j,w] = data[i+j,w] + runif(1,0.1,0.25)*sum.p*dr.ratio # the height of the spike is proportional to the sum of the other emotions and the ratio of the derivative of the latent variable
      }
    }
  }
  }
  # emphasize positive emotions
  pos_col <- 6+2*num_observables
  dp.inc <- which(data[,6]>0.2)
  for (i in dp.inc){
    if (i>20 & i<(num_steps-k)){
    dr.ratio <- data[i,6]/0.2
    q = runif(1)
    if (q < p){
      for (j in 1:k){
        sum.p <- sum(data[i+j,(neg_col+1):pos_col])
        w = which(data[i+j,(neg_col+1):pos_col]==max(data[i+j,(neg_col+1):pos_col])) + 6 + num_observables
        r = which((neg_col+1):pos_col!=w)+6+num_observables
        data[i+j,w] = data[i+j,w] + runif(1,0,0.25)*sum.p*dr.ratio
    }
    }
    }
  }
  data[,7:pos_col] <- apply(data[,7:pos_col], 2, calculate_moving_average, 2) # smooth the observable variables, even out the huge spikes due to large derivatives
  return(data)
}
#' @title Plot the latent or the observable emotion scores.
#' @description Function to plot the latent or the observable emotion scores.
#' @param df Data frame.
#' The data frame containing the latent and observable variables created by the \code{simulate_video} function.
#' @param mode Character.
#' The mode of the plot. Can be either 'latent', 'positive' or 'negative'.
#' @param title Character.
#' The title of the plot. Default is an empty title, ' '.
#' @return A plot of the latent or the observable emotion scores.
#' @export
# Updated: 8.01.2024.
plot_sim_emotions <- function(df, mode = 'latent', title = ' '){
  n_obs <- (ncol(df)-6)/2
  if (n_obs > 4) {
    blues <- colorRampPalette(c("#ade4f7", "#1B1BD0"))(n_obs)
    reds  <- colorRampPalette(c("lightpink", "#D01B1B"))(n_obs)
  }
  else{
    blues <- c("#1B1BD0", "#ade4f7", "#95D2EC", "#47abd8")
    reds <- c("#D01B1B", "lightpink", "#FF4242", "#F47A7A")

  }
  cols = c(reds, blues)
  if (mode == 'latent')
  {
    title_label <-  ifelse(title==' ', paste0("Latent Emotion Scores"), title)
    plot(df$Nt, type = "l", col = "#7C878EB2", ylim = c(0, 1), main = title_label, xlab = "Frames", ylab = "Scores")
    lines(df$Ng, col = "#D01B1B")
    lines(df$Ps, col = "#1B1BD0")
    legend("topleft", legend = c("Neutral","Negative","Positive"), col = c("#7C878EB2","#D01B1B", "#1B1BD0"), lty = 1)
  }
  else if (mode=='positive') {
     title_label <-  ifelse(title==' ', paste0("Positive Emotion Scores"), title)
     maxy <- max(df[,(6+n_obs+1):(6+2*n_obs)]) + 0.1
     maxy <- ifelse(maxy > 1, 1, maxy)
     plot(df$P1, type = "l", col = "#1B1BD0", ylim = c(0, maxy), main = "Positive emotions", xlab = "Frames", ylab = "Scores")
     for (i in 2:n_obs) {
       lines(df[,6+n_obs+i], col = blues[i])
     }
      legend("topleft", legend = c(rep(paste0("P",1:n_obs))), col = blues, lty = 1)
  } else {
    title_label <-  ifelse(title==' ', paste0("Negative Emotion Scores"), title)
    maxy <- max(df[,7:(6+n_obs)]) + 0.1
    maxy <- ifelse(maxy > 1, 1, maxy)
    plot(df$N1, type = "l", col = "#D01B1B", ylim = c(0, maxy), main = "Negative emotions", xlab = "Frames", ylab = "Scores")
    for (i in 2:n_obs) {
      lines(df[,6+i], col = reds[i])
    }
    legend("topleft", legend = c(rep(paste0("N",1:n_obs))), col = reds, lty = 1)
  }
}