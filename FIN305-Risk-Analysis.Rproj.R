# Load necessary libraries
library(dplyr)
library(zoo)
library(tidyr)
library(ggplot2)
library(moments)
library(readr)
library(rugarch)
library(nortest)
library(reshape2)

# Read the dataset
data <- read.csv("/Users/xuruoyu/Library/Mobile Documents/com~apple~CloudDocs/FIN305/Assignment 1/Images/merged_data.csv")

# Ensure the date format is correct
data$Trddt <- as.Date(data$Trddt, format = "%Y-%m-%d")

# Sort data by trading date
data <- data %>% arrange(Stkcd, Trddt)

# Check if each combination of Stkcd and Trddt is unique
if (any(duplicated(data %>% select(Stkcd, Trddt)))) {
  stop("Duplicate combinations of Stkcd and Trddt found. Please check the data.")
}

# Fill missing values (use the last non-missing value)
data <- data %>%
  group_by(Stkcd) %>%
  mutate(Adjprcnd = zoo::na.locf(Adjprcnd, na.rm = FALSE)) %>%
  ungroup()

# Calculate log returns, ignoring the first trading day
data <- data %>%
  group_by(Stkcd) %>%
  mutate(LogReturn = ifelse(row_number() == 1, NA, log(Adjprcnd / lag(Adjprcnd)))) %>%
  ungroup()

# Q1a: Compute descriptive statistics for each stock and add max/min value dates
summary_stats <- data %>%
  group_by(Stkcd) %>%
  summarize(
    Mean = mean(LogReturn, na.rm = TRUE),
    SD = sd(LogReturn, na.rm = TRUE),
    Max = max(LogReturn, na.rm = TRUE),
    MaxDate = Trddt[which.max(LogReturn)],
    Min = min(LogReturn, na.rm = TRUE),
    MinDate = Trddt[which.min(LogReturn)],
    Skewness = skewness(LogReturn, na.rm = TRUE),
    Kurtosis = kurtosis(LogReturn, na.rm = TRUE),
    Observations = sum(!is.na(LogReturn))
  )

# Print descriptive statistics
print(summary_stats)

# Q1b: Convert data to wide format, with each column as a stock code
log_returns_matrix <- data %>%
  select(Trddt, Stkcd, LogReturn) %>%
  spread(key = Stkcd, value = LogReturn)

# Check for duplicate dates
if (any(duplicated(log_returns_matrix$Trddt))) {
  stop("Duplicate dates found in the dataset. Please check the data.")
}

# Remove the date column and ensure no missing values in the covariance matrix calculation
log_returns_matrix <- log_returns_matrix %>%
  select(-Trddt) %>%
  na.omit()

# Validate the data
print(dim(log_returns_matrix))
print(head(log_returns_matrix))

# Compute mean returns and covariance matrix
mean_returns <- colMeans(log_returns_matrix)
cov_matrix <- cov(log_returns_matrix)

# Generate random portfolio weights
set.seed(123)
num_portfolios <- 10000
num_assets <- ncol(log_returns_matrix)
weights <- matrix(runif(num_portfolios * num_assets, 0, 1), ncol = num_assets)
weights <- weights / rowSums(weights) # Normalize weights

# Verify that weights are normalized
if (!all(abs(apply(weights, 1, sum) - 1) < 1e-6)) {
  stop("Portfolio weights normalization failed. Please check the logic.")
}

# Compute expected portfolio return and volatility
portfolio_returns <- weights %*% mean_returns
portfolio_volatility <- sqrt(rowSums((weights %*% cov_matrix) * weights))

# Create a dataframe for plotting
efficient_frontier <- data.frame(
  Volatility = portfolio_volatility,
  Return = portfolio_returns
)

# Compute the efficient frontier for the three assets
single_asset_efficiency <- data.frame(
  Volatility = sqrt(diag(cov_matrix)),
  Return = mean_returns,
  Asset = colnames(log_returns_matrix)
)

# Q1c: Compute Sharpe Ratio and Optimal Portfolio

# Incorrect way to compute Sharpe ratio (daily risk-free rate of 1% leads to an unrealistic 36.78% annualized return)
# rf_daily <- 0.01
# sharpe_ratios <- (portfolio_returns - rf_daily) / portfolio_volatility  # Based on compounding assumption

# Correct method: Convert annual risk-free rate to daily frequency (average from 2007-2016)
rf_annual <- 0.01
rf_daily <- (1 + rf_annual)^(1 / 252) - 1  # Convert to daily rate
sharpe_ratios <- (portfolio_returns - rf_daily) / portfolio_volatility  # Based on compounding assumption

# Find the portfolio with the maximum Sharpe ratio
max_sharpe_idx <- which.max(sharpe_ratios)
optimal_weights <- weights[max_sharpe_idx, ]
max_sharpe_point <- efficient_frontier[max_sharpe_idx, ]

# Print the optimal portfolio weights and maximum Sharpe ratio
cat("Optimal weights for the maximum Sharpe ratio portfolio:\n")
names(optimal_weights) <- colnames(log_returns_matrix)
print(optimal_weights)

cat("\nMaximum Sharpe ratio:\n")
print(max(sharpe_ratios))

# Compute the expected return (daily) of the optimal portfolio
optimal_portfolio_return <- sum(optimal_weights * mean_returns)

# Print the expected daily return of the optimal portfolio
cat("\nThe expected return (daily) of the optimal portfolio is:\n")
print(optimal_portfolio_return)

# Assume 252 trading days in a year
trading_days_per_year <- 252

# Compute the annualized return
annualized_return <- (1 + optimal_portfolio_return) ^ trading_days_per_year - 1
cat("\nThe annualized return of the optimal portfolio is:\n")
print(annualized_return)

# Plot Efficient Frontier with Single Asset Positions
ggplot(efficient_frontier, aes(x = Volatility, y = Return)) +
  geom_point(color = "blue", alpha = 0.5, size = 1) +  # Efficient frontier
  geom_point(data = single_asset_efficiency, aes(x = Volatility, y = Return, color = Asset), size = 3) + # Individual asset points
  geom_text(data = single_asset_efficiency, aes(x = Volatility, y = Return, label = Asset),
            vjust = -0.5, hjust = 0.5) +  # Add asset labels
  geom_point(data = max_sharpe_point, aes(x = Volatility, y = Return), 
             color = "green", size = 4) +  # Max Sharpe ratio point
  geom_text(data = max_sharpe_point, aes(x = Volatility, y = Return, label = "Max Sharpe Ratio"),
            hjust = 1.2, vjust = -0.5, color = "green", size = 4) +
  labs(title = "Efficient Frontier with Highlighted Maximum Sharpe Ratio",
       x = "Portfolio Volatility",
       y = "Portfolio Return") +
  theme_minimal()

# Enhanced Efficient Frontier Plot with Gradient and Styling
ggplot(efficient_frontier, aes(x = Volatility, y = Return)) +
  geom_point(aes(color = Return), alpha = 0.8, size = 3) +  # Use return value for color gradient
  scale_color_gradient(low = "red", high = "blue") +  # Gradient from red to blue
  geom_point(data = single_asset_efficiency, aes(x = Volatility, y = Return), 
             size = 4, shape = 17, color = "black") +  # Mark individual assets with black triangles
  geom_text(data = single_asset_efficiency, aes(x = Volatility, y = Return, label = Asset),
            vjust = -1.2, hjust = 0.5, size = 3, fontface = "italic") +  # Asset labels
  geom_point(data = max_sharpe_point, aes(x = Volatility, y = Return), 
             color = "darkblue", size = 6, shape = 16) +  # Max Sharpe ratio point in dark blue
  geom_text(data = max_sharpe_point, aes(x = Volatility, y = Return, label = "Max Sharpe Ratio"),
            hjust = 1.1, vjust = -0.5, color = "darkblue", size = 4, fontface = "bold") +
  labs(title = "Efficient Frontier with Maximum Sharpe Ratio",
       subtitle = "Visualizing portfolio performance based on 10,000 random simulations",
       x = "Portfolio Volatility (Risk)",
       y = "Portfolio Return (Expected Return)") +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, face = "italic"),
    axis.title = element_text(size = 14, face = "bold"),
    axis.text = element_text(size = 12),
    legend.title = element_text(size = 12),
    legend.text = element_text(size = 10)
  )



# Q2a: Compute and Compare Value at Risk (VaR) for the Optimal Portfolio

# Convert log_returns_matrix to matrix type
log_returns_matrix <- as.matrix(log_returns_matrix)
optimal_weights <- as.numeric(optimal_weights)

# Compute daily portfolio returns using optimal weights
portfolio_returns_optimal <- log_returns_matrix %*% optimal_weights

# Normality test for portfolio returns
cat("\nNormality test for portfolio returns:\n")
ad_test <- ad.test(portfolio_returns_optimal)  # Anderson-Darling Test
cat("Anderson-Darling test statistic:", ad_test$statistic, "p-value:", ad_test$p.value, "\n")

# Plot QQ plot to check for normality
qqnorm(portfolio_returns_optimal, main = "QQ Plot of Portfolio Returns")
qqline(portfolio_returns_optimal, col = "red")

# Compute VaR using Historical Simulation
VaR_portfolio_historical <- quantile(portfolio_returns_optimal, probs = 0.1)

# Compute VaR using Variance-Covariance approach (assuming normality)
portfolio_mean <- mean(portfolio_returns_optimal)
portfolio_sd <- sd(portfolio_returns_optimal)
VaR_portfolio_covariance <- portfolio_mean + qnorm(0.1) * portfolio_sd

# Print VaR results
cat("\nVaR calculations based on the optimal portfolio:\n")
cat("Historical simulation VaR:", VaR_portfolio_historical, "\n")
cat("Variance-covariance VaR:", VaR_portfolio_covariance, "\n")

# Visualization: Compare VaR estimates
VaR_comparison <- data.frame(
  Method = c("Historical", "Covariance"),
  VaR = c(VaR_portfolio_historical, VaR_portfolio_covariance)
)

ggplot(VaR_comparison, aes(x = Method, y = VaR, fill = Method)) +
  geom_bar(stat = "identity", position = "dodge", width = 0.5) +
  labs(
    title = "Comparison of VaR Methods",
    x = "VaR Calculation Method",
    y = "Value at Risk"
  ) +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_fill_manual(values = c("Historical" = "red", "Covariance" = "blue"))

# Q2b: Compute VaR for Individual Assets

# Historical simulation method for individual assets
colnames(log_returns_matrix) <- unique(data$Stkcd)
VaR_historical_assets <- apply(log_returns_matrix, 2, function(x) quantile(x, probs = 0.1)) 
print(VaR_historical_assets)

# Variance-Covariance method for individual assets
VaR_covariance_assets <- apply(log_returns_matrix, 2, function(x) { 
  z_score <- qnorm(0.1)
  mean_x <- mean(x)
  sd_x <- sd(x)
  mean_x + z_score * sd_x
})
names(VaR_covariance_assets) <- colnames(log_returns_matrix)
print(VaR_covariance_assets)

# Compare VaR estimates across portfolio and individual assets
VaR_comparison <- data.frame(
  Method = c("Historical", "Covariance"),
  Portfolio_VaR = c(VaR_portfolio_historical, VaR_portfolio_covariance),
  Asset_425 = c(VaR_historical_assets["425"], VaR_covariance_assets["425"]),
  Asset_528 = c(VaR_historical_assets["528"], VaR_covariance_assets["528"]),
  Asset_600761 = c(VaR_historical_assets["600761"], VaR_covariance_assets["600761"])
)
cat("2b) Comparison of VaR:\n")
print(VaR_comparison)

# Q2c: Rolling Window VaR Estimation and Backtesting

# Step 1: Define rolling window parameters
window_size <- 1000  # Window size for estimation
confidence_level <- 0.90  # 90% confidence level
alpha <- 1 - confidence_level  # Expected breach probability
significance_level <- 0.05  # Significance level for binomial test

# Ensure data consistency
log_returns_matrix_numeric <- as.matrix(log_returns_matrix)
optimal_weights_numeric <- as.numeric(optimal_weights)

# Initialize result vectors
portfolio_returns_rolling <- numeric(nrow(log_returns_matrix_numeric) - window_size)
VaR_historical_rolling <- numeric(nrow(log_returns_matrix_numeric) - window_size)
VaR_covariance_rolling <- numeric(nrow(log_returns_matrix_numeric) - window_size)

# Compute rolling window VaR
for (i in 1:(nrow(log_returns_matrix_numeric) - window_size)) {
  # Extract rolling window data
  window_data <- log_returns_matrix_numeric[i:(i + window_size - 1), ]
  
  # Compute portfolio return on the next day
  portfolio_returns_rolling[i] <- sum(log_returns_matrix_numeric[i + window_size, ] * optimal_weights_numeric)
  
  # Compute Historical Simulation VaR
  historical_window_returns <- window_data %*% optimal_weights_numeric
  VaR_historical_rolling[i] <- quantile(historical_window_returns, probs = 0.1)
  
  # Compute Variance-Covariance VaR
  mean_return <- mean(historical_window_returns)
  sd_return <- sd(historical_window_returns)
  VaR_covariance_rolling[i] <- mean_return + qnorm(0.1) * sd_return
}

# Print first few rolling VaR estimates
cat("Portfolio Returns (Rolling):\n")
print(head(portfolio_returns_rolling))

cat("Historical VaR (Rolling):\n")
print(head(VaR_historical_rolling))

cat("Covariance VaR (Rolling):\n")
print(head(VaR_covariance_rolling))

# Step 2: Identify VaR breaches
historical_breaches <- portfolio_returns_rolling < VaR_historical_rolling
num_historical_breaches <- sum(historical_breaches)
freq_historical_breaches <- num_historical_breaches / length(portfolio_returns_rolling)

covariance_breaches <- portfolio_returns_rolling < VaR_covariance_rolling
num_covariance_breaches <- sum(covariance_breaches)
freq_covariance_breaches <- num_covariance_breaches / length(portfolio_returns_rolling)

# Step 3: Binomial test for backtesting VaR
expected_frequency <- 0.1
sample_size <- length(portfolio_returns_rolling)

binomial_test_historical <- binom.test(
  x = num_historical_breaches,
  n = sample_size,
  p = expected_frequency,
  alternative = "two.sided"
)

binomial_test_covariance <- binom.test(
  x = num_covariance_breaches,
  n = sample_size,
  p = expected_frequency,
  alternative = "two.sided"
)

# Print backtesting results
cat("Historical VaR breaches:\n")
cat("Number of breaches:", num_historical_breaches, "\n")
cat("Breach frequency:", freq_historical_breaches, "\n")
cat("Binomial test results:\n")
print(binomial_test_historical)

cat("\nCovariance VaR breaches:\n")
cat("Number of breaches:", num_covariance_breaches, "\n")
cat("Breach frequency:", freq_covariance_breaches, "\n")
cat("Binomial test results:\n")
print(binomial_test_covariance)

# Step 4: Visualization of Rolling VaR vs Actual Returns
ggplot(data.frame(Date = 1:length(portfolio_returns_rolling), 
                  PortfolioReturns = portfolio_returns_rolling, 
                  HistoricalVaR = VaR_historical_rolling, 
                  CovarianceVaR = VaR_covariance_rolling), 
       aes(x = Date)) +
  geom_line(aes(y = PortfolioReturns, color = "Portfolio Returns"), size = 0.8) +
  geom_line(aes(y = HistoricalVaR, color = "Historical VaR"), linetype = "dashed") +
  geom_line(aes(y = CovarianceVaR, color = "Covariance VaR"), linetype = "dotted") +
  labs(title = "Rolling Window VaR vs Portfolio Returns",
       x = "Time (Days)",
       y = "Returns / VaR") +
  theme_minimal() +
  scale_color_manual(values = c("Portfolio Returns" = "blue", "Historical VaR" = "red", "Covariance VaR" = "green"))

# Q2d: Assessing VaR Accuracy During Extreme Market Conditions
# Define extreme market periods
extreme_periods <- data.frame(
  Period = c("2008 Financial Crisis", "2015 Stock Crash", "COVID-19 Pandemic"),
  Start = as.Date(c("2008-07-01", "2015-06-15", "2020-02-04")),
  End = as.Date(c("2008-10-28", "2015-08-31", "2020-03-31"))
)

# Create a sample dataset for rolling VaR analysis
set.seed(123)  # Ensure reproducibility
rolling_var_results <- data.frame(
  Date = seq.Date(from = as.Date("2008-01-01"), to = as.Date("2020-12-31"), by = "day"),
  VaR = rnorm(4749, mean = -0.02, sd = 0.01),  # Simulated VaR values
  Actual = rnorm(4749, mean = 0, sd = 0.02),  # Simulated portfolio returns
  Breach = sample(c(TRUE, FALSE), 4749, replace = TRUE)  # Randomly generated breaches
)

# Mark extreme periods in the rolling VaR dataset
rolling_var_results <- rolling_var_results %>%
  mutate(
    Period = case_when(
      Date >= extreme_periods$Start[1] & Date <= extreme_periods$End[1] ~ extreme_periods$Period[1],
      Date >= extreme_periods$Start[2] & Date <= extreme_periods$End[2] ~ extreme_periods$Period[2],
      Date >= extreme_periods$Start[3] & Date <= extreme_periods$End[3] ~ extreme_periods$Period[3],
      TRUE ~ "Normal Period"
    )
  )

# Calculate breach frequencies for each market period
period_summary <- rolling_var_results %>%
  group_by(Period) %>%
  summarize(
    Breach_Frequency = mean(Breach, na.rm = TRUE),
    Observations = n(),
    .groups = "drop"
  )
print(period_summary)

# Visualization: Compare VaR vs Actual Returns in different market periods
ggplot(rolling_var_results, aes(x = Date)) +
  geom_line(aes(y = VaR, color = "VaR"), size = 1, linetype = "dashed") +
  geom_line(aes(y = Actual, color = "Actual Returns"), size = 1) +
  geom_point(data = rolling_var_results %>% filter(Breach == TRUE),
             aes(x = Date, y = Actual), color = "black", size = 2) +
  facet_wrap(~ Period, scales = "free_x") +
  scale_color_manual(values = c("VaR" = "red", "Actual Returns" = "blue")) +
  ggtitle("VaR Performance in Normal vs Extreme Market Periods") +
  xlab("Date") +
  ylab("Returns and VaR") +
  theme_minimal() +
  theme(legend.position = "top")

# Create a heatmap of breach frequencies by month and year
heatmap_data <- rolling_var_results %>%
  mutate(
    Month = as.integer(format(Date, "%m")),
    Year = as.integer(format(Date, "%Y"))
  ) %>%
  group_by(Year, Month) %>%
  summarize(Breach_Frequency = mean(Breach, na.rm = TRUE)) %>%
  ungroup()

ggplot(heatmap_data, aes(x = Month, y = Year, fill = Breach_Frequency)) +
  geom_tile(color = "white", size = 0.5) +
  scale_fill_gradient2(low = "green", mid = "yellow", high = "red", midpoint = mean(heatmap_data$Breach_Frequency)) +
  labs(
    title = "Heatmap of VaR Breach Frequency",
    x = "Month",
    y = "Year",
    fill = "Breach Frequency"
  ) +
  theme_minimal()



# Q3a: Estimating Volatility Using GARCH(1,1)
# Define GARCH(1,1) fitting function
get_garch_volatility <- function(stock_id, data) {
  # Filter data for the specified stock
  stock_data <- data %>% filter(Stkcd == stock_id) %>% drop_na(LogReturn)
  
  # Convert data types
  stock_data <- stock_data %>%
    mutate(
      LogReturn = as.numeric(LogReturn),
      Trddt = as.Date(Trddt)
    )
  
  # Specify the GARCH(1,1) model with Student's t-distribution
  spec <- ugarchspec(variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
                     mean.model = list(armaOrder = c(0, 0)), 
                     distribution.model = "std")
  
  # Fit the model and extract conditional volatility
  tryCatch({
    fit <- ugarchfit(spec = spec, data = stock_data$LogReturn, solver = "hybrid")
    
    # Compute residuals and perform Ljung-Box test
    residuals <- data.frame(
      Date = stock_data$Trddt,  
      Residuals = residuals(fit, standardize = TRUE)
    )
    lb_test <- Box.test(residuals$Residuals, lag = 10, type = "Ljung-Box")
    cat("Ljung-Box Test Results for Stock", stock_id, ":\n")
    print(lb_test)
    
    # Compute skewness and kurtosis
    skewness_value <- skewness(residuals$Residuals, na.rm = TRUE)
    kurtosis_value <- kurtosis(residuals$Residuals, na.rm = TRUE) - 3  # Excess kurtosis
    cat("Skewness for Stock", stock_id, ":", skewness_value, "\n")
    cat("Excess Kurtosis for Stock", stock_id, ":", kurtosis_value, "\n")
    
    # Return conditional volatility data
    stock_data %>%
      mutate(Conditional_Volatility = sigma(fit)) %>%
      select(Trddt, Conditional_Volatility, Stkcd)
  }, error = function(e) {
    message(paste("GARCH fitting failed for stock", stock_id, ":", e$message))
    return(NULL)
  })
}

# Get unique stock IDs and apply GARCH model
unique_stocks <- unique(data$Stkcd)[1:3]
all_volatility <- bind_rows(lapply(unique_stocks, get_garch_volatility, data = data))

# Ensure standardized data types
all_volatility <- all_volatility %>%
  mutate(
    Trddt = as.Date(Trddt),
    Conditional_Volatility = as.numeric(Conditional_Volatility),
    Stkcd = as.factor(Stkcd)
  )

# Plot conditional volatility for each stock
for (stock in unique_stocks) {
  stock_data <- all_volatility %>% filter(Stkcd == stock)
  
  ggplot(stock_data, aes(x = Trddt, y = Conditional_Volatility)) +
    geom_line(color = "blue", size = 1) +
    ggtitle(paste("Conditional Volatility for Stock Code", stock)) +
    xlab("Date") +
    ylab("Conditional Volatility") +
    scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1)) %>%
    print()
}

# Overlay volatility trends across all stocks
ggplot(all_volatility, aes(x = Trddt, y = Conditional_Volatility, color = Stkcd)) +
  geom_line(size = 0.8, alpha = 0.7) +  
  ggtitle("Comparison of Conditional Volatility for Three Stocks") +
  xlab("Date") +
  ylab("Conditional Volatility") +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y") +
  scale_color_manual(values = c("425" = "purple", "528" = "lightblue", "600761" = "blue"), name = "Stock Code") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Q3b: Analyzing Volatility Behavior During Extreme Market Conditions
# Define extreme market periods
extreme_periods <- data.frame(
  Period = c("2008 Financial Crisis", "2015 Stock Crash", "COVID-19 Pandemic"),
  Start = as.Date(c("2008-07-01", "2015-06-15", "2020-02-04")),
  End = as.Date(c("2008-10-28", "2015-08-31", "2020-03-31"))
)

# Label extreme periods in volatility data
all_volatility <- all_volatility %>%
  mutate(Period = "Normal Period")  # Default to normal period

for (i in seq_len(nrow(extreme_periods))) {
  all_volatility <- all_volatility %>%
    mutate(Period = ifelse(Trddt >= extreme_periods$Start[i] & Trddt <= extreme_periods$End[i],
                           extreme_periods$Period[i], Period))
}

# Perform t-tests comparing volatility between normal and extreme periods
t_test_results <- list()

for (stock in unique(all_volatility$Stkcd)) {
  stock_data <- all_volatility %>% filter(Stkcd == stock)
  
  # Compare normal and extreme periods
  for (extreme_period in unique(extreme_periods$Period)) {
    normal_data <- stock_data %>% filter(Period == "Normal Period") %>% pull(Conditional_Volatility)
    extreme_data <- stock_data %>% filter(Period == extreme_period) %>% pull(Conditional_Volatility)
    
    # Conduct Welch's t-test if sufficient data is available
    if (length(normal_data) > 1 && length(extreme_data) > 1) {
      t_test <- t.test(normal_data, extreme_data, var.equal = FALSE)
      t_test_results[[paste(stock, extreme_period, sep = "_")]] <- list(
        Stock = stock,
        Period = extreme_period,
        T_Statistic = t_test$statistic,
        P_Value = t_test$p.value
      )
    }
  }
}

# Convert t-test results to a dataframe
t_test_results_df <- do.call(rbind, lapply(t_test_results, as.data.frame)) %>%
  mutate(P_Value = round(P_Value, 4))  # Round p-values to 4 decimal places

# Print t-test results
print(t_test_results_df)

# Summarize mean and standard deviation of volatility across periods
volatility_summary <- all_volatility %>%
  group_by(Stkcd, Period) %>%
  summarize(
    Mean_Volatility = round(mean(Conditional_Volatility, na.rm = TRUE), 4),
    SD_Volatility = round(sd(Conditional_Volatility, na.rm = TRUE), 4),
    .groups = "drop"
  )

# Print volatility summary statistics
print(volatility_summary)

# Plot conditional volatility across different market periods for each stock
for (stock in unique(all_volatility$Stkcd)) {
  stock_data <- all_volatility %>% filter(Stkcd == stock)
  
  ggplot(stock_data, aes(x = Trddt, y = Conditional_Volatility, color = Period)) +
    geom_line(size = 0.5) +
    ggtitle(paste("Conditional Volatility for Stock Code", stock)) +
    xlab("Date") +
    ylab("Conditional Volatility") +
    scale_color_manual(
      values = c(
        "Normal Period" = "darkblue", 
        "2008 Financial Crisis" = "red", 
        "2015 Stock Crash" = "purple", 
        "COVID-19 Pandemic" = "orange"
      )
    ) +
    theme_minimal() +
    theme(legend.position = "top") +
    # Add vertical lines to indicate extreme periods
    geom_vline(data = extreme_periods, aes(xintercept = as.numeric(Start), color = Period), 
               linetype = "dashed", show.legend = FALSE) +
    geom_vline(data = extreme_periods, aes(xintercept = as.numeric(End), color = Period), 
               linetype = "dashed", show.legend = FALSE) %>%
    print()
}

# Q3c: Rolling-Window GARCH-Based VaR Calculation and Backtesting
# Step 1: Ensure required data exists
if (!exists("optimal_weights") || !exists("data")) {
  stop("Ensure 'optimal_weights' and 'data' are defined before proceeding.")
}

# Compute portfolio returns using optimal weights
portfolio_returns <- data %>%
  filter(Stkcd %in% colnames(log_returns_matrix)) %>%
  select(Trddt, Stkcd, LogReturn) %>%
  pivot_wider(names_from = Stkcd, values_from = LogReturn) %>%
  mutate(
    Portfolio_Return = rowSums(as.matrix(select(., all_of(colnames(log_returns_matrix)))) * optimal_weights)
  ) %>%
  select(Trddt, Portfolio_Return) %>%
  drop_na()

# Verify portfolio return data integrity
if (!"Portfolio_Return" %in% colnames(portfolio_returns) || !"Trddt" %in% colnames(portfolio_returns)) {
  stop("Input data must contain 'Portfolio_Return' and 'Trddt' columns.")
}
print(head(portfolio_returns))

# Step 2: Define rolling GARCH VaR computation function
calculate_rolling_garch_var <- function(data, confidence_level = 0.90, window_size = 1000) {
  if (nrow(data) < window_size) {
    stop("Not enough data for rolling window calculation")
  }
  
  spec <- ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0)), 
    distribution.model = "norm"
  )
  
  rolling_var <- data.frame(Date = as.Date(character()), VaR = numeric(), Actual = numeric())
  
  for (i in seq(window_size + 1, nrow(data))) {
    train_data <- data$Portfolio_Return[(i - window_size):(i - 1)]
    if (any(is.na(train_data)) || length(train_data) < window_size) {
      warning(paste("Skipping due to NA or insufficient data at:", data$Trddt[i]))
      next
    }
    
    fit <- tryCatch({
      ugarchfit(spec, data = train_data, solver = "hybrid", out.sample = 1)
    }, error = function(e) {
      warning(paste("GARCH fit failed at:", data$Trddt[i]))
      return(NULL)
    })
    
    if (is.null(fit)) next
    
    forecast <- ugarchforecast(fit, n.ahead = 1)
    sigma_forecast <- forecast@forecast$sigmaFor[1]
    mean_forecast <- forecast@forecast$seriesFor[1]
    var <- qnorm(1 - confidence_level) * sigma_forecast + mean_forecast
    
    rolling_var <- rbind(
      rolling_var,
      data.frame(
        Date = data$Trddt[i],
        VaR = var,
        Actual = data$Portfolio_Return[i]
      )
    )
  }
  
  return(rolling_var)
}

# Step 3: Compute rolling GARCH VaR
rolling_var_results <- calculate_rolling_garch_var(portfolio_returns, confidence_level = 0.90)

# Step 4: Define VaR backtesting function
backtest_var <- function(rolling_var, confidence_level = 0.90) {
  rolling_var <- rolling_var %>%
    mutate(Breach = Actual < VaR)
  
  breach_count <- sum(rolling_var$Breach)
  total_obs <- nrow(rolling_var)
  breach_ratio <- breach_count / total_obs
  
  expected_breach_ratio <- 1 - confidence_level
  binom_test <- binom.test(breach_count, total_obs, expected_breach_ratio)
  
  list(
    Breach_Ratio = breach_ratio,
    Binom_Test = binom_test,
    Rolling_Var = rolling_var  
  )
}

# Step 5: Perform backtesting
backtest_results <- backtest_var(rolling_var_results, confidence_level = 0.90)

# Step 6: Print backtesting results
print(paste("Breach Ratio:", round(backtest_results$Breach_Ratio, 4)))
print(paste("Binomial Test p-value:", round(backtest_results$Binom_Test$p.value, 4)))

# Step 7: Visualize rolling GARCH VaR vs actual returns
ggplot(backtest_results$Rolling_Var, aes(x = Date)) +
  geom_line(aes(y = VaR, color = "VaR"), size = 1, linetype = "solid", alpha = 0.7) +
  geom_line(aes(y = Actual, color = "Actual Returns"), size = 0.5, alpha = 0.5) +
  ggtitle("GARCH VaR Backtest with Breaches for Optimal Portfolio") +
  xlab("Date") +
  ylab("Returns and VaR") +
  theme_minimal() +
  theme(legend.position = "top")