# Load data
data <- data.frame(
  Year = c(2008:2023),
  MAU = c(100, 300, 608, 845, 1056, 1228, 1393, 1591, 1860, 2129, 2320, 2498, 2797, 2912, 2963, 3065),
  Total_Assets = c(505, 1009, 2990, 6331, 15103, 17895, 39966, 49407, 64961, 84524, 97334, 133376, 159316, 165987, 185727, 229623)
)

# Plot MAU vs Total Assets
plot(data$MAU, data$Total_Assets, type = "l", col = "blue", lwd = 2, xlab = "MAU", ylab = "Total Assets", main = "MAU vs Total Assets")

# Define Metcalfe utility function
metcalfe_function <- function(x, a) {
  return(a * x * (x - 1) / 2)
}

# Fit Metcalfe utility function
metcalfe_fit <- nls(Total_Assets ~ metcalfe_function(MAU, a), data = data, start = list(a = 1))

# Fit linear regression
linear_fit <- lm(Total_Assets ~ MAU, data = data)

# Plot data and fitted curves
lines(data$MAU, fitted(metcalfe_fit), col = "red", lwd = 2, lty = 2)
lines(data$MAU, predict(linear_fit), col = "green", lwd = 2, lty = 2)

# Add legend
legend("topleft", legend = c("Data", "Metcalfe Utility", "Linear Regression"), col = c("blue", "red", "green"), lwd = 2, lty = 1)




# Calculate R-squared for linear regression
linear_r_squared <- summary(linear_fit)$r.squared

# Print R-squared value for linear regression
cat("R-squared for Linear Regression:", linear_r_squared, "\n")

# Calculate sum of squared residuals for Metcalfe utility function
metcalfe_residuals <- residuals(metcalfe_fit)
metcalfe_ssr <- sum(metcalfe_residuals^2)

# Calculate total sum of squares
total_ssr <- sum((data$Total_Assets - mean(data$Total_Assets))^2)

# Calculate R-squared for Metcalfe utility function
metcalfe_r_squared <- 1 - metcalfe_ssr / total_ssr

# Print R-squared value for Metcalfe utility function
cat("R-squared for Metcalfe Utility Function:", metcalfe_r_squared, "\n")




