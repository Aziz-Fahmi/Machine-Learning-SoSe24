# Load the CSV data into RStudio
Facebook <- read.csv("C:/Users/Aziz/Desktop/FH SWF/SoSe24/Machine Learning SoSe24/ML Task 1/Facebook.csv", sep = "\t", header = TRUE)

# Remove the unused column 'X'
Facebook_clean <- Facebook[, -which(colnames(Facebook) == "X")]

# Plot MAU against Total Assets
plot(Facebook_clean$MAU, Facebook_clean$Total.Assets,
     xlab = "Monthly Active Users (MAU)",
     ylab = "Total Assets",
     main = "MAU vs Total Assets Scatter Plot",
     col = "blue", pch = 16)

# Fit a linear regression model
linear_model <- lm(Total.Assets ~ MAU, data = Facebook_clean)

# Add linear regression line to the plot
abline(linear_model, col = "red")

# Define the Metcalfe utility function
metcalfe_utility <- function(x, a) {
  return(a * x * (x - 1) / 2)
}

# Define a function to calculate sum of squared residuals
ssr <- function(a) {
  predicted_assets <- metcalfe_utility(Facebook_clean$MAU, a)
  return(sum((predicted_assets - Facebook_clean$Total.Assets)^2))
}

# Optimize for the parameter 'a' using Brent's method
optimized_a <- optimize(f = ssr, interval = c(0, 1e5))$minimum

# Print the optimized value of 'a'
cat("Optimized value of 'a':", optimized_a, "\n")


# Add Metcalfe utility function curve to the plot with optimized 'a'
curve(metcalfe_utility(x, a = optimized_a), add = TRUE, col = "black")

# Add legend
legend("topleft",
       legend = c("Data Points", "Linear Regression", "Metcalfe Utility Function"),
       col = c("blue", "red", "black"),
       pch = c(16, NA, NA),
       lty = c(NA, 1, 1)
)

# Summary of Metcalfe utility function
metcalfe_model <- lm(Total.Assets ~ MAU + I(MAU^2), data = Facebook_clean)
summary(metcalfe_model)

# Summary of linear regression function
summary(linear_model)

# Calculate BIC for linear regression model
BIC_linear <- BIC(linear_model)

# Calculate BIC for Metcalfe utility function model
residuals <- residuals(metcalfe_model)
n <- length(residuals)
k <- length(coef(metcalfe_model))  # Number of parameters including intercept
BIC_metcalfe <- n * log(sum(residuals^2) / n) + k * log(n)

# Print BIC values for comparison
cat("BIC for Linear Regression Model:", BIC_linear, "\n")
cat("BIC for Metcalfe Utility Function Model:", BIC_metcalfe, "\n")
