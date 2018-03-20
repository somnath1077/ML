# Load libraries
library(MASS)
library(ISLR)

lm.fit = lm(medv ~ lstat, data=Boston)

print(summary(lm.fit))
print(names(lm.fit))
print(coef(lm.fit))

# Predict with 95% confidence interval
predict(lm.fit, data.frame(lstat=c(5, 10, 15, 20)), interval="confidence")
