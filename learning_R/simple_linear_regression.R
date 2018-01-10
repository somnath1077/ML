# Load libraries
library(MASS)
library(ISLR)

lm.fit = lm(medv ~ lstat, data=Boston)
print(summary(lm.fit))

