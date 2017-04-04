set.seed(10)
x <- rep(0:1, each = 5)
e <- rnorm(10, 0, 20)
y <- 0.5 + 2 * x + e
#
library(datasets)
Rprof()
fit <- lm(y ~ x1 + x2)
Rprof(NULL
#
outcome <- read.csv("outcome-of-care-measures.csv", colClasses = "character")
head(outcome)