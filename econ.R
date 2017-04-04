setwd("~/Pythprog/Rprog/data")

library(dplyr)
library(ggplot2)
library(mcsm)

initial <- read.csv("binom2.csv", header = TRUE)

#row.names(initial) 							# writing row names to the console (also counting no of rows)
colnames(initial) 								# writing variable or field names to the console

summary(initial)

# building a binary logit model ###########

model <- glm(formula = dbt ~ bm + ag, family = "binomial", data = initial)
summary(model)

# trial simulations ###################
Nsim = 10^4
x = runif(Nsim)
x1 = x[-Nsim]
x2 = x[-1]

par(mfrow=c(1,3)) 
hist(x) 
plot(x1,x2) 
acf(x)

# simulations ###################
Nsim = 10^4
x = runif(Nsim)
x1 = initial$ag
x2 = initial$bm

par(mfrow=c(1,3)) 
hist(x) 
plot(x1,x2) 
acf(x)
