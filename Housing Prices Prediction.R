# Joaquin Hidalgo-Estrada
# RIDGE vs LASSO Model Performance

#Set seed replicability
set.seed(1)

### libraries ###

library(car)
library(MASS)
library(olsrr)
library(ISLR)
library(Metrics)

################################

df <- data.frame(housing_csv)

attach(df)
head(df)
dim(df)
summary(df)

################################


### Split Data ###

train_obs <- sample(1:nrow(df), 0.8*nrow(df), replace=FALSE)

train_set <- df[train_obs,]
dim(train_set)
summary(train_set)

test_obs <- -train_obs
test_set <- df[test_obs,]
dim(test_set)


### Data Cleaning ###

# remove NA values since they represent only 1% of the entire dataset
train_set_2 <- train_set[!is.na(train_set$total_bedrooms),]
summary(train_set_2)
dim(train_set_2)

test_set_2 <- test_set[!is.na(test_set$total_bedrooms),]
summary(test_set_2)
dim(test_set_2)


### Exploratory Data Analysis ###

# categorical
table(ocean_proximity)
barplot(table(train_set_2$ocean_proximity), main = "Ocean Proximity")

# numeric
hist(train_set_2$housing_median_age)                                 # distribution looks good
boxplot(train_set_2$housing_median_age, xlab = "Housing Median Age by block") # no visible outliers

hist(train_set_2$total_rooms)                                     # extreme right skewness
boxplot(train_set_2$total_rooms, xlab = "Total Rooms by block")   # numerous potential outliers

hist(train_set_2$longitude)                                       # unusual distribution
boxplot(train_set_2$longitude, xlab = "Longitude (how far west)") # no visible outliers

hist(train_set_2$latitude)                                        # unusual distribution
boxplot(train_set_2$latitude, xlab = "Latitude (how far north)")  # no visible distribution

hist(train_set_2$total_bedrooms)                                        # extreme right skewness
boxplot(train_set_2$total_bedrooms, xlab = "Total Bedrooms by block")   # numerous outliers

hist(train_set_2$population)                                     # extreme right skewness
boxplot(train_set_2$population, xlab = "Population by block")    # numerous outliers

hist(train_set_2$households)                                                     # extreme right skewness
boxplot(train_set_2$households, xlab = "Total Number of Households by block")    # numerous outliers

hist(train_set_2$median_income)                                     # moderate right skewness
boxplot(train_set_2$median_income, xlab = "Median Income by block") # visible outliers

hist(train_set_2$median_house_value)                                          # low right skewness
boxplot(train_set_2$median_house_value, xlab = "Median House Value by block") # few outliers

# basic scatterplots (linearity)
pairs(train_set_2[, -c(10)])

# correlation matrix
cor(train_set_2[, -c(10)])  # potential multicollinearity issues


### First Model ###

# Partial Residual Plots
first_model <- lm(median_house_value ~ ., data=train_set_2)
crPlots(first_model)

plot(first_model)
boxCox(first_model)

### Second Model ###

second_model <- lm(log(median_house_value) ~ ., data=train_set_2) # better results
crPlots(second_model)

plot(second_model)

### Third Model ###
# best model

third_model <- lm((median_house_value)^(1/4) ~ ., data=train_set_2) # even better results
crPlots(third_model)

plot(third_model)
summary(third_model)


########################
### Residual Analysis ###

# cook's Distance
2*sqrt(10/16357)
tail(sort(cooks.distance(third_model)), n=10) # flag 9881, 8317, 15361
flagged <- train_set[c(9881, 15361, 8317), ]
print(flagged)

# remove because even though values are plausible they dont represent our data well.
to_remove <- c(9881, 15361, 8317)
train_set_clean <- train_set_2[-to_remove, ]
dim(train_set_clean)

########################


### Max Model ###

max_model <- lm((median_house_value)^(1/4) ~ ., data=train_set_clean) # even better results
crPlots(max_model)

plot(max_model)
summary(max_model)


### Multicollinearity ###
ols_coll_diag(max_model)


### Scaling Data ###
train_set_clean$longitude <- ((train_set_clean$longitude-mean(train_set_clean$longitude))/sd(train_set_clean$longitude))
train_set_clean$latitude <- (train_set_clean$latitude-mean(train_set_clean$latitude))/sd(train_set_clean$latitude)
train_set_clean$housing_median_age <- ((train_set_clean$housing_median_age-mean(train_set_clean$housing_median_age))/sd(train_set_clean$housing_median_age)) 
train_set_clean$total_rooms <- (train_set_clean$total_rooms-mean(train_set_clean$total_rooms))/sd(train_set_clean$total_rooms)
train_set_clean$total_bedrooms <- ((train_set_clean$total_bedrooms-mean(train_set_clean$total_bedrooms))/sd(train_set_clean$total_bedrooms)) 
train_set_clean$population <- (train_set_clean$population-mean(train_set_clean$population))/sd(train_set_clean$population)
train_set_clean$households <- ((train_set_clean$households-mean(train_set_clean$households))/sd(train_set_clean$households)) 
train_set_clean$median_income <- (train_set_clean$median_income-mean(train_set_clean$median_income))/sd(train_set_clean$median_income)

summary(train_set_clean)

max_model <- lm((median_house_value)^(1/4) ~ ., data=train_set_clean)
ols_coll_diag(max_model) # remove total_bedrooms first

### Addressing Multicollinearity ###

collinearity_model <- lm((median_house_value)^(1/4) ~ longitude + latitude +
                    housing_median_age + total_rooms +
                    population + households + median_income +
                    ocean_proximity, data=train_set_clean)

summary(collinearity_model)
ols_coll_diag(collinearity_model) # remove latitude next

collinearity_model_2 <- lm((median_house_value)^(1/4) ~ longitude +
                           housing_median_age + total_rooms +
                           population + households + median_income +
                           ocean_proximity, data=train_set_clean)
ols_coll_diag(collinearity_model_2) # remove households next

# include the transformed response in the dataset
train_set_clean$median_house_value_qtr <- train_set_clean$median_house_value^0.25

summary(train_set_clean)
collinearity_model_3 <- lm(median_house_value_qtr ~ longitude +
                             housing_median_age + total_rooms +
                             population + median_income +
                             ocean_proximity, data=train_set_clean)
ols_coll_diag(collinearity_model_3) # DONE

# lower R^2 but multicollinearity solved
summary(collinearity_model_3)


### Model Selection ###

# stepwise
ols_step_both_p(collinearity_model_3, p_enter = 0.1, p_remove = 0.1)

final_model <- lm(median_house_value_qtr ~ longitude +
                    housing_median_age + total_rooms +
                    population + median_income +
                    ocean_proximity, data=train_set_clean)

# final model summary
summary(final_model)

### Model Testing ###

#Calculate y-hats for training model.
pred_train <- predict(final_model, train_set_clean)
mse(train_set_clean$median_house_value_qtr, pred_train)

# standardize test set same way train set was standardized
test_set_2$longitude <- ((test_set_2$longitude-mean(test_set_2$longitude))/sd(test_set_2$longitude))
test_set_2$latitude <- (test_set_2$latitude-mean(test_set_2$latitude))/sd(test_set_2$latitude)
test_set_2$housing_median_age <- ((test_set_2$housing_median_age-mean(test_set_2$housing_median_age))/sd(test_set_2$housing_median_age)) 
test_set_2$total_rooms <- (test_set_2$total_rooms-mean(test_set_2$total_rooms))/sd(test_set_2$total_rooms)
test_set_2$total_bedrooms <- ((test_set_2$total_bedrooms-mean(test_set_2$total_bedrooms))/sd(test_set_2$total_bedrooms)) 
test_set_2$population <- (test_set_2$population-mean(test_set_2$population))/sd(test_set_2$population)
test_set_2$households <- ((test_set_2$households-mean(test_set_2$households))/sd(test_set_2$households)) 
test_set_2$median_income <- (test_set_2$median_income-mean(test_set_2$median_income))/sd(test_set_2$median_income)

summary(test_set_2)

#Calculate y-hats for testing model.
pred_test <- predict(final_model, test_set_2)
mse(test_set_2$median_house_value^(1/4), pred_test)


#########################
# Ridge vs Lasso Analysis

library(glmnet)

# list of possible lambda values
grid = 10^seq(10, -2, length = 100)

# drop response var
X1 <- model.matrix(~ . -1, data = train_set_clean[, -c(9, 11)])
y <- train_set_clean$median_house_value_qtr 
summary(X1)

#Set seed replicability
set.seed(1)

# cross validation and hyperparameter tuning
cv.out=cv.glmnet(X1,train_set_clean$median_house_value_qtr,alpha=1) # lasso

cv.out2=cv.glmnet(X1,train_set_clean$median_house_value_qtr,alpha=0) # ridge

# plots log(lambda) vs MSE to find best lambda value
plot(cv.out)

plot(cv.out2)

#Print out penalty for coefficients
bestlam1 <- cv.out$lambda.min # lasso
print(bestlam1)

bestlam2 <- cv.out2$lambda.min # ridge
print(bestlam2)

# use the optimized lambda value in lasso regression
out = glmnet(X1, train_set_clean$median_house_value_qtr, alpha = 1, lambda = grid)

out2 = glmnet(X1, train_set_clean$median_house_value_qtr, alpha = 0, lambda = grid)

# print out the lasso coefficients
lasso.coef = predict(out, type = "coefficients", s = bestlam1)
lasso.coef

ridge.coef <- predict(out2, type = "coefficients", s = bestlam2)
ridge.coef

### MSE for ridge and lasso ###
# RIDGE

X_test <- model.matrix(~ . -1, data = test_set_2[, -c(9, 11)])
y_test <- test_set_2$median_house_value^(1/4)
summary(X_test)

# calculate y-hats for training data.
pred_train_r <- predict(out2, s = bestlam2, newx = X1)
mse_train_r <- mean((y - pred_train_r)^2)
print(mse_train_r)

# calculate y-hats for testing data
pred_test_r <- predict(out2, s = bestlam2, newx = X_test)
mse_test_r <- mean((y_test - pred_test_r)^2)
print(mse_test_r)


# LASSO

# calculate y-hats for training data.
pred_train_l <- predict(out, s = bestlam1, newx = X1)
mse_train_l <- mean((y - pred_train_l)^2)
print(mse_train_l)

# calculate y-hats for testing data
pred_test_l <- predict(out, s = bestlam1, newx = X_test)
mse_test_l <- mean((y_test - pred_test_l)^2)
print(mse_test_l)
