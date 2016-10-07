##############################################
# CMPT 459 Programming Assignment 3
# Ivy Tse
##############################################

#***** QUESTION 1 *****#
titanic <- read.csv("titanic.csv")
titanic[titanic==""] <- NA # set blank values to NA
set.seed(1)
data <- sort(sample(nrow(titanic), nrow(titanic)*0.8))
training <- titanic[data,] # 80% training data
test <- titanic[-data,] # 20% test data
 
#***** QUESTION 2 *****#
# Insert average age for missing age values
# Delete missing rows for missing Embarked values
# Delete entire Cabin Column
# Removed Name, PassengerID, and Ticket columns due to unique values

training$Age[is.na(training$Age)] <- mean(training$Age, na.rm=TRUE)
test$Age[is.na(test$Age)] <- mean(test$Age, na.rm=TRUE)

training <- training[!is.na(training$Embarked),]
test <- test[!is.na(test$Embarked),]

training$Cabin <- NULL
test$Cabin <- NULL

# Set the Survived values as factors and not as integers so that a 
# classification tree can be generated
training$Survived <- factor(training$Survived)
test$Survived <- factor(test$Survived)

new_training <- training[c("Survived", "Pclass", "Sex", "Age", "SibSp",
"Parch", "Fare", "Embarked")]

new_test <- test[c("Survived", "Pclass", "Sex", "Age", "SibSp","Parch", 
"Fare", "Embarked")]

#***** QUESTION 3 *****#
library(tree)
training_tree <- tree(Survived ~., data = new_training)
training_tree

plot.new()
plot(training_tree)
text(training_tree, all = TRUE)

# The size of the tree is 15 since there are 15 nodes in total in this
# tree. The attributes that this tree uses are Sex, Pclass, Fare, Age, and 
# SibSp.

#***** QUESTION 4 *****#
cv_tree_training <- cv.tree(training_tree)

#find the index of the smallest standard deviation as the best index
best_index <- which(cv_tree_training$dev == min(cv_tree_training$dev))
best_tree_training <- cv_tree_training$size[best_index]

tree_training_pruned <- prune.misclass(training_tree, 
best = best_tree_training)

plot.new()
plot(tree_training_pruned)
text(training_tree, all = TRUE)

# The size of the best decision tree is also of size 15.

#***** QUESTION 5 *****#
library(ROCR)
pruned_predictions <- predict(tree_training_pruned, new_test, type="class")
actual_results <- test$Survived
confusion_matrix <- table(actual_results, pruned_predictions)
confusion_matrix

mean(actual_results == pruned_predictions) # Accuracy = 0.8156425

# convert both pruned_predictions and actual_results to integers so that
# prediction function can be used
pp <- as.integer(as.character(pruned_predictions))
actual <- as.integer(as.character(actual_results))

prediction_v_actual <- prediction(pp, actual)
ROC_curve <- performance(prediction_v_actual, measure="tpr", x.measure="fpr")
plot(ROC_curve)

auc <- performance(prediction_v_actual, measure = "auc")
auc@y.values[[1]] # auc of regression model is 0.7927106

# The accuracy of the model is 0.8156425 and the AUC is 0.7927106

#***** QUESTION 6 *****#
library(randomForest)

randForest261 <- randomForest(Survived ~., data = new_training, nTree = 261)
rf261_predict <- predict(randForest261, new_test, type="class")
mean(actual_results == rf261_predict) # Accuracy = 0.8324022

# Convert random forest predictions to integers in order to use the 
# "prediction" function
rf261 <- as.integer(as.character(rf261_predict)) 
rf261_prediction_v_actual <- prediction(rf261, actual)

rf261_ROC_curve <- performance(rf261_prediction_v_actual, measure="tpr", 
x.measure="fpr")
plot(rf261_ROC_curve)

rf261_auc <- performance(rf261_prediction_v_actual, measure="auc")
rf261_auc@y.values[[1]] # auc of random forest with 261 trees is 0.8091018

randForest251 <- randomForest(Survived ~., data = new_training, nTree = 251)
rf251_predict <- predict(randForest251, new_test, type="class")
mean(actual_results == rf251_predict) # Accuracy = 0.8324022

# Convert random forest predictions to integers in order to use the
# "prediction" function
rf251 <- as.integer(as.character(rf251_predict))
rf251_prediction_v_actual <- prediction(rf251, actual)

rf251_ROC_curve <- performance(rf251_prediction_v_actual, measure="tpr",
x.measure="fpr")
plot(rf251_ROC_curve)

rf251_auc <- performance(rf251_prediction_v_actual, measure="auc")
rf251_auc@y.values[[1]] # auc random forest with 251 trees is 0.803105

# The accuracy of the random forest model with 261 trees is 0.8324022. 
# The AUC of the random forest with 261 trees is 0.8091018.
# The accuracy of the random forest model with 251 trees is 0.8324022. 
# The AUC of the random forest with 251 trees is 0.803105.



