# Author: John Sokol
# Machine Learning in R
# 13 February 2018

# Perceptron neural network implementation

# Algorithm steps: 
# PART 1
# 1. For every input, mulitply the input by its corresponding weight
# 2. Sum all of the weighted inputs 
# 3. Compute the output of the perceptron based on the sum passed 
#    through an activation function (sign())
#  PART 2 
# 1. Provide the perceptron with inputs for which there is a known answer 
# 2. Ask the perceptron to guess an answer 
# 3. Adjust all weights according to the error 
# 4. Return to step 1 and repeat 

# setwd("./Desktop")

# import dataset
clean_data <- read.csv("CleanData.csv", sep = ",")
not_clean <- read.csv("percep_data.csv", sep = ",")

# divides complete dataset into test and train datasets
indexes <- sample(1:nrow(not_clean), size=0.8*nrow(not_clean))
train <- not_clean[indexes,]
test <- not_clean[-indexes,]

# returns a colored scatter plot of the labeled training data to observe linear separability 
# prior to using the perceptron function. Accepts dataset of form x[1] = data_x1, x[2] = data_x2, x[3] = label
init_percep_plot <- function(percep_data_plot) {
  
    # iterate over all rows of the dataset
    for (i in 1:nrow(percep_data_plot)) {
    
        # base R plot col argument does not accept values < 0, so change 
        # label values of -1 to 2
        if (percep_data_plot[i,3] == -1) {
            percep_data_plot[i,3] = 2
        }
    }
    plot(percep_data_plot[,1], percep_data_plot[,2], col = percep_data_plot[,3], xlab = "X1",
    ylab = "X2", main = "Train Dataset with Known Labels")
}

# returns the following; callable using the $ operator when function is instantiated 
#     1. weights: callable via list indexes, i.e weightX1 = weights[1] 
#     2. bias 
#     3. true_count_percentage: correct label values / total number of elements in dataset. 1 = all correct
#     4. prediction: returns list of label prediction values

# function reaches convergence once prediction accuracy is > 95% compared to known training labels
# if function does NOT reach convergence after 5000 while loop iterations, the function stops; the weights at 
# that moment must be used for test data prediction and visualizations. 
# Accepts train dataset of form x[1] = data_x1, x[2] = data_x2, x[3] = label
percep_function <- function (train) {
  
    # initialize weight vectors and bias = 0
    # bias will be corrected assuming separator does NOT pass through origin
    weightX1 <- 0
    weightX2 <- 0
    bias     <- 0
  
    # initialize prediction vector 
    prediction <- vector("integer", nrow(train))
  
    # initialize boolean value = FALSE to enter while loop
    all_classified <- FALSE
    
    # initialize number of while loop iterations to sufficiently train weights
    epochs <- 0
    
    while (!all_classified) {
    
        # after appropriate amount of iterations all_classified should remain TRUE to exit while loop
        all_classified <- TRUE
        
        # loop iterates over all rows in input dataset
        for (i in 1:nrow(train)) {
            
            # initial prediction values take into consideration bias, multiplication 
            # of weights and x[1], x[2] values
            prediction[i] <- sign((bias + (weightX1 * train[i,1]) + (weightX2 * train[i,2])))
      
            # conditional to modify weights if there is a mislabel
            if (train[i,3] != prediction[i]) {
                
                # calculcate error (correct label - predicted)
                error <- (train[i,3] - prediction[i])
                
                # calculate new weights and bias
                weightX1 <- weightX1 +  (error * train[i,1])
                weightX2 <- weightX2 +  (error * train[i,2])
                bias <- bias + (error)
                
                # looping through if statement indicates not all predictions are correct; weights 
                # must still be updated 
                all_classified <- FALSE
            }
        }
        
        # verify predicted label values align with true labeled values in dataset
        z <- train$Y == prediction
        true_count <- table(z)["TRUE"]
        true_count_percentage <- true_count / nrow(train)
        
        if (true_count_percentage > 0.94) {
            all_classified <- TRUE
        }
        
        # update while loop iterator; takes ~ 22 epochs to reach convergence for not_clean data
        epochs <- epochs + 1
        
        # Perceptron Classifier has NOT reached convergence (> 95% label accuracy)
        if (epochs > 2000) {
            all_classified <- TRUE
        }
    } 
    
    # stores weights into vector 
    weights <- c(weightX1, weightX2)
    
    return(list(weights = weights, bias = bias, true_count_percentage = true_count_percentage, prediction = prediction, epochs = epochs))
}

# instantiates perceptron function for plotting in following function
instan_percep <- percep_function(train)

# returns a colored scatter plot of the predicted labels with original percep_data values
# also depicts linear separability with abline()
# use original dataset along with instantiated precep_function as function arguments
final_percep_plot <- function(train, instan_percep) {
    
    # iterates over length of prediction vector 
    for (i in 1:length(instan_percep$prediction)) {
      
        # base R plot col argument does not accept values < 0, so change 
        # -1 labels to 2
        if (instan_percep$prediction[i] == -1) {
            instan_percep$prediction[i] = 2
        }
    }
    # plots colored scatter plot 
    plot(train$X1, train$X2, col = instan_percep$prediction, xlab = "X1", ylab = "X2", 
    main = "Train Dataset Depicted with Weighted Predictions")
    
    # slope intercept form is defined as y = mx + b 
    # standard form is Ax + By = C or W1x + W2y = C
    # using algebra: A = m = -(W1 / W2) and b = -(bias / W2)
    b <- -(instan_percep$bias / instan_percep$weights[2])
    m <- -(instan_percep$weights[1] / instan_percep$weights[2])
    
    # plot linear data separator
    abline(b,m)
}

# utilizes the weights established in percep_function() to run over test data
# Accepts test dataset of form x[1] = data_x1, x[2] = data_x2, x[3] = label
percep_test <- function(test_data, instan_percep) {
  
    # initialize prediction vector 
    test_prediction <- vector("integer", nrow(test_data))
  
    # test prediction values take into consideration percep_function bias, multiplication 
    # of weights and x[1], x[2] values; weights DO NOT change
    for (i in 1:nrow(test_data)) {
        test_prediction[i] <- sign((instan_percep$bias + (instan_percep$weights[1] * test_data[i,1]) + 
        (instan_percep$weights[2] * test_data[i,2])))
    }
    
    # iterates over length of test prediction vector 
    for (i in 1:length(test_prediction)) {
      
        # base R plot col argument does not accept values < 0, so change 
        # -1 labels to 2
        if (test_prediction[i] == -1) {
            test_prediction[i] = 2
        }
    }
    
    z_test <- test$Y == test_prediction
    true_count <- table(z_test)["TRUE"]
    true_count_percentage <- true_count / nrow(test)
    
    # establishes y intercept and slope values; see final_percep_plot() for details
    b <- -(instan_percep$bias / instan_percep$weights[2])
    m <- -(instan_percep$weights[1] / instan_percep$weights[2])
    
    plot(test_data$X1, test_data$X2, col = test_prediction, xlab = "X1", ylab = "X2", 
    main = "Test Dataset with Learned Weights")
    abline(b,m)
    return(list(true_count_percentage = true_count_percentage))
}
