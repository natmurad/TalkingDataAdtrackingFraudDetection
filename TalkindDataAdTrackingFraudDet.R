###
#title: "TalkingDataAdTrackingFraudDetectionv2"
#author: "Natália Faraj Murad"
#date: "03/07/2021"
###

## Prepare the Environment & Load the Data
setwd("~/TalkingData")

library(data.table)
library(tidyr)
library(dplyr)
library(knitr)
library(bigreadr)
library(ggplot2)
library(Matrix)
library(MASS)
library(pROC)
library(ROSE)
library(caret)
library(gmodels)
library(xgboost)
library(fastAdaboost)
library(lightgbm)

## Preprocessing data in chunks
# Define parameters
nlinesfile <- nlines("train.csv")
oldrows <- 1
chunksize <- 1000000

# Read the file in chunks & split classes in 2 different files
while(nlinesfile>oldrows){
  chunk <- read.csv2("train.csv", 
                     nrows=chunksize, skip = oldrows,
                     stringsAsFactors=FALSE, sep = ",", header =F,
                     na.strings = "")
  colnames(chunk) <- c("ip", "app", "device", "os", "channel",
                       "click_time", "attributed_time",
                       "is_attributed")
  chunk %>%
    filter(is_attributed==1) %>%
    # Write the file class 1
    fwrite("datafilt1.csv", append = TRUE, sep = ",",
           col.names = F)
  chunk %>%
    filter(is_attributed==0) %>%
    # Write the file class 0
    fwrite("datafilt0.csv", append = TRUE, sep = ",",
           col.names = F)
  # Update values
  oldrows <- oldrows + nrow(chunk)
  cat("Read: lines", oldrows, "to", oldrows + nrow(chunk), "\n")
  rm(chunk)
}

# Classes are unbalanced
sizeClass1 <-  nlines("datafilt1.csv")
sizeClass0 <- nlines("datafilt0.csv")

# Creating indexes to sample Class 0
indexes <- sample(2:sizeClass0, size = 5*sizeClass1, replace = FALSE)

# Read the files
class0 <- fread("datafilt0.csv", stringsAsFactors = F,sep = ",",
                na.strings = "")

# Create a sample
class0sample <- class0 %>%
  sample_n(2284230)

fwrite(class0sample, "class0sample.csv", append = TRUE, sep = ",",
       col.names = F)

data <- fread("databalanced.csv", stringsAsFactors = F, sep = ",", header =T, na.strings = "")

head(data)
dim(data)
str(data)


## Format the Dates
data$click_time <- as.POSIXct(data$click_time)

# Check NA values
sum(is.na(data$click_time))

# Splitting the click time and date
data$day <- as.numeric(sapply(data$click_time, wday))
data$hour <- as.numeric(sapply(data$click_time, hour))
data$min <- as.numeric(sapply(data$click_time, minute))

## Exploration

### Number of Unique Values in the Features
n_ip <- length(unique(data$ip))
n_app <- length(unique(data$app))
n_os <- length(unique(data$os))
n_ch <- length(unique(data$channel))
n_device <- length(unique(data$device))
n_days <- length(unique(data$day))

counts <- rbind(n_app, n_os, n_ch, n_device, n_days)

barplot(counts[,1], main = 'Exclusive Items', ylab = 'Number of Exclusive',
        col = c('#836FFF', '#1E90FF', '#FFD700', '#32CD32','#DC143C'))

message(c('The total number of ips is ', n_ip, '.\n',
          'The total number of apps is ', n_app, '.\n',
          'The total number of operational systems is ', n_os, '.\n',
          'The total number of channels is ', n_ch, '.\n',
          'The total number of devices is ', n_device, '.\n',
          'The total number of day is ', n_days, '.'))
rm(counts, n_ip, n_app, n_os, n_ch, n_device, n_days)
invisible(gc())

### Number of Clicks by IP
# Total number of clicks by IP
n_clicks_by_ip <- as.data.frame(table(data$ip))
summary(as.data.frame(n_clicks_by_ip))

#data1 <- merge(data1, n_clicks_by_ip, by.x = 'ip', by.y = 'Var1')
barplot(n_clicks_by_ip$Freq~n_clicks_by_ip$Var1, xlab = "IP",
        ylab = 'Number of Clicks',
        main = "Number of Clicks by IP", ylim = c(0,18000))

x <- data %>%
  group_by(is_attributed) %>%
  count(ip)

y <- as.data.frame(subset(x, is_attributed==1))
z <- as.data.frame(subset(x, is_attributed==0))

cat("Max clicks by IP - Not Download: ", max(z$n), "\n")
cat("Max clicks by IP - Download: ", max(y$n), "\n")

par(mfrow = c(1,2))
barplot(z$n~z$ip, ylim=c(0,18000), xlab = "IP",
        ylab = 'Number of Clicks',
        main = "Number of Clicks by IP- Not Download")
barplot(y$n~y$ip, ylim=c(0,18000), xlab = "IP",
        ylab = 'Number of Clicks',
        main = "Number of Clicks by IP - Download")
rm(n_clicks_by_ip, x, y)
invisible(gc())

### Percentage of Clicks by Hour

par(mfrow = c(1,1))

# Percent of clicks by hour
total_cliques <- data %>%
  group_by(is_attributed) %>%
  count()

clicksHdownload <- data %>%
  filter(is_attributed=='1') %>%
  group_by(hour) %>%
  mutate(percentclickDown = sum(hour)/456846)

clicksHNotdownload <- data %>%
  filter(is_attributed=='0') %>%
  group_by(hour) %>%
  mutate(percentclickNotDown = sum(hour)/2284230)

ggplot()+
  geom_step(aes(x = clicksHdownload$hour,
                y = clicksHdownload$percentclickDown, col = "Downloads"))+
  geom_step(aes(x = clicksHNotdownload$hour,
                y = clicksHNotdownload$percentclickNotDown, col = "Not Downloads"))+
  labs(x = "Hour", y = "Percentage of Clicks")+
  ggtitle("Percentage of Clicks by Hour") +
  theme(plot.title = element_text(hjust = 0.5))
rm(clicksHdownload, clicksHNotdownload, total_cliques)
invisible(gc())

## Feature Engineering
# Add some count vars
data <- data %>%
  add_count(ip, name = 'ip_ct') %>%
  add_count(ip, day, hour, name = 'ip_d_h') %>%
  add_count(ip, hour, channel, name = 'ip_h_ch') %>%
  add_count(ip, hour, os, name = 'ip_h_os') %>%
  add_count(ip, hour, app, name = 'ip_h_app') %>%
  add_count(ip, hour, device, name = 'ip_h_dev')

# How popular is the app or channel?
data <- data %>%
  add_count(app, channel, name = 'app_ch')

#Next click
next_click <- function(vector){
  nxt_click <- as.vector(0)
  for(i in 1:length(vector)){
    if(i < length(vector)){
      nxt_click[i] <- as.numeric(difftime(vector[i+1], vector[i], units = 'secs'))
    } else {
      nxt_click[i] <- 0
    }
    return(nxt_click[i])
  }
}

#next click by ip, os, device
data <- data %>%
  group_by(ip, os, device) %>%
  arrange(click_time) %>%
  mutate(ip_os_dev_nxcl = next_click(click_time))

# Excluding variables that will be not used
data$click_time <- NULL
data$ip <- NULL

## Split Train & Test Datasets
trainset <- filter(data, day<5)
testset <- filter(data, day == 5)

# Check proportion of each class
prop.table(table(trainset$is_attributed))
barplot(prop.table(table(trainset$is_attributed)),
        col = c('#6495ED', '#FF69B4'))

# Check na values
sum(is.na(trainset))

## Balancing Categories
balanced_sample = ovun.sample(is_attributed~., trainset, method="under",
                              p=0.25, subset=options("subset")$subset,
                              seed = 15)$data
dim(balanced_sample)

# Check proportion of each class
prop.table(table(balanced_sample$is_attributed))
barplot(prop.table(table(balanced_sample$is_attributed)),
        col = c('#6495ED', '#FF69B4'))

rm(trainset)
rm(data)
invisible(gc())

## XGBoost
# GridSearch in order to find best parameters

# Set up the cross-validated hyper-parameter search
xgb_grid_1 = expand.grid(
  nrounds = 100,
  max_depth = c(30, 31, 35, 36, 40, 41),
  eta = c(0.1, 0.5, 0.6, 0.7, 0.8, 0.9, 1),
  gamma = 0.001,
  colsample_bytree = 0.8,
  subsample = 1,
  min_child_weight = c(0,1,3,5)
)

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

factoris_attributed <- as.factor(ifelse(balanced_sample$is_attributed==0, "No", "Download"))

# train the model for each parameter combination in the grid,
#   using CV to evaluate
xgb_train_1 = train(
  x = as.matrix(balanced_sample[, impfeat]),
  y = factoris_attributed,
  trControl = xgb_trcontrol_1,
  tuneGrid = xgb_grid_1,
  method = "xgbTree",
  nthread = 50
)

### Result
#Aggregating results
#Selecting tuning parameters
#Fitting nrounds = 100, max_depth = 30, eta = 0.1, gamma = 0.001, colsample_bytree = 0.8, min_child_weight = 5, subsample = 1 on full training set

# Fit the Model

model_xgboost <- xgboost(
  data      = as.matrix(balanced_sample[, colnames(balanced_sample) != "is_attributed"]), 
  label     = as.matrix(balanced_sample$is_attributed),
  max.depth = 30,                            
  eta       = 0.1,                             
  gamma     = 0.001,
  colsample_bytree = 0.8,
  nthread   = 40, 
  min_child_weight = 5,
  subsample = 1,
  nrounds   = 100,                          
  objective = "binary:logistic",             
  verbose   = F                              
)

# Check importance of the features
imp.xgb <- xgb.importance(model = model_xgboost)
kable(imp.xgb)

# app, channel, ip_ct, ip_h_dev, ip_os_dev_nxcl, min, hour,
# os, app_ch, ip_d_h, ip_h_os, day, device, ip_h_app, ip_h_ch

# Predictions
test_features <- testset %>% dplyr::select(-c("is_attributed"))
pred <- predict(model_xgboost, as.matrix(test_features))

# Classification
cutoff <- 0.5
predClass <- ifelse(pred > cutoff, 1, 0)

# Confusion Matrix 
confusionMatrix(table(pred = predClass, data = testset$is_attributed))

# AUC
xgboostAuc <- auc(roc(as.integer(testset$is_attributed), as.integer(predClass)))


#plot ROC
roc(testset$is_attributed, pred, plot = TRUE, col = "steelblue", lwd = 1, 
    levels=base::levels(as.factor(testset$is_attributed)), grid=TRUE) 
rm(test_features)
invisible(gc())

## LightGBM
# Set categorical features
categorical_features = c("app", "device", "os", "channel", "day", "hour", "min")

train <- as.data.frame(balanced_sample)
train[categorical_features] <- apply(train[categorical_features],2, as.factor)

test <- as.data.frame(testset)
test[categorical_features] <- apply(test[categorical_features],2, as.factor)

# Prepare datasets
dtrain = lgb.Dataset(data = as.matrix(train[, colnames(train) != "is_attributed"]), 
                     label = train$is_attributed, categorical_feature = categorical_features)
dvalid = lgb.Dataset(data = as.matrix(test[, colnames(test) != "is_attributed"]), 
                     label = test$is_attributed, categorical_feature = categorical_features)

invisible(gc())

# Set parameters
params = list(objective = "binary", 
              metric = "auc", 
              learning_rate= 0.25, 
              num_leaves= 38,
              max_depth= 21,
              min_child_samples= 100,
              max_bin= 100,
              subsample= 0.7,
              subsample_freq= 1,
              colsample_bytree= 0.7,
              min_child_weight= 0,
              min_split_gain= 0)

# Train the model
model <- lgb.train(params, dtrain, valids = list(validation = dvalid),
                   nthread = 45, nrounds = 1000, verbose= 1,
                   early_stopping_rounds = 50, eval_freq = 50)

rm(dtrain, dvalid)
invisible(gc())

# AUC
cat("Best AUC: ", 
    max(unlist(model$record_evals[["validation"]][["auc"]][["eval"]])))

# Get feature importance
implgb = lgb.importance(model, percentage = TRUE)
kable(implgb)

val_preds = predict(model, data = as.matrix(testset[, colnames(testset) != "is_attributed"]), n = model$best_iter)

# Classification
cutoff <- 0.5
predLGBM <- ifelse(val_preds > cutoff, 1, 0)

# Confusion Matrix 
confusionMatrix(table(pred = predLGBM, data = testset$is_attributed))

# Plot ROC
roc(testset$is_attributed, val_preds, plot = TRUE, col = "steelblue",
    lwd = 1, levels=base::levels(as.factor(testset$is_attributed)),
    grid=TRUE)   

rm(balanced_sample, test, testset, train)
invisible(gc())

### Applying the Model to the Test Dataset
# Read the files
testdata <- fread("test.csv", stringsAsFactors = F,sep = ",",
                  na.strings = "")

# Creating features
testdata$click_time <- as.POSIXct(testdata$click_time)

# Splitting the click time and date
testdata <- testdata %>%
  mutate(day = wday(testdata$click_time)) %>%
  mutate(hour = hour(testdata$click_time)) %>%  
  mutate(min = minute(testdata$click_time))

# Add some count vars
testdata <- testdata %>%
  add_count(ip, name = 'ip_ct') %>%
  add_count(ip, day, hour, name = 'ip_d_h') %>%
  add_count(ip, hour, channel, name = 'ip_h_ch') %>%
  add_count(ip, hour, os, name = 'ip_h_os') %>%
  add_count(ip, hour, app, name = 'ip_h_app') %>%
  add_count(ip, hour, device, name = 'ip_h_dev')

# How popular is the app or channel?
testdata <- testdata %>%
  add_count(app, channel, name = 'app_ch')

#next click by ip, os, device
testdata <- testdata %>%
  group_by(ip, os, device) %>%
  arrange(click_time) %>%
  mutate(ip_os_dev_nxcl = next_click(click_time))

# Excluding variables that will be not used
testdata$click_time <- NULL
testdata$ip <- NULL

# Predictions
val_preds = predict(model, data = as.matrix(testdata[, colnames(testdata) != "click_id"]), n = model$best_iter)

# Classification
cutoff <- 0.5
is_attributed <- ifelse(val_preds > cutoff, 1, 0)

predictions <- cbind(testdata$click_id, is_attributed)
colnames(predictions) <- c('click_id', 'is_attributed')

# Writing a file with preds
fwrite(predictions, 'predictionsLGBM.csv', sep = ',',
       col.names = TRUE)