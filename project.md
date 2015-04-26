---
title: "Practical ML project"
author: "Oskar Jarczyk"
date: "Sunday, April 26, 2015"
output: html_document
---

### Loading neccesary libraries


```r
library(doParallel)
library(knitr)
library(caret)
library(randomForest)
library(ggplot2)
library(reshape2)
```

### Setting up working directory and register parallel


```r
setwd("/mnt/data1/oskar/pml/")
registerDoParallel(detectCores())
# knitr sometimes refuses to co-work with Parallel package
# but works with instruction knit2html()
```



### Reading data


```r
ultima <- read.csv("/mnt/data1/oskar/pml/pml-testing.csv",
                   header = T,
                   na.strings = "NA", row.names = 1)
dataset <- read.csv("/mnt/data1/oskar/pml/pml-training.csv",
                   header = T,
                   na.strings = c("NA","#DIV/0!"),
                   row.names = 1)
```

### Cleaning data


```r
# data is dirty, cleaning it here and fixing col types
dataset$kurtosis_roll_belt = as.numeric(dataset$kurtosis_roll_belt)
dataset$kurtosis_picth_dumbbell = as.numeric(dataset$kurtosis_picth_dumbbell)
dataset$skewness_roll_dumbbell = as.numeric(dataset$skewness_roll_dumbbell)
dataset$skewness_pitch_dumbbell = as.numeric(dataset$skewness_pitch_dumbbell)
dataset$max_yaw_dumbbell = as.numeric(dataset$max_yaw_dumbbell)
dataset$min_yaw_dumbbell = as.numeric(dataset$min_yaw_dumbbell)
# remove unwanted columns, too obvious or useless
p1 <- qplot(dataset$user_name, geom="histogram",
            fill=I("blue"), col=I("red"))
p2 <- qplot(dataset$cvtd_timestamp, geom="histogram",
            fill=I("grey"), col=I("red"))
p3 <- qplot(dataset$new_window, geom="histogram")
p4 <- qplot(dataset$total_accel_belt, geom="histogram",
            fill=I("blue"), col=I("red"))
multiplot(p1, p2, p3, p4, cols=2)
```

```
## stat_bin: binwidth defaulted to range/30. Use 'binwidth = x' to adjust this.
```

![plot of chunk clean_data](figure/clean_data-1.png) 

### Removing unwanted columns, which are too obvious and may cause overfitting


```r
dataset <- dataset[,-which(names(dataset) %in% c(
   "user_name","cvtd_timestamp", "new_window", "num_window"))]
```

### Partitioning data for train and test cases, p=0.75 by rule of thumb


```r
testIndex = createDataPartition(dataset$classe, p = 3/4)[[1]]
testing = dataset[-testIndex,]
training = dataset[testIndex,]
```

### Data cleaning part 2


```r
# remove columns which have only NA in all rows
testing <- testing[,colSums(is.na(testing))<nrow(testing)]
training <- training[,colSums(is.na(training))<nrow(training)]
# remove zero variance columns
zero_v_train <-
   names(training[, sapply(
      training, function(v) var(v, na.rm=TRUE)==0)])
training <- training[,-which(names(training) %in% zero_v_train)]
testing <- testing[,-which(names(testing) %in% zero_v_train)]
```

### Principal component analysis


```r
cls_indx = which(names(training) %in% c("classe"))
set.seed(31337)
preProc <- preProcess(training[,-cls_indx], method="pca", thresh=0.85)
preProc
```

```
## 
## Call:
## preProcess.default(x = training[, -cls_indx], method = "pca", thresh
##  = 0.85)
## 
## Created from 161 samples and 145 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 27 components to capture 85 percent of the variance
```

```r
# delete columns with at least 1 or more NA value
training <- training[, colSums(is.na(training)) == 0]
testing <- testing[, colSums(is.na(testing)) == 0]
# check again PCA with threshold 0.85
cls_indx = which(names(training) %in% c("classe"))
set.seed(31337)
preProc <- preProcess(training[,-cls_indx], method="pca", thresh=0.85)
preProc
```

```
## 
## Call:
## preProcess.default(x = training[, -cls_indx], method = "pca", thresh
##  = 0.85)
## 
## Created from 14718 samples and 54 variables
## Pre-processing: principal component signal extraction, scaled, centered 
## 
## PCA needed 17 components to capture 85 percent of the variance
```

### Check different models below


```r
variable.group = colnames(training)
melted <- melt(preProc$rotation[,1:9]) # cbind(variable.group, 
barplot <- ggplot(data=melted) +
  geom_bar(aes(x=Var1, y=value, fill=variable.group), stat="identity") +
  facet_wrap(~Var2)
barplot
```

```
## Error: Aesthetics must either be length one, or the same length as the dataProblems:variable.group
```

### Check models on PCA-selected features only

#### Naive Bayes

```r
set.seed(31337)
modelFit <- train(classe ~ ., method="nb", data=training,
                  trControl = trainControl(
                     preProcOptions = list(thresh = 0.85)
                  ),
                  preProcess = "pca")
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 860  82 256 163  34
##          B  76 579 140  67  87
##          C 182  83 523  34  33
##          D  62  67 132 480  63
##          E  61 144  74  73 549
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6099          
##                  95% CI : (0.5961, 0.6236)
##     No Information Rate : 0.2531          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5089          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6930   0.6063   0.4649  0.58752   0.7167
## Specificity            0.8539   0.9063   0.9121  0.92072   0.9149
## Pos Pred Value         0.6165   0.6101   0.6117  0.59701   0.6093
## Neg Pred Value         0.8914   0.9049   0.8513  0.91780   0.9458
## Prevalence             0.2531   0.1947   0.2294  0.16660   0.1562
## Detection Rate         0.1754   0.1181   0.1066  0.09788   0.1119
## Detection Prevalence   0.2845   0.1935   0.1743  0.16395   0.1837
## Balanced Accuracy      0.7735   0.7563   0.6885  0.75412   0.8158
```

#### Generalized Linear Model
##### Because 'classe' is a field suggesting a multi-label classification problem, we shouldn't really use glm which works best on binary classes {0,1}

```r
# set.seed(31337)
# modelFit <- train(classe ~ ., method="glm", data=training,
#                   trControl = trainControl(
#                      preProcOptions = list(thresh = 0.85)
#                   ),
#                   preProcess = "pca")
# confusionMatrix(testing$classe, predict(modelFit,testing))
```

#### Linear Discriminant Analysis

```r
set.seed(31337)
modelFit <- train(classe ~ ., method="lda",
                  preProcess="pca",
                  trControl = trainControl(
                     preProcOptions = list(thresh = 0.85)
                  ),
                  data=training)
# took around 2 minutes
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 946 101 127 181  40
##          B 191 417 119 130  92
##          C 352 100 299  53  51
##          D  94 171  82 375  82
##          E 147 147  91  98 418
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5006          
##                  95% CI : (0.4865, 0.5147)
##     No Information Rate : 0.3528          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3627          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5468  0.44551  0.41643  0.44803  0.61201
## Specificity            0.8585  0.86593  0.86718  0.89452  0.88557
## Pos Pred Value         0.6781  0.43941  0.34971  0.46642  0.46393
## Neg Pred Value         0.7766  0.86877  0.89652  0.88732  0.93380
## Prevalence             0.3528  0.19086  0.14641  0.17068  0.13927
## Detection Rate         0.1929  0.08503  0.06097  0.07647  0.08524
## Detection Prevalence   0.2845  0.19352  0.17435  0.16395  0.18373
## Balanced Accuracy      0.7027  0.65572  0.64181  0.67127  0.74879
```

#### k-Nearest Neighbors


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="knn", 
                  trControl = trainControl(
                    method="cv", number = 5,
                    preProcOptions = list(thresh = 0.85)
                  ),
                  preProcess="pca",
                  data=training)
# took around 7-8 minutes
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1354   12   15   11    3
##          B   22  900   24    3    0
##          C   10   16  807   17    5
##          D    8    4   40  748    4
##          E    5    8    9    6  873
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9547          
##                  95% CI : (0.9485, 0.9604)
##     No Information Rate : 0.2853          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9427          
##  Mcnemar's Test P-Value : 0.004504        
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9678   0.9574   0.9017   0.9529   0.9864
## Specificity            0.9883   0.9876   0.9880   0.9864   0.9930
## Pos Pred Value         0.9706   0.9484   0.9439   0.9303   0.9689
## Neg Pred Value         0.9872   0.9899   0.9783   0.9910   0.9970
## Prevalence             0.2853   0.1917   0.1825   0.1601   0.1805
## Detection Rate         0.2761   0.1835   0.1646   0.1525   0.1780
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9781   0.9725   0.9449   0.9696   0.9897
```

##### From now on, all models with high accuracy will get k-fold cross validated
##### Hence the trainControl with method="cv" instructions in algorithms producing > 0.75 accuracy

##### High accuracy, printing possible solutions


```r
answers1 <- predict(modelFit, ultima)
# answers produced from k-nn model with PCA
```

#### Random Forests


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="rf", 
                  trControl = trainControl(
                    method = "oob",
                    preProcOptions = list(thresh = 0.85)
                  ),
                  preProcess="pca",
                  data=training)
# took around 7-8 minutes
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1376    4    5    8    2
##          B   14  926    8    0    1
##          C    4   15  832    2    2
##          D    7    7   35  752    3
##          E    0   11    5    4  881
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9721          
##                  95% CI : (0.9671, 0.9765)
##     No Information Rate : 0.2857          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9647          
##  Mcnemar's Test P-Value : 2.002e-08       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9822   0.9616   0.9401   0.9817   0.9910
## Specificity            0.9946   0.9942   0.9943   0.9874   0.9950
## Pos Pred Value         0.9864   0.9758   0.9731   0.9353   0.9778
## Neg Pred Value         0.9929   0.9906   0.9869   0.9966   0.9980
## Prevalence             0.2857   0.1964   0.1805   0.1562   0.1813
## Detection Rate         0.2806   0.1888   0.1697   0.1533   0.1796
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9884   0.9779   0.9672   0.9846   0.9930
```

```r
summary(modelFit$err.rate)
```

```
## Length  Class   Mode 
##      0   NULL   NULL
```

##### Random forest is an exception described in literature, and for validation it uses 'oob' ("out of bounds") method

##### High accuracy, printing possible solutions


```r
answers2 <- predict(modelFit, ultima)
# answers produced from Random Forest with PCA
```

#### Multinomial


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="multinom",
                  trControl = trainControl(
                     preProcOptions = list(thresh = 0.85)
                  ),
                  preProcess="pca", data=training)
```

```
## # weights:  95 (72 variable)
## initial  value 23687.707195 
## iter  10 value 21574.031616
## iter  20 value 21477.168569
## iter  30 value 20151.034153
## iter  40 value 19703.260371
## iter  50 value 19083.679481
## iter  60 value 19019.347996
## iter  70 value 18979.468143
## iter  80 value 18962.890994
## final  value 18962.837988 
## converged
```

```r
# and that took around 5-7 minutes..
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 964  69 125 183  54
##          B 188 387 117 151 106
##          C 344  82 325  54  50
##          D 102 140  77 392  93
##          E 137 136 118 103 407
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5047          
##                  95% CI : (0.4906, 0.5188)
##     No Information Rate : 0.3538          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.3682          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5556  0.47543  0.42651  0.44394  0.57324
## Specificity            0.8640  0.86259  0.87204  0.89754  0.88221
## Pos Pred Value         0.6910  0.40780  0.38012  0.48756  0.45172
## Neg Pred Value         0.7803  0.89204  0.89207  0.88024  0.92431
## Prevalence             0.3538  0.16599  0.15538  0.18006  0.14478
## Detection Rate         0.1966  0.07892  0.06627  0.07993  0.08299
## Detection Prevalence   0.2845  0.19352  0.17435  0.16395  0.18373
## Balanced Accuracy      0.7098  0.66901  0.64928  0.67074  0.72773
```

#### Stochastic Gradient Boosting


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="gbm",
                  trControl = trainControl(
                    method="cv", number = 5,
                    preProcOptions = list(thresh = 0.85)
                  ),
                  preProcess="pca", data=training)
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.1040
##      2        1.5419            -nan     0.1000    0.0779
##      3        1.4921            -nan     0.1000    0.0605
##      4        1.4526            -nan     0.1000    0.0601
##      5        1.4147            -nan     0.1000    0.0488
##      6        1.3824            -nan     0.1000    0.0448
##      7        1.3533            -nan     0.1000    0.0447
##      8        1.3251            -nan     0.1000    0.0384
##      9        1.3008            -nan     0.1000    0.0331
##     10        1.2798            -nan     0.1000    0.0281
##     20        1.1354            -nan     0.1000    0.0148
##     40        0.9890            -nan     0.1000    0.0102
##     60        0.8898            -nan     0.1000    0.0084
##     80        0.8179            -nan     0.1000    0.0062
##    100        0.7566            -nan     0.1000    0.0019
##    120        0.7095            -nan     0.1000    0.0017
##    140        0.6653            -nan     0.1000    0.0033
##    150        0.6468            -nan     0.1000    0.0020
```

```r
# and that took around 5-7 minutes..
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1176   41   75   95    8
##          B   86  717   79   32   35
##          C   90   72  642   32   19
##          D   51   29   88  605   31
##          E   60   75   49   33  684
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7798          
##                  95% CI : (0.7679, 0.7913)
##     No Information Rate : 0.2983          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.721           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8038   0.7677   0.6881   0.7591   0.8803
## Specificity            0.9364   0.9416   0.9464   0.9515   0.9474
## Pos Pred Value         0.8430   0.7555   0.7509   0.7525   0.7592
## Neg Pred Value         0.9182   0.9451   0.9281   0.9532   0.9768
## Prevalence             0.2983   0.1905   0.1903   0.1625   0.1584
## Detection Rate         0.2398   0.1462   0.1309   0.1234   0.1395
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8701   0.8546   0.8172   0.8553   0.9139
```

##### High accuracy, printing possible solutions


```r
answers3 <- predict(modelFit, ultima)
# answers produced from Stochastic Gradient Boosting with PCA
```

#### Check models on sensory-data only

##### I'll take raw sensor data, which are, according to paper


```r
colNamesRD <- c("accel_dumbbell_x", "accel_dumbbell_y", "accel_dumbbell_z",
                "magnet_dumbbell_x", "magnet_dumbbell_y", "magnet_dumbbell_z",
                "magnet_arm_x", "magnet_arm_y", "magnet_arm_z",
                "accel_arm_x", "accel_arm_y", "accel_arm_z",
                "magnet_belt_x", "magnet_belt_y", "magnet_belt_z",
                "accel_belt_x", "accel_belt_y", "accel_belt_z",
                "classe")
```

#### Naive Bayes


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="nb",
                  data=training[,colNamesRD])
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1029   20  145  178   23
##          B  121  528  146  126   28
##          C   93   62  611   81    8
##          D   84   20  115  536   49
##          E   43  116   74  105  563
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6662          
##                  95% CI : (0.6528, 0.6794)
##     No Information Rate : 0.2794          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.579           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7511   0.7078   0.5600   0.5224   0.8390
## Specificity            0.8964   0.8987   0.9360   0.9309   0.9202
## Pos Pred Value         0.7376   0.5564   0.7146   0.6667   0.6249
## Neg Pred Value         0.9028   0.9449   0.8815   0.8805   0.9730
## Prevalence             0.2794   0.1521   0.2225   0.2092   0.1368
## Detection Rate         0.2098   0.1077   0.1246   0.1093   0.1148
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8238   0.8033   0.7480   0.7267   0.8796
```

#### GLM (Generalized Linear Model)


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="glm",
                  data=training[,colNamesRD])
```

```
## Error in train.default(x, y, weights = w, ...): final tuning parameters could not be determined
```

```r
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1029   20  145  178   23
##          B  121  528  146  126   28
##          C   93   62  611   81    8
##          D   84   20  115  536   49
##          E   43  116   74  105  563
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6662          
##                  95% CI : (0.6528, 0.6794)
##     No Information Rate : 0.2794          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.579           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7511   0.7078   0.5600   0.5224   0.8390
## Specificity            0.8964   0.8987   0.9360   0.9309   0.9202
## Pos Pred Value         0.7376   0.5564   0.7146   0.6667   0.6249
## Neg Pred Value         0.9028   0.9449   0.8815   0.8805   0.9730
## Prevalence             0.2794   0.1521   0.2225   0.2092   0.1368
## Detection Rate         0.2098   0.1077   0.1246   0.1093   0.1148
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8238   0.8033   0.7480   0.7267   0.8796
```

#### LDA (Linear Discriminant Analysis)


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="lda",
                  data=training[,colNamesRD])
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1079   59  104  145    8
##          B  185  570  116   42   36
##          C  253   88  351  149   14
##          D  117   65  112  441   69
##          E  107  181   70  155  388
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5769          
##                  95% CI : (0.5629, 0.5908)
##     No Information Rate : 0.355           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4602          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6198   0.5919  0.46614  0.47318  0.75340
## Specificity            0.9001   0.9038  0.87858  0.90861  0.88312
## Pos Pred Value         0.7735   0.6006  0.41053  0.54851  0.43063
## Neg Pred Value         0.8113   0.9006  0.90072  0.88024  0.96827
## Prevalence             0.3550   0.1964  0.15355  0.19005  0.10502
## Detection Rate         0.2200   0.1162  0.07157  0.08993  0.07912
## Detection Prevalence   0.2845   0.1935  0.17435  0.16395  0.18373
## Balanced Accuracy      0.7599   0.7479  0.67236  0.69089  0.81826
```

#### k-Nearest Neighbors


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="knn",
                  trControl = trainControl(
                    method="cv", number = 5
                  ),
                  data=training[,colNamesRD])
# took around 7-8 minutes
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1309   22   27   34    3
##          B   76  773   48   27   25
##          C   11   27  784   22   11
##          D   13    8   69  705    9
##          E   14   36   32   23  796
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8905          
##                  95% CI : (0.8814, 0.8991)
##     No Information Rate : 0.2902          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8615          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9199   0.8926   0.8167   0.8693   0.9431
## Specificity            0.9753   0.9564   0.9820   0.9758   0.9741
## Pos Pred Value         0.9384   0.8145   0.9170   0.8769   0.8835
## Neg Pred Value         0.9675   0.9765   0.9565   0.9741   0.9880
## Prevalence             0.2902   0.1766   0.1958   0.1654   0.1721
## Detection Rate         0.2669   0.1576   0.1599   0.1438   0.1623
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9476   0.9245   0.8993   0.9226   0.9586
```

##### High accuracy, printing possible solutions


```r
answers4 <- predict(modelFit, ultima)
# answers produced from K-NN model with sensor-only data
```

#### Random Forests


```r
set.seed(31337)
modelFit <- randomForest(classe ~ ., data=training[,colNamesRD],
                         importance = T, nodeSize = 10,
                         mtry=5)
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1367    8    9   10    1
##          B   28  893   22    4    2
##          C    9   16  818   11    1
##          D    3    7   44  746    4
##          E    2    6    8   11  874
## 
## Overall Statistics
##                                          
##                Accuracy : 0.958          
##                  95% CI : (0.952, 0.9634)
##     No Information Rate : 0.2873         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9469         
##  Mcnemar's Test P-Value : 7.688e-07      
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9702   0.9602   0.9079   0.9540   0.9909
## Specificity            0.9920   0.9859   0.9908   0.9859   0.9933
## Pos Pred Value         0.9799   0.9410   0.9567   0.9279   0.9700
## Neg Pred Value         0.9880   0.9906   0.9795   0.9912   0.9980
## Prevalence             0.2873   0.1896   0.1837   0.1595   0.1799
## Detection Rate         0.2788   0.1821   0.1668   0.1521   0.1782
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9811   0.9731   0.9493   0.9699   0.9921
```

```r
summary(modelFit$err.rate) # randomForest(...) checks oob by default
```

```
##       OOB                A                 B                 C          
##  Min.   :0.03927   Min.   :0.01434   Min.   :0.06390   Min.   :0.03701  
##  1st Qu.:0.04022   1st Qu.:0.01601   1st Qu.:0.06671   1st Qu.:0.03857  
##  Median :0.04070   Median :0.01673   Median :0.06742   Median :0.03935  
##  Mean   :0.04469   Mean   :0.01901   Mean   :0.07233   Mean   :0.04506  
##  3rd Qu.:0.04303   3rd Qu.:0.01768   3rd Qu.:0.07128   3rd Qu.:0.04207  
##  Max.   :0.17199   Max.   :0.11466   Max.   :0.20367   Max.   :0.20502  
##        D                 E          
##  Min.   :0.06095   Min.   :0.02772  
##  1st Qu.:0.06260   1st Qu.:0.02919  
##  Median :0.06468   Median :0.03104  
##  Mean   :0.06835   Mean   :0.03387  
##  3rd Qu.:0.06841   3rd Qu.:0.03326  
##  Max.   :0.21714   Max.   :0.16139
```

##### High accuracy, printing possible solutions


```r
answers5 <- predict(modelFit, ultima)
# answers produced from Random Forest with sensor-only data
```

#### Multinom


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="multinom", data=training[,colNamesRD])
```

```
## # weights:  100 (76 variable)
## initial  value 23687.707195 
## iter  10 value 21169.976100
## iter  20 value 19598.127823
## iter  30 value 18984.995415
## iter  40 value 18604.394071
## iter  50 value 18534.842193
## iter  60 value 18530.201891
## iter  70 value 18526.124418
## iter  80 value 15838.425710
## iter  90 value 15783.696813
## final  value 15783.572589 
## converged
```

```r
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1075   71  113  106   30
##          B  157  553  126   33   80
##          C  183   89  436  120   27
##          D  104   65  135  408   92
##          E  131  147   62  136  425
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5907          
##                  95% CI : (0.5768, 0.6045)
##     No Information Rate : 0.3365          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.4789          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.6515   0.5978  0.50000   0.5081  0.64985
## Specificity            0.9017   0.9005  0.89608   0.9034  0.88800
## Pos Pred Value         0.7706   0.5827  0.50994   0.5075  0.47170
## Neg Pred Value         0.8361   0.9059  0.89232   0.9037  0.94279
## Prevalence             0.3365   0.1886  0.17781   0.1637  0.13336
## Detection Rate         0.2192   0.1128  0.08891   0.0832  0.08666
## Detection Prevalence   0.2845   0.1935  0.17435   0.1639  0.18373
## Balanced Accuracy      0.7766   0.7492  0.69804   0.7058  0.76892
```

#### GBM


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="gbm",
                  trControl = trainControl(
                    method="cv", number = 5
                  ),
                  data=training[,colNamesRD])
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.1598
##      2        1.5075            -nan     0.1000    0.1218
##      3        1.4322            -nan     0.1000    0.0970
##      4        1.3724            -nan     0.1000    0.0878
##      5        1.3178            -nan     0.1000    0.0660
##      6        1.2760            -nan     0.1000    0.0530
##      7        1.2421            -nan     0.1000    0.0587
##      8        1.2060            -nan     0.1000    0.0488
##      9        1.1754            -nan     0.1000    0.0462
##     10        1.1471            -nan     0.1000    0.0411
##     20        0.9426            -nan     0.1000    0.0231
##     40        0.7429            -nan     0.1000    0.0113
##     60        0.6333            -nan     0.1000    0.0070
##     80        0.5513            -nan     0.1000    0.0046
##    100        0.4961            -nan     0.1000    0.0033
##    120        0.4481            -nan     0.1000    0.0020
##    140        0.4051            -nan     0.1000    0.0021
##    150        0.3883            -nan     0.1000    0.0013
```

```r
confusionMatrix(testing$classe, predict(modelFit,testing[,colNamesRD]))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1323   14   27   25    6
##          B   76  768   58   27   20
##          C   32   42  748   26    7
##          D   24   15   87  659   19
##          E   18   34   26   42  781
## 
## Overall Statistics
##                                           
##                Accuracy : 0.8726          
##                  95% CI : (0.8629, 0.8818)
##     No Information Rate : 0.3004          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.8385          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8982   0.8797   0.7907   0.8460   0.9376
## Specificity            0.9790   0.9551   0.9730   0.9648   0.9705
## Pos Pred Value         0.9484   0.8093   0.8749   0.8197   0.8668
## Neg Pred Value         0.9573   0.9735   0.9511   0.9707   0.9870
## Prevalence             0.3004   0.1780   0.1929   0.1588   0.1699
## Detection Rate         0.2698   0.1566   0.1525   0.1344   0.1593
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9386   0.9174   0.8818   0.9054   0.9540
```

##### High accuracy, printing possible solutions


```r
answers6 <- predict(modelFit, ultima)
# answers produced from Stochastic Gradient Boosting with sensor-only data
```

### Check models on all 55 features

#### Naive Bayes

```r
set.seed(31337)
modelFit <- train(classe ~ ., method="nb",
                  trControl = trainControl(
                    method="cv", number = 5
                  ),
                  data=training)
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1030   49  142  160   14
##          B   70  682  137   50   10
##          C   29   65  715   41    5
##          D   57    4  150  551   42
##          E   30   66   48   29  728
## 
## Overall Statistics
##                                           
##                Accuracy : 0.7557          
##                  95% CI : (0.7434, 0.7677)
##     No Information Rate : 0.248           
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.6928          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.8470   0.7875   0.5998   0.6631   0.9111
## Specificity            0.9010   0.9339   0.9623   0.9379   0.9579
## Pos Pred Value         0.7384   0.7187   0.8363   0.6853   0.8080
## Neg Pred Value         0.9470   0.9535   0.8822   0.9317   0.9823
## Prevalence             0.2480   0.1766   0.2431   0.1695   0.1629
## Detection Rate         0.2100   0.1391   0.1458   0.1124   0.1485
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.8740   0.8607   0.7811   0.8005   0.9345
```

##### High accuracy, printing possible solutions


```r
answers7 <- predict(modelFit, ultima)
# answers produced from Naive Bayes with all major columns
```

#### k-Nearest Neighbors


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="knn", data=training)
# took around 7-8 minutes
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction   A   B   C   D   E
##          A 643 192 258 151 151
##          B 315 164 155 155 160
##          C 300  87 214 129 125
##          D 179 110 173 191 151
##          E 239 150 144 154 214
## 
## Overall Statistics
##                                           
##                Accuracy : 0.2908          
##                  95% CI : (0.2781, 0.3037)
##     No Information Rate : 0.3418          
##     P-Value [Acc > NIR] : 1               
##                                           
##                   Kappa : 0.097           
##  Mcnemar's Test P-Value : 5.147e-15       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.3837  0.23329  0.22669  0.24487  0.26717
## Specificity            0.7670  0.81314  0.83813  0.85136  0.83256
## Pos Pred Value         0.4609  0.17281  0.25029  0.23756  0.23751
## Neg Pred Value         0.7056  0.86372  0.81971  0.85634  0.85336
## Prevalence             0.3418  0.14335  0.19250  0.15905  0.16334
## Detection Rate         0.1311  0.03344  0.04364  0.03895  0.04364
## Detection Prevalence   0.2845  0.19352  0.17435  0.16395  0.18373
## Balanced Accuracy      0.5753  0.52321  0.53241  0.54811  0.54986
```

#### Random Forests


```r
set.seed(31337)
modelFit <- randomForest(classe ~ ., data=training,
                         importance = T, nodeSize = 10,
                         mtry=5)
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    0    0    0    0
##          B    0  948    1    0    0
##          C    0    3  851    1    0
##          D    0    0    9  794    1
##          E    0    0    0    2  899
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9965         
##                  95% CI : (0.9945, 0.998)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9956         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9968   0.9884   0.9962   0.9989
## Specificity            1.0000   0.9997   0.9990   0.9976   0.9995
## Pos Pred Value         1.0000   0.9989   0.9953   0.9876   0.9978
## Neg Pred Value         1.0000   0.9992   0.9975   0.9993   0.9998
## Prevalence             0.2845   0.1939   0.1756   0.1625   0.1835
## Detection Rate         0.2845   0.1933   0.1735   0.1619   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      1.0000   0.9983   0.9937   0.9969   0.9992
```

```r
summary(modelFit$err.rate)
```

```
##       OOB                 A                  B           
##  Min.   :0.002106   Min.   :0.000239   Min.   :0.001756  
##  1st Qu.:0.002242   1st Qu.:0.000239   1st Qu.:0.002458  
##  Median :0.002446   Median :0.000239   Median :0.002809  
##  Mean   :0.004294   Mean   :0.001079   Mean   :0.005628  
##  3rd Qu.:0.002650   3rd Qu.:0.000239   3rd Qu.:0.003160  
##  Max.   :0.097896   Max.   :0.049981   Max.   :0.154262  
##        C                  D                  E           
##  Min.   :0.002727   Min.   :0.004146   Min.   :0.001109  
##  1st Qu.:0.003506   1st Qu.:0.004975   1st Qu.:0.001478  
##  Median :0.003506   Median :0.005390   Median :0.001848  
##  Mean   :0.005744   Mean   :0.007775   Mean   :0.003405  
##  3rd Qu.:0.003896   3rd Qu.:0.006219   3rd Qu.:0.002217  
##  Max.   :0.111925   Max.   :0.129143   Max.   :0.089109
```

##### High accuracy, printing possible solutions


```r
answers8 <- predict(modelFit, ultima)
# answers produced from Random Forests with all major columns
```

#### Multinomial


```r
set.seed(31337)
modelFit <- train(classe ~ ., method="multinom", data=training)
```

```
## # weights:  280 (220 variable)
## initial  value 23687.707195 
## iter  10 value 22717.234740
## iter  20 value 17387.156958
## iter  30 value 16204.800870
## iter  40 value 15260.648277
## iter  50 value 15172.368666
## iter  60 value 13880.308784
## iter  70 value 13661.025786
## iter  80 value 13413.577832
## iter  90 value 13174.506987
## iter 100 value 13097.518677
## final  value 13097.518677 
## stopped after 100 iterations
```

```r
# and that took around 5-7 minutes..
confusionMatrix(testing$classe, predict(modelFit,testing))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1107   56   93  121   18
##          B  155  588   80   40   86
##          C  153   86  483  104   29
##          D   64   42   83  576   39
##          E   81  159   80  128  453
## 
## Overall Statistics
##                                           
##                Accuracy : 0.654           
##                  95% CI : (0.6405, 0.6673)
##     No Information Rate : 0.3181          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.5608          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7096   0.6316  0.58974   0.5944  0.72480
## Specificity            0.9139   0.9091  0.90894   0.9421  0.89530
## Pos Pred Value         0.7935   0.6196  0.56491   0.7164  0.50277
## Neg Pred Value         0.8709   0.9133  0.91702   0.9041  0.95703
## Prevalence             0.3181   0.1898  0.16701   0.1976  0.12745
## Detection Rate         0.2257   0.1199  0.09849   0.1175  0.09237
## Detection Prevalence   0.2845   0.1935  0.17435   0.1639  0.18373
## Balanced Accuracy      0.8117   0.7704  0.74934   0.7682  0.81005
```

#### Write answers


```r
# choose most often occuring combination
a <- table(cbind(id = 1:20, 
            stack(
              lapply(mget(ls(pattern = "answers\\d+")), 
                     as.character)))[c("id", "values")])
consensus <- colnames(a)[apply(a,1,which.max)]
pml_write_files(consensus)
```

#### Summary

#### Machine Learnings methods which showed to be most effective where: random forests, knn, and stochastic gradient boosting. In those all 3 I always used k-cross validation [with k=5] (or analyzing OOB for RandomForests)* to verify if accuracy is legit. Naive Bayes had accuracy of only 0.65-0.7. Finally, I used Majority Voting algorithm to choose candidates for final classification (hence the 'consensus' variable).

#### Majority Voting shown to be 100% accurate in this assignment

#### *["Project grading/Cross-Validation and Random Forests"](https://class.coursera.org/predmachlearn-013/forum/thread?thread_id=91)

##### Authored by Oskar Jarczyk, 26th April 2015, with Coursera Honor Code
