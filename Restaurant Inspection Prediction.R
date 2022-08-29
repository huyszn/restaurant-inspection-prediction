#####################################
## Name: Huy B. Mai
## Restaurant Inspection Prediction
#####################################

setwd("/Users/huymai/Documents/restaurant-inspection-prediction")

library(C50)
library(gmodels)
library(RWeka)
library(dplyr)
library(stringr)
library(caret)

# Import datasets
# https://data.sfgov.org/Health-and-Social-Services/Restaurant-Scores-LIVES-Standard/pyih-qa8i?row_index=0
# https://data.cityofnewyork.us/Health/DOHMH-New-York-City-Restaurant-Inspection-Results/43nn-pn8j
sf<-read.csv("Restaurant_Scores_-_LIVES_Standard.csv") # San Francisco - 53973 inspections
nyc<-read.csv("DOHMH_New_York_City_Restaurant_Inspection_Results.csv") # New York City - 248054 inspections

# First use algorithms on SF dataset

sf<-sf[,c(1,6,13,15,16,17)] # extract the business id, business postal code, inspection score, violation id, violation description, and risk category
sf<-sf[,c(1,2,3,4,5,6)]

# remove rows where risk category is only "" (empty), remove invalid postal codes
sf<- sf[!(sf$risk_category==""), ]
sf<- sf[!(sf$business_id==""), ]
sf<- sf[!(sf$business_postal_code=="" | sf$business_postal_code=="94122-1909" | sf$business_postal_code=="Ca" | sf$business_postal_code=="CA" | sf$business_postal_code=="94124-1917" | sf$business_postal_code=="94117-3504"), ]
sf<- sf[!(sf$inspection_score==""), ]
sf<- sf[!(sf$violation_id==""), ]

# Remove rows that have NA values
sf<-na.omit(sf)

# create violation code labels
sf$violation_id2 <- str_sub(sf$violation_id, start= -6) # get last 6 characters from id

sf <- sf[order(sf$violation_id2),]
sf <- transform(sf, violation_labels = as.numeric(factor(violation_id2)))

# add column risk_labels where low risk = 1, moderate risk = 2, high risk = 3
sf <- sf %>%
  mutate(risk_labels = case_when(
    risk_category == 'Low Risk' ~ 1,
    risk_category == 'Moderate Risk' ~ 2,
    risk_category == 'High Risk' ~ 3,
  ))

# create data frame that matches violation labels with description
sf_violation_desc_labels <- sf[,c(5,8)]
sf_violation_desc_labels <- sf_violation_desc_labels %>% 
  distinct()

sf<-sf[,c(1,2,3,8,9)]

# Train / Test sets
# use 85% train, 15% test
s = 0.85
train_sample <- sample(dim(sf)[1], s*dim(sf)[1])
str(train_sample)


# 85/15
sf_train <- sf[train_sample, ]
sf_test  <- sf[-train_sample, ]
# pick risk_labels for target
prop.table(table(sf_train$risk_labels))
prop.table(table(sf_test$risk_labels))

# removing target variable risk_labels (5th column)
sf_model <- C5.0(sf_train[,c("inspection_score","violation_labels")], as.factor(sf_train$risk_labels))
sf_model
summary(sf_model)
plot(sf_model)

sf_pred <- predict(sf_model, sf_test)
CrossTable(sf_test$risk_labels, sf_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual risk', 'predicted risk'))

# One Rule Model
sf_1R <- OneR(as.factor(risk_labels) ~ inspection_score+violation_labels, data = sf_train)
sf_1R
summary(sf_1R)

# ADAPTIVE BOOSTING

# run decision trees 10 times. algorithm chooses from 1st tree, 2nd tree, 3rd tree, etc.
sf_boost10 <- C5.0(sf_train[-5], as.factor(sf_train$risk_labels), trials = 10)
sf_boost10
summary(sf_boost10)

sf_boost_pred10 <- predict(sf_boost10, sf_test)
CrossTable(sf_test$risk_labels, sf_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual risk', 'predicted risk'))

# Second use algorithms on NYC dataset

nyc<-nyc[,c(1,3,6,8,11,12,13,15,18)] # extract the CAMIS, borough, zip code, cuisine description, inspection grade, inspection type, violation code, violation description, and critical flag

# remove grades that are N,P,Z,G and blanks, remove blank violations
nyc<- nyc[!(nyc$GRADE=="" | nyc$GRADE=="N" | nyc$GRADE=="P" | nyc$GRADE=="Z" | nyc$GRADE=="G"), ]
nyc<- nyc[!(nyc$VIOLATION.CODE==""), ]
nyc<- nyc[!(nyc$ZIPCODE=="" | nyc$ZIPCODE=="N/A"),]

nyc<-na.omit(nyc)

# create violation code labels
nyc <- nyc[order(nyc$VIOLATION.CODE),]
nyc <- transform(nyc, violation_labels = as.numeric(factor(VIOLATION.CODE)))

# create data frame that matches violation labels with description
nyc_violation_desc_labels <- nyc[,c(10,6)]
nyc_violation_desc_labels <- nyc_violation_desc_labels %>% 
  distinct()

# create borough code labels
nyc <- nyc[order(nyc$BORO),]
nyc <- transform(nyc, borough_labels = as.numeric(factor(BORO)))

# create grade labels
nyc <- nyc[order(nyc$GRADE),]
nyc <- transform(nyc, grade_labels = as.numeric(factor(GRADE)))

# create inspection labels
nyc <- nyc[order(nyc$INSPECTION.TYPE),]
nyc <- transform(nyc, inspection_labels = as.numeric(factor(INSPECTION.TYPE)))

# create cuisine labels
nyc <- nyc[order(nyc$CUISINE.DESCRIPTION),]
nyc <- transform(nyc, cuisine_labels = as.numeric(factor(CUISINE.DESCRIPTION)))

# create critical labels
nyc <- nyc %>%
  mutate(critical_labels = case_when(
    CRITICAL.FLAG == 'Not Critical' ~ 1,
    CRITICAL.FLAG == 'Critical' ~ 2
  ))

nyc<-nyc[,c(1,3,10,11,12,13,14,15)] 

# Train / Test sets
# use 85% train, 15% test
s = 0.85
train_sample <- sample(dim(nyc)[1], s*dim(nyc)[1])
str(train_sample)


# 85/15
nyc_train <- nyc[train_sample, ]
nyc_test  <- nyc[-train_sample, ]
# pick critical_labels for target
prop.table(table(nyc_train$critical_labels))
prop.table(table(nyc_test$critical_labels))

# removing target variable critical_labels (8th column)
nyc_model <- C5.0(nyc_train[,c('violation_labels','grade_labels')], as.factor(nyc_train$critical_labels))
nyc_model
summary(nyc_model)
plot(nyc_model)

nyc_pred <- predict(nyc_model, nyc_test)
CrossTable(nyc_test$critical_labels, nyc_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual critical', 'predicted critical'))

# One Rule Model
nyc_1R <- OneR(as.factor(critical_labels) ~ grade_labels+violation_labels, data = nyc_train)
nyc_1R
summary(nyc_1R)

# ADAPTIVE BOOSTING

# run decision trees 10 times. algorithm chooses from 1st tree, 2nd tree, 3rd tree, etc.
nyc_boost10 <- C5.0(nyc_train[-8], as.factor(nyc_train$critical_labels), trials = 10)
nyc_boost10
summary(nyc_boost10)

nyc_boost_pred10 <- predict(nyc_boost10, nyc_test)
CrossTable(nyc_test$critical_labels, nyc_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual critical', 'predicted critical'))

###### Now use algorithms on both datasets by combining them

# both datasets have inspection scores, violations, and risk/critical categories

# Extract inspection score, violation id, and risk columns in each dataset

sf<-read.csv("Restaurant_Scores_-_LIVES_Standard.csv") # San Francisco - 53973 inspections

sf<-sf[,c(13,15,17)] # extract the inspection score, violation id, and risk category

# Remove rows that have NA values
sf<-na.omit(sf)

# remove rows where columns only have"" (empty)
sf<- sf[!(sf$risk_category==""), ]
sf<- sf[!(sf$inspection_score==""), ]
sf<- sf[!(sf$violation_id==""), ]

# create violation code labels
sf$violation_id2 <- str_sub(sf$violation_id, start= -6) # get last 6 characters from id

sf <- sf[order(sf$violation_id2),]
sf <- transform(sf, violation_labels0 = as.numeric(factor(violation_id2)))

# have to match violations in nyc and sf dataset

# sf 1 matches nyc 19
# sf 2 matches nyc none
sf<- sf[!(sf$violation_labels0=="2"), ]
# sf 3 matches nyc none
sf<- sf[!(sf$violation_labels0=="3"), ]
# sf 4 matches nyc 8
# sf 5 matches nyc 1
# sf 6 matches nyc 3
# sf 7 matches nyc 23
# sf 8 matches nyc 41
# sf 9 matches nyc 10
# sf 10 matches nyc none
sf<- sf[!(sf$violation_labels0=="10"), ]
# sf 11 matches nyc 21
# sf 12 matches nyc none
sf<- sf[!(sf$violation_labels0=="12"), ]
# sf 13 matches nyc none
sf<- sf[!(sf$violation_labels0=="13"), ]
# sf 14 matches nyc none
sf<- sf[!(sf$violation_labels0=="14"), ]
# sf 15 matches nyc none
sf<- sf[!(sf$violation_labels0=="15"), ]
# sf 16 matches nyc 19
# sf 17 matches nyc 34
# sf 18 matches nyc none
sf<- sf[!(sf$violation_labels0=="18"), ]
# sf 19 matches nyc 45
# sf 20 matches nyc 24
# sf 21 matches nyc none
sf<- sf[!(sf$violation_labels0=="21"), ]
# sf 22 matches nyc 41
# sf 23 matches nyc 11
# sf 24 matches nyc none
sf<- sf[!(sf$violation_labels0=="24"), ]
# sf 25 matches nyc 44
# sf 26 matches nyc 6
# sf 27 matches nyc none
sf<- sf[!(sf$violation_labels0=="27"), ]
# sf 28 matches nyc 31
# sf 29 matches nyc none
sf<- sf[!(sf$violation_labels0=="29"), ]
# sf 30 matches nyc 52
# sf 31 matches nyc 40
# sf 32 matches nyc none
sf<- sf[!(sf$violation_labels0=="32"), ]
# sf 33 matches nyc none
sf<- sf[!(sf$violation_labels0=="33"), ]
# sf 34 matches nyc 38
# sf 35 matches nyc none
sf<- sf[!(sf$violation_labels0=="35"), ]
# sf 36 matches nyc 20
# sf 37 matches nyc none
sf<- sf[!(sf$violation_labels0=="37"), ]
# sf 38 matches nyc 66
# sf 39 matches nyc 59
# sf 40 matches nyc 61
# sf 41 matches nyc none
sf<- sf[!(sf$violation_labels0=="41"), ]
# sf 42 matches nyc 42
# sf 43 matches nyc 56
# sf 44 matches nyc 58
# sf 45 matches nyc 43
# sf 46 matches nyc 55
# sf 47 matches nyc none
sf<- sf[!(sf$violation_labels0=="47"), ]
# sf 48 matches nyc 54
# sf 49 matches nyc none
sf<- sf[!(sf$violation_labels0=="49"), ]
# sf 50 matches nyc none
sf<- sf[!(sf$violation_labels0=="50"), ]
# sf 51 matches nyc 60
# sf 52 matches nyc none
sf<- sf[!(sf$violation_labels0=="52"), ]
# sf 53 matches nyc none
sf<- sf[!(sf$violation_labels0=="53"), ]
# sf 54 matches nyc none
sf<- sf[!(sf$violation_labels0=="54"), ]
# sf 55 matches nyc none
sf<- sf[!(sf$violation_labels0=="55"), ]
# sf 56 matches nyc 45
# sf 57 matches nyc none
sf<- sf[!(sf$violation_labels0=="57"), ]
# sf 58 matches nyc none
sf<- sf[!(sf$violation_labels0=="58"), ]
# sf 59 matches nyc none
sf<- sf[!(sf$violation_labels0=="59"), ]
# sf 60 matches nyc none
sf<- sf[!(sf$violation_labels0=="60"), ]
# sf 61 matches nyc none
sf<- sf[!(sf$violation_labels0=="61"), ]
# sf 62 matches nyc none
sf<- sf[!(sf$violation_labels0=="62"), ]
# sf 63 matches nyc none
sf<- sf[!(sf$violation_labels0=="63"), ]
# sf 64 matches nyc none
sf<- sf[!(sf$violation_labels0=="64"), ]
# sf 65 matches nyc 30

# create violation_labels and replace violation labels with the matches from nyc
sf$violation_labels <- NA

sf$violation_labels[sf$violation_labels0==1] <- 19
sf$violation_labels[sf$violation_labels0==4] <- 8
sf$violation_labels[sf$violation_labels0==5] <- 1
sf$violation_labels[sf$violation_labels0==6] <- 3
sf$violation_labels[sf$violation_labels0==7] <- 23
sf$violation_labels[sf$violation_labels0==8] <- 41
sf$violation_labels[sf$violation_labels0==9] <- 10
sf$violation_labels[sf$violation_labels0==11] <- 21
sf$violation_labels[sf$violation_labels0==16] <- 19
sf$violation_labels[sf$violation_labels0==17] <- 34
sf$violation_labels[sf$violation_labels0==19] <- 45
sf$violation_labels[sf$violation_labels0==20] <- 24
sf$violation_labels[sf$violation_labels0==22] <- 41
sf$violation_labels[sf$violation_labels0==23] <- 11
sf$violation_labels[sf$violation_labels0==25] <- 44
sf$violation_labels[sf$violation_labels0==26] <- 6
sf$violation_labels[sf$violation_labels0==28] <- 31
sf$violation_labels[sf$violation_labels0==30] <- 52
sf$violation_labels[sf$violation_labels0==31] <- 40
sf$violation_labels[sf$violation_labels0==34] <- 38
sf$violation_labels[sf$violation_labels0==36] <- 20
sf$violation_labels[sf$violation_labels0==38] <- 66
sf$violation_labels[sf$violation_labels0==39] <- 59
sf$violation_labels[sf$violation_labels0==40] <- 61
sf$violation_labels[sf$violation_labels0==42] <- 42
sf$violation_labels[sf$violation_labels0==43] <- 56
sf$violation_labels[sf$violation_labels0==44] <- 58
sf$violation_labels[sf$violation_labels0==45] <- 43
sf$violation_labels[sf$violation_labels0==46] <- 55
sf$violation_labels[sf$violation_labels0==48] <- 54
sf$violation_labels[sf$violation_labels0==51] <- 60
sf$violation_labels[sf$violation_labels0==56] <- 45
sf$violation_labels[sf$violation_labels0==65] <- 30

# remove all rows with moderate risk
sf<- sf[!(sf$risk_category=="Moderate Risk"), ]

# add column risk_labels where low risk = 1, high risk = 2
sf <- sf %>%
  mutate(risk_labels = case_when(
    risk_category == 'Low Risk' ~ 1,
    risk_category == 'High Risk' ~ 2,
  ))

# add inspection_grade to convert scores to grades
sf <- sf %>%
  mutate(grade_labels = case_when(
    inspection_score >= 90 ~ 1, # A
    inspection_score < 90 & inspection_score >= 80 ~ 2, # B
    inspection_score < 80 ~ 3, # C
  ))

# final sf with violation_labels, risk_labels, and grade_labels
sf <- sf[,c(6,7,8)]


nyc<-read.csv("DOHMH_New_York_City_Restaurant_Inspection_Results.csv") # New York City - 248054 inspections

nyc<-nyc[,c(11,13,15)] # extract the inspection grade, violation code, and critical flag

# remove grades that are N,P,Z,G and blanks, remove blank violations
nyc<- nyc[!(nyc$GRADE=="" | nyc$GRADE=="N" | nyc$GRADE=="P" | nyc$GRADE=="Z" | nyc$GRADE=="G"), ]
nyc<- nyc[!(nyc$VIOLATION.CODE==""), ]

nyc<-na.omit(nyc)

# create violation code labels
nyc <- nyc[order(nyc$VIOLATION.CODE),]
nyc <- transform(nyc, violation_labels = as.numeric(factor(VIOLATION.CODE)))

# create grade labels
nyc <- nyc[order(nyc$GRADE),]
nyc <- transform(nyc, grade_labels = as.numeric(factor(GRADE)))

# create risk labels
nyc <- nyc %>%
  mutate(risk_labels = case_when(
    CRITICAL.FLAG == 'Not Critical' ~ 1,
    CRITICAL.FLAG == 'Critical' ~ 2
  ))

# remove rows with violations that were not in sf
nyc<- nyc[!(nyc$violation_labels=="2"), ]
nyc<- nyc[!(nyc$violation_labels=="4"), ]
nyc<- nyc[!(nyc$violation_labels=="5"), ]
nyc<- nyc[!(nyc$violation_labels=="7"), ]
nyc<- nyc[!(nyc$violation_labels=="9"), ]
nyc<- nyc[!(nyc$violation_labels=="12"), ]
nyc<- nyc[!(nyc$violation_labels=="13"), ]
nyc<- nyc[!(nyc$violation_labels=="14"), ]
nyc<- nyc[!(nyc$violation_labels=="15"), ]
nyc<- nyc[!(nyc$violation_labels=="16"), ]
nyc<- nyc[!(nyc$violation_labels=="17"), ]
nyc<- nyc[!(nyc$violation_labels=="18"), ]
nyc<- nyc[!(nyc$violation_labels=="22"), ]
nyc<- nyc[!(nyc$violation_labels=="25"), ]
nyc<- nyc[!(nyc$violation_labels=="26"), ]
nyc<- nyc[!(nyc$violation_labels=="27"), ]
nyc<- nyc[!(nyc$violation_labels=="28"), ]
nyc<- nyc[!(nyc$violation_labels=="29"), ]
nyc<- nyc[!(nyc$violation_labels=="32"), ]
nyc<- nyc[!(nyc$violation_labels=="33"), ]
nyc<- nyc[!(nyc$violation_labels=="35"), ]
nyc<- nyc[!(nyc$violation_labels=="36"), ]
nyc<- nyc[!(nyc$violation_labels=="37"), ]
nyc<- nyc[!(nyc$violation_labels=="39"), ]
nyc<- nyc[!(nyc$violation_labels=="46"), ]
nyc<- nyc[!(nyc$violation_labels=="47"), ]
nyc<- nyc[!(nyc$violation_labels=="48"), ]
nyc<- nyc[!(nyc$violation_labels=="49"), ]
nyc<- nyc[!(nyc$violation_labels=="50"), ]
nyc<- nyc[!(nyc$violation_labels=="51"), ]
nyc<- nyc[!(nyc$violation_labels=="53"), ]
nyc<- nyc[!(nyc$violation_labels=="57"), ]
nyc<- nyc[!(nyc$violation_labels=="62"), ]
nyc<- nyc[!(nyc$violation_labels=="63"), ]
nyc<- nyc[!(nyc$violation_labels=="64"), ]
nyc<- nyc[!(nyc$violation_labels=="65"), ]

# final nyc with violation_labels, risk_labels, and grade_labels
nyc <- nyc[,c(4,5,6)]

# combine nyc and sf together

sfnyc <- rbind(sf, nyc)

# Train / Test sets
# use 85% train, 15% test
s = 0.85
train_sample <- sample(dim(sfnyc)[1], s*dim(sfnyc)[1])
str(train_sample)


# 85/15
sfnyc_train <- sfnyc[train_sample, ]
sfnyc_test  <- sfnyc[-train_sample, ]
# pick risk_labels for target
prop.table(table(sfnyc_train$risk_labels))
prop.table(table(sfnyc_test$risk_labels))

# removing target variable risk_labels (2nd column)
sfnyc_model <- C5.0(sfnyc_train[-2], as.factor(sfnyc_train$risk_labels))
sfnyc_model
summary(sfnyc_model)
plot(sfnyc_model)

sfnyc_pred <- predict(sfnyc_model, sfnyc_test)
CrossTable(sfnyc_test$risk_labels, sfnyc_pred,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual risk', 'predicted risk'))

# One Rule Model
sfnyc_1R <- OneR(as.factor(risk_labels) ~ ., data = sfnyc_train)
sfnyc_1R
summary(sfnyc_1R)

# ADAPTIVE BOOSTING

# run decision trees 10 times. algorithm chooses from 1st tree, 2nd tree, 3rd tree, etc.
sfnyc_boost10 <- C5.0(sfnyc_train[-2], as.factor(sfnyc_train$risk_labels), trials = 10)
sfnyc_boost10
summary(sfnyc_boost10)
plot(sfnyc_boost10)

sfnyc_boost_pred10 <- predict(sfnyc_boost10, sfnyc_test)
CrossTable(sfnyc_test$risk_labels, sfnyc_boost_pred10,
           prop.chisq = FALSE, prop.c = FALSE, prop.r = FALSE,
           dnn = c('actual risk', 'predicted risk'))

