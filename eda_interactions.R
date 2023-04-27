################### Statistical learning using R ###################
#                      Valentina Zangirolami                       #
#                           28/04/2023                             #
####################################################################

# What we're going to see today -------------------------------------------

# - Missing values analysis
# - Exploration data analysis (Data Visualization with ggplot) 
# - Interaction terms/feature engineering

# Dataset -----------------------------------------------------------------

#Download the dataset from Kaggle: https://www.kaggle.com/datasets/dansbecker/melbourne-housing-snapshot
library(tidyverse)
#Load the dataset
getwd() # to show current directory

#if you want to set a new directory, use setwd() as we saw in the first class
data_house_prices <- read_csv("melb_data.csv")

head(data_house_prices) # head of the dataset
dim(data_house_prices) # dimensions (13580 rows, 21 variables)

summary(data_house_prices) # summary: it's useful to 
#show main characteristics for each variable

# We can observe that the dataset includes both numeric and factor variables
# Car, BuildingArea and YearBuilt contain missing data

#We also can see that we have a variable depending on date-format
library(lubridate)

str(data_house_prices$Date) #chr class
data_house_prices$Date <- dmy(data_house_prices$Date) #convert in date class
str(data_house_prices$Date) #date class!

#Why is it important? Lubridate provides some functions for date class
#e.g. for calculating the difference between two or more date
#look the cheat sheet: https://rawgit.com/rstudio/cheatsheets/main/lubridate.pdf
#you can also click on Help --> cheat sheets

# Missing values ----------------------------------------------------------

library(ggplot2)
library(naniar) 
library(finalfit)
library(funModeling)

#NB: The goal is not to go deeply in the analysis of missing values. 
#Btw we're going to see faster how to handle them.

#First of all, you need to remember that not all missing values are the same.
#There are three kind of NAs: MCAR, MAR and NMAR (Worst case! :( ).
#A way to start NA's analysis can be: try to understand if the missing values of 
#a variable depend on other variables. For instance, When we have a variable with 
#NAs where NAs depend on another predictor (e.g. gender type). 
#--> NAs cannot be "Missing completely at random (MCAR)"

# Begin with quantifying the percentage of NAs
data_house_prices %>% ff_glimpse() #CouncilArea has NAs

#plot for NA values

data_house_prices %>% select(Car, BuildingArea, YearBuilt, CouncilArea) %>% vis_miss()
data_house_prices %>% missing_plot()

#NAs: Car <1%, YearBuilt and BuildingArea >40%, CouncilArea roughly 10%
#Looking for another plot to extract more info 

data_house_prices %>% select(Car, BuildingArea, YearBuilt, CouncilArea) %>% gg_miss_upset()

#This plot give us some useful info about "the position" of missing data
#30 rows contain missing data among all 4 variables
#4513 rows for both YearBuilt and BuildingArea 

#In a simple way, we can remove rows with NAs for Car and CouncilArea, 
#and the variable BuildingArea (missing values are roughly 50%)

data_house_prices <- data_house_prices %>% filter(!is.na(Car), !is.na(CouncilArea)) %>% select(-BuildingArea)
dim(data_house_prices) #check

#Handling NA for YearBuilt

ggplot(data_house_prices, aes(x=YearBuilt, y=Price)) + geom_miss_point()
ggplot(data_house_prices, aes(x=YearBuilt, y=Postcode)) + geom_miss_point()

#No relation with YearBuilt and (Price, Postcode), let's assume that YearBuilt 
#is MCAR without analyzing more the phenomena. For instance, we can replace NAs 
#with the median or predicted values

#If we want to replace NA with the median, we can use

data_house_prices %>% mutate(YearBuilt=ifelse(is.na(YearBuilt),median(data_house_prices$YearBuilt,na.rm=T),YearBuilt))

#Is it a right choice in this case? Maybe! Handling missing value is not 
#a straightforward stuff (in this phase, it's a prediction problem!)
#btw, we can compare two kind of models (and the median above) looking for 
#the best model to predict the values of NAs

#Create a set without NAs
df_nomiss <- data_house_prices %>% filter(!is.na(YearBuilt)) %>% select(Rooms, Postcode, Bedroom2, Bathroom, Car, YearBuilt, Propertycount, Regionname, Distance, Type)

#Create the training and test set. In general, the percentage used to generate
#the training set is 80% (and the last 20% observations could be used for
#building the validation set and the test set). In this case, we just used the
#training and test set 

set.seed(123) #seed for reproducibility

df_nomiss <-df_nomiss %>% mutate(rn=row_number())
data_test <- df_nomiss %>% slice_sample(prop=-0.8)
data_train <-df_nomiss %>% slice(-pull(data_test,rn))
df_nomiss <-df_nomiss %>% select(-rn)

y_test <- as.factor(data_test$YearBuilt)
y_train <- as.factor(data_train$YearBuilt)
data_test <- data_test %>% select(-YearBuilt)

#median
train_median <- median(data_train$YearBuilt)
predicted_median <- rep(train_median, times=nrow(data_test))
train_median <- rep(train_median, times= length(y_train))

#Proportional Odds Logistic Regression
data_train$YearBuilt <- as.factor(data_train$YearBuilt)

polr_model <- MASS::polr(YearBuilt~.,data= data_train, Hess=TRUE)
predicted_polr <- predict(polr_model, data_test)
pred_train <- predict(polr_model, data_train)

#Random Forest
library(randomForest)
rf_model <- randomForest(YearBuilt~., data=data_train, proximity=TRUE)
predicted_rf <- predict(rf_model, data_test)

y_test <- factor(y_test, levels=levels(y_train))

#Model evaluation
accuracy <- function(predicted_values, true_response){
  round(length(which(true_response==predicted_values))/length(true_response),2)}

(train_accuracy <- data.frame('Median'=accuracy(train_median, y_train), 'POLR'=accuracy(pred_train, y_train), 'RF'= accuracy(rf_model$predicted, y_train)))
(test_accuracy <- data.frame('Median'=accuracy(predicted_median, y_test), 'POLR'=accuracy(predicted_polr, y_test), 'RF'= accuracy(predicted_rf, y_test)))

#low accuracy for all the models, but rf is the best model
rm(accuracy, train_accuracy, test_accuracy, predicted_median, predicted_polr, predicted_rf, train_median)
rm(y_test, y_train, pred_train, rf_model, polr_model, data_train, data_test)

#replace NAs
rf_model <- randomForest(YearBuilt~., data=df_nomiss, proximity=TRUE)

data_house_prices <- data_house_prices %>% mutate(YearBuilt=ifelse(is.na(YearBuilt),predict(rf_model,.),YearBuilt))
rm(rf_model, df_nomiss)

#check
any_na(data_house_prices) #no more NAs

gc()

# Exploration analysis (Data Visualization with ggplot)  ------------------

#First of all, we want to reduce the dataset to speed up this example 
ggplot(data_house_prices, aes(x = Regionname)) +geom_bar()
#Focus our attention on South and West Australia
data_sample <- data_house_prices%>% filter(Regionname %in% c("Western Metropolitan", "Southern Metropolitan"))
rm(data_house_prices)

# Let's start exploration of data

data_sample %>% ff_glimpse() #summary for each variable
#more useful than classical summary because we can see also info about factors 
#Address has 6954 levels (clearly) --> let's remove it 
data_sample <- data_sample %>% select(-Address)

#CouncilArea
ggplot(data_sample, aes(x = CouncilArea)) +geom_bar()
unique(data_sample$CouncilArea)
#What's up?
#There is a level "unavailable" which corrisponds to NA:
data_sample<- data_sample %>% filter(CouncilArea!= "Unavailable")

#Suburb

max_level <- length(unique(data_sample$Suburb))
#Given that this variable contains a lot of levels, we can use
#scale discrete to rename them 
ggplot(data_sample, aes(x = Suburb)) +geom_bar() + scale_x_discrete(labels=as.character(seq(1,max_level)))
rm(max_level)
#it's not very useful, we can just draw out Suburb levels greater than a threshold
list_suburb <- data_sample %>% group_by(Suburb) %>%count %>% filter(n > 60)%>% select(Suburb)

#plot
data_sample %>% 
  group_by(Suburb) %>%
  count %>%
  filter(n > 60) %>%
  ggplot(aes(x = Suburb, y = n, fill=Suburb)) + 
  geom_bar(stat = "identity") + scale_x_discrete(labels=as.character(seq(1,nrow(list_suburb))))+
  scale_fill_discrete(name = "Suburb", labels = list_suburb$Suburb)

#Bentleigh is the level with more observations

list_suburb <- data_sample %>% group_by(Suburb) %>%count %>% filter(n < 30)%>% select(Suburb)

#plot
data_sample %>% 
  group_by(Suburb) %>%
  count %>%
  filter(n < 30) %>%
  ggplot(aes(x = Suburb, y = n, fill=Suburb)) + 
  geom_bar(stat = "identity") + scale_x_discrete(labels=as.character(seq(1,nrow(list_suburb))))+
  scale_fill_discrete(name = "Suburb", labels = list_suburb$Suburb)

#Williams Landing, Rockbank, Kooyong and Keilor Lodge less popular

rm(list_suburb)

####EXERCISEs####

# 1. Try to explore Landsize. What did you find out? How can we handle this situation?
# 2. Explore SellerG.

#################

#### Relation between predictors ####

#Method and Region

ggplot(data_sample, aes(x=Method, fill=Method, group=Regionname)) +
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") + facet_wrap(~ Regionname)

#Most popular selling method is: S - property sold

#Type and Region

ggplot(data_sample, aes(x=Type, fill=Type, group=Regionname)) +
  geom_bar(aes(y = ..prop.., fill = factor(..x..)), stat="count") + 
  scale_y_continuous(labels=scales::percent) +
  ylab("relative frequencies") + facet_wrap(~ Regionname)
#type h is the most popular in both south and west, but look the proportion...
#in south australia the type h and u are both preferred, in west australia
#type h is the most preferred

#### EXERCISES ####

# Propose other plots to explore much more relation between predictors.

###################

#### Predictors and Response variable ####

#Price vs Distance-Type-Regionname

ggplot(data_sample, aes(x=Distance, y=Price, colour=factor(Type))) + geom_jitter() + facet_grid(cols = vars(Regionname)) + theme_minimal()

#Price vs Method-Date-Regionname

ggplot(data_sample, aes(x=Date, y=Price, colour=Method)) + geom_line() + 
  facet_grid(cols = vars(Regionname)) + 
  theme_minimal() +scale_x_date(date_labels = "%m-%Y")

#In the south Australia we can find more expensive houses
#type u houses are cheaper, type h the most expensive
#In the south, finding houses far from CBD greater than 20 miles is uncommon

#### EXERCISES ####

# Explore other relationships between predictors and response variable 

##################

# Interactions & Feature engineering -------------------------------------------------------

#Until now, we saw some relations between predictors and response variable.
#For instance, We saw how the price change with the number of rooms but also
#considering the distance from CBD we observed a quite different effect:
#It's true that the price tends to increase as the number of rooms increase 
#But this effect is more significant when the house is closed to CBD

#How can we improve the predictive relationships with the response?
#How can we create useful interaction/new predictors?

#We can start by visualizing the correlation between variables

require(corrgram)
data_sample %>% dplyr::select(where(is.numeric)) %>% 
  corrgram(main="Correlation between variables - House prices",lower.panel=panel.pts)

#NB: correlation just gives us an information about the LINEAR relationship
#among variables

#Collinearity

data_sample %>% ff_glimpse()

#First consideration: Our dataset contains different predictors for describing 
#the position of the houses. Regionname is the low level description of areas until
#the suburb that's the highest level. Council Area is in the middle.
#To going on in a more easy way, can be useful to use just
#Council Area and Regionname 

data_sample <- data_sample %>% select(-Suburb)

library(mctest)
library(car)

#categorical variables

categorical_var <- data_sample %>% dplyr::select(where(is.character),Price) 
vif(lm(Price~., data=categorical_var), type=c("terms", "predictor"))
alias(lm(Price~., data=categorical_var)) #There's a problem of collinearity
#with Regionname and Council Area

#numerical variables

numeric_var <- data_sample %>% dplyr::select(where(is.numeric)) 
model_lm <- lm(Price~., data=numeric_var)
imcdiag(model_lm) #Detected collinearity among Bedroom, Bathroom and Rooms

rm(model_lm, numeric_var, categorical_var)

#looking VIF values, we can say that Rooms and Bedroom have much huge troubles
#with collinearity (as Regionname-CouncilArea)
#Since the nature of these variable, we can remove bedroom and bathroom
#keeping the variable rooms which involves an "overall" information or deciding 
#to combine these variables.

#let's start by looking the interaction plot
library(sjPlot)

#Bathroom - Bedroom2 - Rooms
model_lm <- lm(Price~Bathroom*Rooms, data_sample)
plot_model(model_lm , type = "pred", terms = c("Bathroom", "Rooms"))
model_lm <- lm(Price~Bedroom2*Rooms, data_sample)
plot_model(model_lm , type = "pred", terms = c("Bedroom2", "Rooms"))
model_lm <- lm(Price~Bedroom2*Rooms+Bathroom*Rooms+Bathroom*Rooms*Bedroom2, data_sample)
plot_model(model_lm , type = "pred", terms = c("Bathroom", "Rooms","Bedroom2"))

#Regionname -CouncilArea

model_lm <- lm(Price~Regionname*CouncilArea, data_sample)
plot_model(model_lm , type = "pred", terms = c("Regionname","CouncilArea"))

#Look the warning message: Estimation without full rank
#From the interaction plot we can observe that CouncilArea give us a redudant 
#information. In this case, perhaps it's better remove CouncilArea
#While we can create second and third order interaction for Bedroom2-Bathroom-Rooms

data_sample <- data_sample %>% select(-CouncilArea, -SellerG)
#*SellerG was just removed to obtain a better visualization of the below results

#Evaluating interactions

mod_noint <- lm(Price~Bathroom + Rooms + Bedroom2, data_sample)
summary(mod_noint)
#By following the hierarchy principle...
mod_int2 <- update(mod_noint, . ~ .+Bedroom2*Rooms+Bathroom*Rooms+Bedroom2*Bathroom)
summary(mod_int2)
#What about the relationship between variables?
#By adding the second order interactions, we can see:
# - Rooms & Bedroom2: the relationship is antagonistic
# - Rooms & Bathroom: the relationship is atypical
# - Bedroom2 & Bathroom: the relationship is atypical
#Let's look adding the third order interaction
mod_int3 <- update(mod_int2, . ~ .+Bathroom*Rooms*Bedroom2)
summary(mod_int3)

#What about the relationship between variables?

(R_squared_adj <- c(summary(mod_noint)$adj.r.squared, summary(mod_int2)$adj.r.squared, summary(mod_int3)$adj.r.squared))
(R_squared <- c(summary(mod_noint)$r.squared, summary(mod_int2)$r.squared, summary(mod_int3)$r.squared))

rm(mod_noint, mod_int2, mod_int3)

#Let's start with baseline linear model
model_lm <- lm(Price~., data_sample)
summary(model_lm) #0.654

#Second order interactions

model_lm_int2 <- lm(Price~.+Bedroom2*Rooms+Bathroom*Rooms+Bedroom2*Bathroom, data=data_sample)
summary(model_lm_int2) #Rsquared around 0.6588
#What can we conclude about effect sparsity?
#Adding the second interaction into the model, the coefficient of bathroom 
#(main effect) become less significant...

#Third order interaction

model_lm_int3 <- lm(Price~.+Bedroom2*Rooms+Bathroom*Rooms+Bathroom*Rooms*Bedroom2, data=data_sample)
summary(model_lm_int3) #Rsquared around 0.6631

#What can we conclude about effect sparsity?

rm(model_lm, model_lm_int2)

options(warn=-1)
par(mfrow=c(2,2))
plot(model_lm_int3) #non-linear relationship?
options(warn=0)

#### EXERCISE #### 

#Try to evaluate other interactions starting, for instance, from the 
#correlation matrix or the linear model.
