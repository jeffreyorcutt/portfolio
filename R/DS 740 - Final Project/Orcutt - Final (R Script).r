library(here)
library(dplyr)
library(corrplot)
library(randomForest)
library(caret)
library(ggdendro)
library(ggformula)
library(tidyr)
library(mice)
library(ggcorrplot)
library(NbClust)

#load the CSVs and combine them into a single dataframe. 

arabica <- read.csv(here("coffee-quality-database","data", "arabica_data_cleaned.csv"))
robusta <- read.csv(here("coffee-quality-database","data", "robusta_data_cleaned.csv"))
#had five columns that did not align perfectly, establish which those are and 
#rename the offending columns in robusta

#find the columns and match manually
setdiff(names(arabica), names(robusta))
setdiff(names(robusta), names(arabica))

#rename the offending columns
robusta <- robusta %>% rename(
  Aroma = Fragrance...Aroma,
  Acidity = Salt...Acid,
  Sweetness = Bitter...Sweet,
  Body = Mouthfeel,
  Uniformity = Uniform.Cup
)
#concatenate the two dataframes 

joined_data <- rbind(arabica, robusta)



#move low-frequency countries into "other-continent" groupings
#Other-Asia - Japan, Myanmar, Philippines, Vietnam, Laos, Papua New Guinea, 
#Other-Africa - Cote d?Ivoire, Rwanda, Zambia, Burundi, 
#Other-CentAmIsl - Panama, Haiti, U.S(Puerto Rico)

joined_data$Country.of.Origin <- ifelse(joined_data$Country.of.Origin %in%
          c("Japan","Myanmar","Philippines", "India", "China", "Vietnam","Laos",
            "Papua New Guinea"), "Asia-Other", joined_data$Country.of.Origin)

joined_data$Country.of.Origin <- ifelse(joined_data$Country.of.Origin %in%
          c("Cote d?Ivoire","Malawi", "Rwanda", "Zambia", "Burundi", "Mauritius"),
        "Africa-Other", joined_data$Country.of.Origin)

joined_data$Country.of.Origin <- ifelse(joined_data$Country.of.Origin %in%
          c("Panama", "Peru", "Ecuador","Haiti", "United States","United States (Puerto Rico)"),
        "CentrAmIsl-Other", joined_data$Country.of.Origin)

#combine "Bluish-Green" into "Blue-Green", "none" isn"t a color, 
#impute it with the median value (green)
joined_data$Color <- ifelse(joined_data$Color %in% c("Bluish-Green"), 
                            "Blue-Green", 
                            joined_data$Color)

joined_data$Color <- ifelse(joined_data$Color == "None", 
                            "Green", 
                            joined_data$Color)

#fix a few math errors from the calculation of mean_altitude
joined_data$altitude_mean_meters <- ifelse(
  joined_data$altitude_mean_meters > 9000, 
  joined_data$altitude_mean_meters / 1000, 
  joined_data$altitude_mean_meters)

#None isn"t a valid drying processing method, impute with median value (Natural Dry)
joined_data$Processing.Method <- ifelse(
  joined_data$Processing.Method == "None", "Natural...Dry", 
  joined_data$Processing.Method)

#remove rows with missing country and Total.Cup.Points data
joined_data <- joined_data %>% filter(Country.of.Origin != "")
joined_data <- joined_data %>% filter(Total.Cup.Points != 0)
# joined_data <- joined_data %>% filter(Color != "")
# joined_data <- joined_data %>% filter(Processing.Method != "")

#cleanup the Harvest.Year column 
joined_data <- joined_data %>% mutate(Year = 
                case_when(Harvest.Year == "08/09 crop" ~ "2008", 
                          Harvest.Year == "2009 - 2010" ~ "2009",
                          Harvest.Year == "2009 / 2010" ~ "2009",
                          Harvest.Year == "2009-2010" ~ "2009",
                          Harvest.Year == "2009/2010" ~ "2009",
                          Harvest.Year == "Sept 2009 - April 2010" ~ "2009",
                          Harvest.Year == "December 2009-March 2010" ~ "2009",
                          Harvest.Year == "Fall 2009" ~ "2009",
                          Harvest.Year == "47/2010" ~ "2010",
                          Harvest.Year == "4T/10" ~ "2010",
                          Harvest.Year == "4t/2010" ~ "2010",
                          Harvest.Year == "4T/2010" ~ "2010",
                          Harvest.Year == "23 July 2010" ~ "2010",   
                          Harvest.Year == "4T72010" ~ "2010", 
                          Harvest.Year == "2010-2011" ~ "2010",    
                          Harvest.Year == "March 2010" ~ "2010",                                         
                          Harvest.Year == "1t/2011" ~ "2011",
                          Harvest.Year == "1T/2011" ~ "2011",
                          Harvest.Year == "3T/2011" ~ "2011",
                          Harvest.Year == "Abril - Julio /2011" ~ "2011",
                          Harvest.Year == "4t/2011" ~ "2011",
                          Harvest.Year == "Spring 2011 in Colombia." ~ "2011", 
                          Harvest.Year == "January 2011" ~ "2011",                                        
                          Harvest.Year == "2011/2012" ~ "2011",
                          Harvest.Year == "2013/2014" ~ "2013",
                          Harvest.Year == "2014/2015" ~ "2015",
                          Harvest.Year == "2015/2016" ~ "2015",
                          Harvest.Year == "2016 / 2017" ~ "2016",
                          Harvest.Year == "2016/2017" ~ "2016", 
                          Harvest.Year == "2017 / 2018" ~ "2017",
                          Harvest.Year == "Abril - Julio" ~ "",
                          Harvest.Year == "August to December" ~ "",
                          Harvest.Year == "January Through April" ~ "",
                          Harvest.Year == "May-August" ~ "",
                          Harvest.Year == "Mayo a Julio" ~ "",
                          Harvest.Year == "mmm" ~ "",
                          Harvest.Year == "TEST" ~ "",
                          TRUE ~ Harvest.Year))

#Shorten the drying method column values to industry terms, some weirdness was 
#happening when named Processing.Method, so created a new column
joined_data <- joined_data %>% mutate(Processing.Method_2 = 
                case_when(Processing.Method == "Natural / Dry" ~ "Dry",
                          Processing.Method == "Pulped natural / honey" ~ "Honey Pulped",
                          Processing.Method == "Semi-washed / Semi-pulped" ~ "Semi-Pulped",
                          Processing.Method == "Washed / Wet" ~ "Wet",
                          TRUE ~ Processing.Method))


#getting rid of farm name, lot number, mill, certification information, 
#company information, producer, number of bags, bag weights, 
#in country partner, grading date, owner, variety
joined_data <- joined_data %>% select(
  c("Country.of.Origin",
    "Species",
    "Processing.Method_2",
    "Aroma",
    "Flavor",
    "Aftertaste",
    "Acidity",
    "Sweetness", 
    "Body",
    "Uniformity",
    "Clean.Cup",
    "Total.Cup.Points",
    "Variety",
    "Moisture", 
    "Category.One.Defects", 
    "Quakers", 
    "Color",
    "Category.Two.Defects",
    "altitude_mean_meters", 
    "Year"))

# table(joined_data$Processing.Method)
# table(joined_data$Color)

#set several variables as factors
joined_data$Color <- as.factor(joined_data$Color)
joined_data$Country.of.Origin <- as.factor(joined_data$Country.of.Origin)
joined_data$Processing.Method <- as.factor(joined_data$Processing.Method)
joined_data$Species <- as.factor(joined_data$Species)
joined_data$Variety <- as.factor(joined_data$Variety)

#impute NAs with mean for numeric columns and mode for factor columns
#this function was borrowed from: 
#https://www.r-bloggers.com/2020/04/how-to-impute-missing-values-in-r/
#quick visual way to check for NAs
md.pattern(joined_data)
getmode <- function(v){
  v=v[nchar(as.character(v))>0]
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}
for (cols in colnames(joined_data)) {
  if (cols %in% names(joined_data[,sapply(joined_data, is.numeric)])) {
    joined_data <- joined_data %>% 
      mutate(!!cols := replace(!!rlang::sym(cols),
                               is.na(!!rlang::sym(cols)),
                               mean(!!rlang::sym(cols), na.rm=TRUE)
      )
      )
  }
  else {
    joined_data <- joined_data %>% 
      mutate(!!cols := replace(!!rlang::sym(cols), 
                               !!rlang::sym(cols)=="", 
                               getmode(!!rlang::sym(cols))
      )
      )
    
  }
}
md.pattern(joined_data)
#table below shows no missing values remaining
table(joined_data$Color)

#make a correlation plot for below

#make a dataframe with no indicator, no factors
joined_data_no_ind <- joined_data %>% 
  select(-c("Country.of.Origin", "Processing.Method_2","Year",
            "Species","Color","Processing.Method", "Variety"))

#create correlation matrix
corrdata <- cor(joined_data_no_ind)
ggcorrplot(corrdata,
           lab=F,
           type="lower")

#I didn"t end up using the correlation matrix in my paper. 

#make a copy of the cleaned dataset for use in graphics and analysis
orig_data <- joined_data

#rename column to shorten predictors after Dummy variables, then apply dummyVars
#to factors and remove original columns
joined_data <- joined_data %>% rename(ProcMethod = Processing.Method_2)
joined_data_tnsf<- dummyVars(" ~ Species + ProcMethod+ Color + Year ",
                             data = joined_data)
joined_data_ohe <- data.frame(predict(joined_data_tnsf, newdata = joined_data))
joined_data <- joined_data %>% 
  select(-c(Species, Processing.Method, Color, Year, ProcMethod))
joined_data <- cbind(joined_data, joined_data_ohe)
joined_data_no_ind <- joined_data %>% select(-c("Country.of.Origin"))

#this section is to establish the no information rate of the country variable. 
country_tbl <- table(joined_data$Country.of.Origin)
max_country <- tail(names(sort(table(joined_data$Country.of.Origin))), 1)
max_country_n <- country_tbl[names(country_tbl) == max_country]
country_NIR <- max_country_n / dim(joined_data[1])[1]

dataused = joined_data
set.seed(88)
ctrl <- trainControl(method="cv", number = 5)

ncols <- ncol(joined_data)

# random forest with all predictors
fit_caret_rf_select <-
  train(Country.of.Origin ~ .,
        data = dataused,
        method = "rf",
        na.action = na.roughfix,
        tuneGrid = expand.grid(.mtry = c(1:(ncols -1 ))),
        nodesize = c(1:10),
        ntrees = c(100, 250, 500, 1000, 2000),
        trControl = ctrl)
#random forest with a subset of predictors
fit_caret_rf_subset_select <- 
  train(Country.of.Origin ~ Aroma + Flavor + Aftertaste +
          Acidity + Sweetness + Body + Uniformity + Clean.Cup +
          Total.Cup.Points + altitude_mean_meters,
        data = dataused,
        method = "rf",
        na.action = na.roughfix,
        tuneGrid = expand.grid(.mtry = c(1:9)),
        nodesize = c(1:10),
        ntrees = c(100, 250, 500, 1000, 2000),
        trControl = ctrl)


#find the maximum accuracy of each model
max(fit_caret_rf_select$results$Accuracy)
max(fit_caret_rf_subset_select$results$Accuracy)

#create variable importance plot for the selected model
varImpPlot(fit_caret_rf_select$finalModel, n.var = 15)

fit_caret_rf_select$bestTune

##### model assessment OUTER shell #####
# produce loops for 5-fold cross-validation for model ASSESSMENT
n = nrow(joined_data)  # number of data points
nfolds_outer = 5  # number of fold in outer CV
set.seed(88)
groups = rep(1:nfolds_outer,length=n)  #produces list of group labels
cvgroups = sample(groups,n)  #orders randomly

# set up storage for outer 5-fold cross-validation for model ASSESSMENT
cv_pred = factor(vector(length=n), 
                 levels=levels(joined_data$Country.of.Origin)) # storage for output from train function
allbesttrain = list(rep(NA,nfolds_outer))  # storage for output from train function
allpredicted_outer = rep(NA,n)   # storage for honest predictions
ncols = ncol(joined_data)
# set up training method for inner CV
ctrl = trainControl(method="cv", number = 5)

for (jj in 1: nfolds_outer) {    #  we will use jj for the outer loop index
  in_train_outer = (cvgroups != jj)     # logical vector indicating NOT group jj
  in_test_outer = (cvgroups == jj)     # logical vector indicating group jj
  
  train_set_outer = joined_data[in_train_outer, ]  # all data EXCEPT for group jj
  test_set_outer = joined_data[in_test_outer, ]   # data in group jj
  
  #specify data to be used - only use data from outer-loop training set
  dataused = train_set_outer
  ####################### ENTIRE MODEL-SELECTION PROCESS ########################
  ################ Step 1. ##############
  # KNN with all predictors
  kvals = c(1:(ncol(joined_data)-1))
  
  # random forest with all predictors
  fit_caret_rf_select_2 <-
    train(Country.of.Origin ~ .,
          data = dataused,
          method = "rf",
          na.action = na.roughfix,
          tuneGrid = expand.grid(.mtry = c(1:(ncols -1 ))),
          nodesize = c(1:10),
          ntrees = c(100, 250, 500, 1000, 2000),
          trControl = ctrl)
  #random forest with a subset of predictors
  fit_caret_rf_subset_select_2 <- 
    train(Country.of.Origin ~ Aroma + Flavor + Aftertaste +
            Acidity + Sweetness + Body + Uniformity + Clean.Cup +
            Total.Cup.Points + altitude_mean_meters,
          data = dataused,
          method = "rf",
          na.action = na.roughfix,
          tuneGrid = expand.grid(.mtry = c(1:9)),
          nodesize = c(1:10),
          ntrees = c(100, 250, 500, 1000, 2000),
          trControl = ctrl)
  
  
  ################ Step 2. ##############
  all_best_accuracies <- c(max(fit_caret_rf_select_2$results$Accuracy),
                           max(fit_caret_rf_subset_select_2$results$Accuracy))
  
  all_train_output <- list(fit_caret_rf_select_2, fit_caret_rf_subset_select_2)
  
  bestmodel_train_output <- all_train_output[[which.max(all_best_accuracies)]]
  
  print(paste("jj =", jj,
              which.max(all_best_accuracies),
              bestmodel_train_output$method))
  # track best output in a list
  allbesttrain[[jj]] = bestmodel_train_output$method
  
  # predict test set in outer loop
  allpredicted_outer[in_test_outer] <-
    bestmodel_train_output |> predict(test_set_outer)
}

conf_mat = table(allpredicted_outer, joined_data$Country.of.Origin)
conf_mat
sum(diag(conf_mat))/n  # This is the most honest 
# assessment of accuracy
#bestmodel_train_output$finalModel

#use the best model"s final model predicted results to talk about model 

fit_caret_rf_select$finalModel
pred_coffee <-  fit_caret_rf_select$finalModel$predicted

#assessment
table(fit_caret_rf_select$finalModel$predicted,joined_data$Country.of.Origin)
Accuracy_valid = sum(pred_coffee == joined_data$Country.of.Origin)/n
Accuracy_valid


########### hierarachial clustering ############

#hierarchical clustering
#this section prepares the data by scaling and selecting appropriate values

x_select <- joined_data_no_ind %>% select(Aroma, Flavor, Aftertaste, Acidity, Sweetness, Body, Uniformity, Clean.Cup, Total.Cup.Points)
n_all <- dim(x_select)[1]
p_select <- dim(x_select)[2]
x_select_scaled <- scale(x_select)
dist.x_select_scaled <- dist(x_select_scaled,method="euclidean")
#dist.x_select_scaled

# create two plots showing the diminishing gains in differences between clusters
# Compute and plot wss for k = 2 to k = 15.
k.max <- 50
# data <- x_all_scaled
data <- x_select_scaled
# wss <- sapply(1:k.max, 
#               function(k){kmeans(data, k, nstart=50,iter.max = 50 )$tot.withinss})
# wss
# plot(1:k.max, wss,
#      type="b", pch = 19, frame = FALSE, 
#      xlab="Number of clusters K",
#      ylab="Total within-clusters sum of squares")

#run with caution - takes ~4 minutes on highend m3 macbook pro, can adjust the max.nc downwards if needed
nb <- NbClust(data, diss=NULL, distance = "euclidean", 
              min.nc=2, max.nc=30, method = "kmeans", 
              index = "all", alphaBeale = 0.1)
hist(nb$Best.nc[1,], breaks = max(na.omit(nb$Best.nc[1,])))

n = nrow(joined_data)  # number of data points
hc.fit = hclust(dist.x_select_scaled,method="complete")
linktype = "Complete Linkage, Euclidean Distance, p=10"

# hc.fit = hclust(dist.x_all_scaled, method="complete")
# linktype = "Complete Linkage, Euclidean Distance, p=14"
# distance at which merge via complete linkage occurs
hc.4321 = hc.fit$height[(n-4):(n-1)]
hc.avg = (hc.fit$height[(n-3):(n-1)]+hc.fit$height[(n-4):(n-2)])/2
# obtaining cluster labels
hc.fit$height[(n-4):(n-1)]
nclust=3 #nclust comes from the above code section
htclust = mean(hc.fit$height[(n-2):(n-1)])
membclust = cutree(hc.fit,k=nclust) # cutree(hc.fit,h = htclust)

dend.form = as.dendrogram(hc.fit)
dend.merge <- ggdendrogram(dend.form, rotate = F,labels=F) + 
  labs(title = linktype) +
  geom_hline(yintercept=hc.4321, linetype="dashed", 
             color = c("red","blue","gold3","gray"))  
dend.merge
dend.merge +
  geom_hline(yintercept=hc.avg, size = 2,
             color = c(rgb(.5,0,1,0.5),rgb(.5,1,0,0.5),rgb(.5,.5,.2,0.5))) 

membHier = cutree(hc.fit,k=nclust)
colused <- c("turquoise3", "red", "black","pink", "brown")[membHier];
pchused <- c(0,3,5,8,16)[membHier]

joined_data %>%
  ggplot( aes(x=Aftertaste, y=Acidity)) +
  geom_point(color=colused, shape=pchused) +
  labs(title = "Lower scores for acidity tend to have lower aftertaste scores") + 
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# started to explore for additional insights, nothing obvious
# joined_data_hier <- cbind(joined_data, membHier)
# table(joined_data_hier$membHier, orig_data$Variety)

