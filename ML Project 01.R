#Clean Global Environment
rm(list = ls())

#Loading relevant packages (Installation needed if package not exist in running computer)
library(plyr)
library(ggplot2)
library(data.table)
library(lubridate)
library(Hmisc) #For Lag function
library(zoo)#For rollmeanr
library(hydroGOF)#For MSE calculation
library(class)#For knn model
library(nnet)#For Neural-Network model
library(rpart)#For Decision Tree model
library(rattle)#For Decision Tree model
library(rpart.plot)#For Decision Tree model
library(RColorBrewer)#For Decision Tree model
library(e1071)#For SVM model
library(ROCR)#For ROC curves
library(RCurl)#For nnet plotting

#Loading data
rawcsv<-read.csv("Israel 2.5.csv", check.names=TRUE, na.strings=c("","NA"))
str(rawcsv)
#Adding Column of Date derived from the time column
rawcsv$Date <- as.Date(rawcsv$time)

#removing Foreshocks & Aftershocks - (extracting only maximum values from each date)
csv <- as.data.table(rawcsv)
csv<-csv[csv[, .I[which.max(mag)], by=Date]$V1]

#Defining a threshold value (median of all years maximums)
threshold<-median(tapply(csv$mag,year(csv$Date),max))
threshold
############ Exploring the data (basic exploration - deeper EDA can be applied)################
csv<-csv[,c("Date","latitude","longitude","depth","mag","magType")]
summary(csv)
hist(csv$depth, main = "Depth Histogram", xlab = "Depth (km)") #Depth
gplot<-ggplot(csv, aes(x=mag)) + ggtitle("Magnitude density curve")
gplot + labs(x="Magnitude (Richter)") + geom_density() #Magnitude
ggplot(csv, aes(x=year(csv$Date)))+ggtitle("Year distribution") + 
  xlab("Year")+theme(text = element_text(size=20))+geom_bar()#Year of date


#calculating average magnitude per year + graph
avg_mag_ByYear<-setnames(aggregate(csv$mag, list(year(csv$Date)), mean),c("Year","Average.Magnitude"))
ggplot(avg_mag_ByYear, aes(x=Year,y=Average.Magnitude)) + geom_point() + ggtitle("Mean Magnitude per year")

#**********************************************Adding Variables (columns)********************************************************************
#Creating Target-Feature - maximum of last year (360 days) of observations called "maxLastYear"
for (i  in 1:nrow(csv)){ # i=1
  csv$maxLastYear[i]=max(csv[csv$Date>(csv$Date[i]-360) & csv$Date<=csv$Date[i],"mag"])
}
#Creating INDICATORS for the models
#Calculating time difference between every 50 events
csv<-mutate(csv,"TimeDiff"=Date - Lag(Date,50)) 
csv$TimeDiff<-as.integer(csv$TimeDiff)#convert outcome as numeric

#Calculating Magnitude average for last 50 events
csv<-mutate(csv,"MagAvg"=rollmeanr(mag,50,fill=NA))

# Calculating The rate of square root of seismic energy released (dE1/2) for last 50 events
E<-10^(11.8+(1.5*csv$mag)) #The Energy calculation formula
csv<-mutate(csv,"dE0.5"=(rollapplyr(E,50,sum,fill=NA))/as.numeric(csv$TimeDiff))

#Calculating the a&b coeffecients of the Gutenberg-Richter inverse power law
##log of total observations of observed mag and above in the last 50 events
csv<-mutate(csv,"log10N(M)"=rollapplyr(mag,50,function(i){log10(sum(i>=tail(i,1)))},fill=NA))
csv<-csv[complete.cases(csv)==TRUE,] #filter to have only complete observations (no NA) in order to be able to apply coefficients extraction 
Coefa<-function(df){coef(lm(`log10N(M)`~mag, as.data.frame(df)))[1]}
Coefb<-function(df){coef(lm(`log10N(M)`~mag, as.data.frame(df)))[2]}
csv<-mutate(csv,"a"=rollapplyr(csv[,c("mag","log10N(M)")],50,Coefa,fill=NA,by.column = F))
csv<-mutate(csv,"b"=rollapplyr(csv[,c("mag","log10N(M)")],50,Coefb,fill=NA,by.column = F))

#Calculating ni (MSE of the created regression line)
csv<-mutate(csv,"a-(b*mag)"=a-(b*mag))
ni<-function(df){mse(df[,"log10N(M)"],df[,"a-(b*mag)"])}
csv<-mutate(csv,"ni"=rollapplyr(csv[,c("log10N(M)","a-(b*mag)")],50,ni,fill=NA,by.column = F))

#Calculating Magnitude deficit (a/b)
csv<-mutate(csv,"ab.Ratio"=a/b)
#********************** Transformations ***************************
hist(csv$TimeDiff, main = "TimeDiff Histogram", xlab = "TimeDiff (days)") #TimeDiff
gplot<-ggplot(csv, aes(x=MagAvg)) + ggtitle("Average Magnitude density curve")
gplot + labs(x="Average Magnitude (Richter)") + geom_density() #Avg Magnitude
boxplot(csv$dE0.5, main = "Siesmic Energy released Histogram", xlab = "dE^0.5", horizontal = T) #Energy
gplot1<-ggplot(csv, aes(x=b)) + ggtitle("b coefficient density curve")
gplot1 + labs(x="b value") + geom_density() #b coefficient
gplot2<-ggplot(csv, aes(x=ni)) + ggtitle("ni density curve")
gplot2 + labs(x="ni value") + geom_density() #ni value
gplot3<-ggplot(csv, aes(x=ab.Ratio)) + ggtitle("a/b Ratio")
gplot3 + labs(x="a/b Ratio") + geom_density() #a/b Ratio

#Log-Transformation and standardisation to some indicators for better models
csv[,"TimeDiff"] <- sapply(csv[,"TimeDiff"], log)
csv[,"ab.Ratio"] <- sapply(csv[,"ab.Ratio"], function(x) abs(x)^(1/3))
csv[,"ni"] <- sapply(csv[,"ni"], log)
csv[,"dE0.5"] <- sapply(csv[,"dE0.5"], function(x) (x-min(x))/(max(x)-min(x)))

#graphics of transformed data
hist(csv$TimeDiff, main = "TimeDiff Histogram", xlab = "TimeDiff (days)") #TimeDiff
gplot<-ggplot(csv, aes(x=ab.Ratio)) + ggtitle("ab.Ratio")
gplot + labs(x="ab.Ratio") + geom_density() #ab.Ratio
boxplot(csv$dE0.5, main = "Siesmic Energy released Histogram", xlab = "dE^0.5", horizontal = T) #Energy
gplot2<-ggplot(csv, aes(x=ni)) + ggtitle("ni density curve")
gplot2 + labs(x="ni value") + geom_density() #ni value


#**********************************************Models****************************************************************************************
#Defining AboveMedian column as factor (True=1, False=0)
csv<-mutate(csv, "AboveMedian"=as.factor(ifelse(maxLastYear>=threshold,1,0)))
csv<-csv[complete.cases(csv)==TRUE,]#filtering all non-completed observations
csv<-csv[,-c("latitude", "longitude","magType")]
sum(csv$AboveMedian==1)/nrow(csv) #Checking % of TRUE AboveMedian values


#Arange sets and correlations for testing
train_set <- csv[1:(nrow(csv)*(2/3)),]#Defining training set (2/3 of events by appearance order)
test_set <- csv[(nrow(csv)*(2/3)):nrow(csv),]#Defining test set (the last 1/3 events)
target_feature <- "AboveMedian"
fmla <- as.formula(paste(target_feature,"~TimeDiff+dE0.5+b+ni+ab.Ratio+MagAvg",sep=""))
#fmla2 <- as.formula(paste(target_feature,"~depth+TimeDiff+dE0.5+b+ni+ab.Ratio+MagAvg",sep=""))
numeric_features <- sapply(csv, is.numeric) #assign variable to all numeric features



#Simple Decision Tree
fit <- rpart(fmla,data=train_set, cp=0.03, method = 'class')
fancyRpartPlot(fit, cex=0.7,tweak=1, sub = "")
title("Decision Tree (gini-based)",line = 3.25)
DT_pred<-predict(fit,test_set, type = "prob")
DT_predClass<-predict(fit,test_set, type = "class")
confusion_matrix_DT <- table(test_set$AboveMedian,DT_predClass)
confusion_matrix_DT
sum(diag(confusion_matrix_DT))/sum(confusion_matrix_DT) # % of true cases


#knn on all numeric features
numeric_features
target_feature
#knn execution k=3 found to be most usefull
knn_3 <- knn(train = train_set[,..numeric_features], test = test_set[,..numeric_features], cl = train_set$AboveMedian, k=3, prob = T)
knn_3_prob<-attr(knn_3, "prob")
knn_3_prob <- 2*ifelse(knn_3 == "-1", 1-knn_3_prob, knn_3_prob) - 1
#confusion matrix of the created knn and its % of corrected predictions
confusion_matrix_knn_3 <- table(test_set$AboveMedian,knn_3)
confusion_matrix_knn_3
sum(diag(confusion_matrix_knn_3))/sum(confusion_matrix_knn_3) # % of true cases
#plotting the model with the test set
plot(train_set[,c("TimeDiff","MagAvg")],
     col=c(4,3)[train_set$AboveMedian], main = "K-Nearest Neighbors")
points(test_set[,c("TimeDiff","MagAvg")], bg=c(4,3)[as.numeric(knn_3)],
       pch=c(21,21)[as.numeric(knn_3)],cex=1.2, col=grey(0))
legend("bottomright", pch = c(1,1,21,21), col=c(4,3), pt.bg=c(4,3), 
       legend = c("Training-0", "Training-1", "Test-0", "Test-1"), bty = "n")
       
# nnet
#nnet model
nnet_model <- nnet(fmla, data = train_set, size = 2 ,maxit=1000, decay=5e-1)
#nnet prediction on test data
nnet_predClass <- predict(nnet_model,test_set, type = "class")
nnet_pred <- predict(nnet_model,test_set, type = "raw")
#confusion matrix of the created nnet and its % of corrected predictions
confusion_matrix_nnet <- table(nnet_predClass,test_set$AboveMedian)
confusion_matrix_nnet
sum(diag(confusion_matrix_nnet))/sum(confusion_matrix_nnet)
#Plotting the model
#import the function from Github
library(devtools)
source_url('https://gist.githubusercontent.com/fawda123/7471137/raw/466c1474d0a505ff044412703516c34f1a4684a5/nnet_plot_update.r')
par(mar=c(0,0,1,1))
plot.nnet(nnet_model, main = "Neural Network", y.lab = "")
mtext("AboveMedian", side=4, line=-4, cex.lab=2,las=3)

# SVM
numeric_features
#SVM model and testing
SVM_model <- svm(fmla,train_set, kernel="radial", scale = F)
SVM_Pred <- predict(SVM_model,test_set, type = "class", decision.values = T)
svm.probs<-attr(SVM_Pred,"decision.values")
SVM_class <- predict(SVM_model,test_set, type = "class")
#confusion matrix of the created SVM and its % of corrected predictions
confusion_matrix_SVM <- table(test_set$AboveMedian,SVM_Pred)
confusion_matrix_SVM
sum(diag(confusion_matrix_SVM))/sum(confusion_matrix_SVM)
par(mar=c(5.1, 4.1, 4.1, 2.1))
plot(SVM_model, train_set, MagAvg ~ TimeDiff, 
     slice=list(dE0.5=0.0164983, b=-0.6761, ni=3.225, ab.Ratio=1.791))

#*********************************Models Evaluation********************************************
#F-Measure
treePrecision<-confusion_matrix_DT[2,2]/sum(confusion_matrix_DT[,2])
treeRecall<-confusion_matrix_DT[2,2]/sum(confusion_matrix_DT[2,])
FDT<-(2*treePrecision*treeRecall/(treeRecall+treePrecision))
FDT

knnPrecision<-confusion_matrix_knn_3[2,2]/sum(confusion_matrix_knn_3[,2])
knnRecall<-confusion_matrix_knn_3[2,2]/sum(confusion_matrix_knn_3[2,])
Fknn<-(2*knnPrecision*knnRecall/(knnRecall+knnPrecision))
Fknn

nnetPrecision<-confusion_matrix_nnet[2,2]/sum(confusion_matrix_nnet[,2])
nnetRecall<-confusion_matrix_nnet[2,2]/sum(confusion_matrix_nnet[2,])
Fnnet<-(2*nnetPrecision*nnetRecall/(nnetRecall+nnetPrecision))
Fnnet

SVMPrecision<-confusion_matrix_SVM[2,2]/sum(confusion_matrix_SVM[,2])
SVMRecall<-confusion_matrix_SVM[2,2]/sum(confusion_matrix_SVM[2,])
FSVM<-(2*SVMPrecision*SVMRecall/(SVMRecall+SVMPrecision))
FSVM

FTable<-matrix(c(FDT, Fknn, Fnnet, FSVM), ncol = 4)
colnames(FTable)<-c("Decision Tree", "KNN", "NNET", "SVM")
rownames(FTable)<-"F-Measure"
FTable

#ROC
SVM_Predict <- prediction(as.numeric(svm.probs),as.numeric(test_set$AboveMedian))
nnet_Pred <- prediction(as.numeric(nnet_pred),as.numeric(test_set$AboveMedian))
knn3_Pred <- prediction(as.numeric(knn_3_prob),as.numeric(test_set$AboveMedian))
DT_Pred <- prediction(as.numeric(DT_pred[,2]),as.numeric(test_set$AboveMedian))
roc_SVM<-performance(SVM_Predict,"tpr","fpr")
roc_nnet<-performance(nnet_Pred,"tpr","fpr")
roc_knn3<-performance(knn3_Pred,"tpr","fpr")
roc_DT<-performance(DT_Pred,"tpr","fpr")

roc.tot<-plot(roc_SVM, lwd=2, main="ROC curve of the models")
roc.tot<-plot(roc_nnet, add=T, col=2, lwd=2)
roc.tot<-plot(roc_knn3, add=T, col=3, lwd=2)
roc.tot<-plot(roc_DT, add=T, col=4, lwd=2)
legend("bottomright",legend = c("SVM", "NNET", "KNN","Decision Tree"),fill = 1:4)

#----------------------------------------------------------------------------------