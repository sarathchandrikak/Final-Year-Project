library('caret')
set.seed(1)
data<-read.csv("a.csv", header=T, na.strings=c("","NA"))
str(data)
summary(data)

NAProp <- (colSums(is.na(data))/nrow(data))*100
unwantedVars <- names(NAProp[NAProp > 50])

if(length(unwantedVars)>0){
  data <- setdiff(data, data[,unwantedVars])
}


data1 <- data.frame(lapply(data[,5:27], as.numeric, stringsAsFactors = FALSE))
data_mld = data1
data1$LD <- as.factor(data1$LD)
data1$LD <- factor(data1$LD,levels = c(1,2),labels = c("no", "yes"))
data1$Dyslexia <- as.factor(data1$Dyslexia)
data1$Dyslexia <- factor(data1$Dyslexia,levels = c(1,2),labels = c("no", "yes"))
data1$Dysgraphia <- as.factor(data1$Dysgraphia)
data1$Dysgraphia <- factor(data1$Dysgraphia,levels = c(1,2),labels = c("no", "yes"))
data1$Dyscalculia <- as.factor(data1$Dyscalculia)
data1$Dyscalculia <- factor(data1$Dyscalculia,levels = c(1,2),labels = c("no", "yes"))
data1$ADHD <- as.factor(data1$ADHD)
data1$ADHD <- factor(data1$ADHD,levels = c(1,2),labels = c("no", "yes"))

levels(data1$Dyslexia) <- c(FALSE,TRUE)
data1$Dyslexia <- as.logical(data1$Dyslexia)

levels(data1$Dysgraphia) <- c(FALSE,TRUE)
data1$Dysgraphia <- as.logical(data1$Dysgraphia)

levels(data1$Dyscalculia) <- c(FALSE,TRUE)
data1$Dyscalculia <- as.logical(data1$Dyscalculia)

levels(data1$ADHD) <- c(FALSE,TRUE)
data1$ADHD <- as.logical(data1$ADHD)

index <- createDataPartition(data1$LD, p=0.75, list=FALSE)

trainSet <- data1[index,]
testSet <- data1[-index,]
train.set=as.integer(row.names(trainSet))

labels <- c("Dyslexia","Dysgraphia","Dyscalculia","ADHD")

library(mlr)
data.task <- makeMultilabelTask(id='multi', data = data1, target = labels)

#----- Problem transformation ------#


#lrn.core= makeLearner("classif.rpart", predict.type = "prob")
#lrn.knn= makeLearner("classif.kknn", predict.type = "prob")
lrn.nb= makeLearner("classif.naiveBayes", predict.type = "prob")
lrn.log= makeLearner("classif.logreg", predict.type = "prob")
#lrn.c= makeLearners(c("rpart", "lda"), type = "classif", predict.type = "prob")

# Wrapped learners

lrn.binrel=makeMultilabelBinaryRelevanceWrapper(lrn.nb)
lrn.chain=makeMultilabelClassifierChainsWrapper(lrn.log)

mod.binrel=mlr::train(lrn.binrel,data.task,subset=train.set)
mod.chain=mlr::train(lrn.chain,data.task,subset=train.set)

pred.binrel=predict(mod.binrel,newdata=testSet)
pred.chain=predict(mod.chain,newdata=testSet)
measures=list(multilabel.acc,multilabel.f1,multilabel.hamloss,multilabel.subset01,multilabel.ppv,multilabel.tpr)

#rdesc = makeResampleDesc("Subsample", iters = 10, split = 2/3)
#r = resample(lrn.chain, data.task, rdesc, measures = multilabel.acc)

p1=performance(pred.binrel,measures)
p2=performance(pred.chain,measures)


#--------- LP ---------#

#library(mldr)
#library(utiml)
#lpmldr <- mldr_from_dataframe(data1, labelIndices = c(20,21,22,23))
#emo_lp <- mldr_transform(lpmldr, "LP")
#library(RWeka)
#classifier <- IBk(classLabel ~ ., data = emo_lp, control = Weka_control(K = 10))
#lp <- evaluate_Weka_classifier(classifier, numFolds = 5)

library(mldr)
library(utiml)
library(C50)
data_mld = read.csv("data_mld.csv", header = TRUE, na.strings=c("","NA"))
data_mld = data_mld[,5:27]
lpmldr <- mldr_from_dataframe(data_mld, labelIndices = c(20,21,22,23))
m <-  c("accuracy", "F1", "hamming-loss","subset-accuracy","precision","recall")
lp<- cv(lpmldr, method="lp", base.algorithm = "C5.0", cv.folds=10, cv.sampling="stratified", cv.measures=m, cv.seed=123)

#-------- Adaption Algorithms --------#

library(randomForestSRC)
set.seed(2018)
lrn.rfsrc = makeLearner("multilabel.randomForestSRC",predict.type = "prob")
mod.rf=mlr::train(lrn.rfsrc,data.task,subset=train.set)
pred.rf=predict(mod.rf,newdata=testSet)
a1 <- performance(pred.rf, measures = measures)

#-------- RAkEL --------------#

library(mldr)
library(utiml)
data_mld = read.csv("data_mld.csv", header = TRUE, na.strings=c("","NA"))
data_mld = data_mld[,5:27]
rkmldr <- mldr_from_dataframe(data_mld, labelIndices = c(20,21,22,23))
m <-  c("accuracy", "F1", "hamming-loss","subset-accuracy","precision","recall")
rakel <- cv(rkmldr, method="rakel", base.algorith="SVM", cv.folds=10, cv.sampling="stratified", cv.measures=m, cv.seed=123)

#cv.folds = 10
#cv.sampling = "stratified"
#cvdata <- create_kfold_partition(rkmldr, cv.folds, cv.sampling)
#results <- parallel::mclapply(seq(cv.folds), function (k)
#{
#  ds <- partition_fold(cvdata, k)
#  model <- do.call("rakel", c(list(mdata=ds$train, base.algorithm = getOption("utiml.base.algorithm", "SVM"))))
#  pred <- predict(model, ds$test)
#  return
#}
#)

#t(results$multilabel)
#round(sapply(results$labels, colMeans), 4)
#-------- Ensemble Stacking ---------#

#1. BR base learner

train.task = subsetTask(data.task, subset = train.set)
rdesc.stack = makeResampleDesc("CV", iters = 10)
rbase1 = resample(lrn.binrel,train.task, rdesc.stack)
b11 = as.numeric(rbase1$pred$data[order(rbase1$pred$data$id), ]$response.Dyslexia)
b12 = as.numeric(rbase1$pred$data[order(rbase1$pred$data$id), ]$response.Dysgraphia)
b13 = as.numeric(rbase1$pred$data[order(rbase1$pred$data$id), ]$response.Dyscalculia)
b14 = as.numeric(rbase1$pred$data[order(rbase1$pred$data$id), ]$response.ADHD)

rbase2 = resample(lrn.chain,train.task, rdesc.stack)
b21 = as.numeric(rbase2$pred$data[order(rbase2$pred$data$id), ]$response.Dyslexia)
b22 = as.numeric(rbase2$pred$data[order(rbase2$pred$data$id), ]$response.Dysgraphia)
b23 = as.numeric(rbase2$pred$data[order(rbase2$pred$data$id), ]$response.Dyscalculia)
b24 = as.numeric(rbase2$pred$data[order(rbase2$pred$data$id), ]$response.ADHD)

df <- data.frame(b11,b12,b13,b14,b21,b22,b23,b24,trainSet$Dyslexia,trainSet$Dysgraphia,trainSet$Dyscalculia,trainSet$ADHD)
names(df)[names(df) == "trainSet.Dyslexia"] <- "Dyslexia"
names(df)[names(df) == "trainSet.Dysgraphia"] <- "Dysgraphia"
names(df)[names(df) == "trainSet.Dyscalculia"] <- "Dyscalculia"
names(df)[names(df) == "trainSet.ADHD"] <- "ADHD"

emldr <- mldr_from_dataframe(df, labelIndices = c(9,10,11,12))
etransform <- mldr_transform(emldr, "LP")
top <- c("b11","b12","b13","b14","b21","b22","b23","b24")
label <- c("classLabel")
fitControl <- trainControl(method = "cv", number = 5, savePredictions = 'final', classProbs = TRUE)
model.ensmble<-train(etransform[,top],etransform[,label],method='rpart',trControl=fitControl,tuneLength=7)
model_glm<-train(classLabel~., data=etransform, method="rpart",trControl=fitControl, tuneLength=7)
#####- Learning part #######
lrn.rpart= makeLearner("classif.rpart", predict.type = "prob")
lrn.ensemble = makeMultilabelClassifierChainsWrapper(lrn.rpart)
mod.ensemble=mlr::train(lrn.ensemble,data.task,subset=1:259)
pred.ensemble=predict(mod.ensemble,newdata=testSet)
e1 <- performance(pred.ensemble, measures)


bar1<-c("BR","CC","Adapt","Ensemble")
bar2<-c(p1[1],p2[1],a1[1],e1[1])
cols <- c("red","lightblue","green","yellow")
barplot(bar2,names.arg = bar1,xlab="Algorithms",ylab="Accuracy",col=cols, ylim = c(0,1), main="Comparision of algorithms")

p1 <- c(0.85000000,0.87837535,0.09411765,0.28235294,0.85451977,0.92937853)
p2 <- c(0.88333333,0.91154062,0.07941176,0.23529412,0.90163934,0.90395480)
a1 <- c(0.92156863,0.93988796,0.05294118,0.15294118,0.94632768,0.92796610)
e1 <- c(0.92745098,0.9324356,0.059321,0.18860,0.93634,0.92456)
#lp <- c(0.66831933,0.69160344,0.06102941,0.81126050,0.70168768,0.69589636)
names(p1) <- c("multilabel.acc","multilabel.f1","multilabel.hamloss","multilabel.subset01","multilabel.ppv","multilabel.tpr")
names(p2) <- c("multilabel.acc","multilabel.f1","multilabel.hamloss","multilabel.subset01","multilabel.ppv","multilabel.tpr")
names(a1) <- c("multilabel.acc","multilabel.f1","multilabel.hamloss","multilabel.subset01","multilabel.ppv","multilabel.tpr")
names(e1) <- c("multilabel.acc","multilabel.f1","multilabel.hamloss","multilabel.subset01","multilabel.ppv","multilabel.tpr")
names(rakel) <- c("multilabel.acc","multilabel.f1","multilabel.hamloss","multilabel.subset01","multilabel.ppv","multilabel.tpr")
names(lp) <- c("multilabel.acc","multilabel.f1","multilabel.hamloss","multilabel.subset01","multilabel.ppv","multilabel.tpr")

performance=as.data.frame(rbind(p1,p2,a1,e1,rakel,lp))
performance$model=c("BR","Chains","Adaption","Stacking","RAkEL","LP")
library(RColorBrewer)
library(tidyr)
plong=gather(performance,metrics,value,multilabel.acc:multilabel.tpr, factor_key=TRUE)
ggplot(plong)+geom_tile(aes(x=model,y=metrics,fill=value),color="black")+geom_text(aes(x=model,y=metrics,label=round(value,3)),color="black")+scale_fill_distiller(palette = "Spectral")

plot(rkmldr, type = "LH")
plot(rkmldr, type = "LC")
plot(rkmldr, type = "LB")
plot(rkmldr, type = "CH")
plot(rkmldr, type = "LSB")
plot(rkmldr, type = "AT")
plot(rkmldr, type = "LSH")