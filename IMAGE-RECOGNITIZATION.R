#IMAGE RECOGNITION IN R

#to install EBImage we nee to install: 
##install.packages("BiocManager") 
##BiocManager::install("EBImage")
library(EBImage)
library(keras)
library(tensorflow)

##set working directory
setwd('C:/Users/revathi pandian/Desktop/image recognition')
pics<-c('c1.jpg','c2.jpg','c3.jpg','c4.jpg','c5.jpg','c6.jpg','p1.jpg','p2.jpg','p3.jpg','p4.jpg','p5.jpg','p6.jpg')
mypic<-list()
for(i in 1:12){mypic[[i]]<-readImage(pics[i])} ##the images in pics will be stored in mypics using for loop all 12 images will read

##explore
print(mypic[[1]])  ##it gives you some values for the pic 1
display(mypic[[1]])
summary(mypic[[1]])
hist(mypic[[8]]) ## it gives you the plot for the 8th pic
str(mypic)


##convert all the pic to same dimension
##resize
for(i in 1:12){mypic[[i]]<-resize(mypic[[i]],28,28)} ##converting the higher dim of pic to smaller dim i.e 28
str(mypic)

##reshape
##for training purpose we should convert into matrix 
for(i in 1:12){mypic[[i]]<-array_reshape(mypic[[i]],c(28,28,3))} ##28,28,3 is resized dim taken from str(mypic)
str(mypic)

##row bind
##is used to convert 12 different item to combine into one 
trainx<-NULL
for(i in 1:5){trainx<-rbind(trainx,mypic[[i]])}##1:5 is the pic of car we use first 5 for train 6th pic for test similarly for plane
str(trainx)
trainx<-NULL
for(i in 7:11){trainx<-rbind(trainx,mypic[[i]])}##1:5 is the pic of car we use first 5 for train 6th pic for test similarly for plane
str(trainx)
testx<-rbind(mypic[[6]],mypic[[12]])  
trainy<-c(0,0,0,0,0,1,1,1,1,1)  ##car is represented as 0 and plane is represented as 1
testy<-c(0,1) ## since in testx we have nly 2 pic one is car and other one is plane

##one hot encoding
trainLabels<-to_categorical(trainy)
testLabels<-to_categorical(testy)
##model
model<-keras_model_sequential()
##creating neural network,
##layer_dense is hidden layer,from str(trainx) we can find the input_shape,
##last dense_layer is the output layer since we have 0 and 1 so we used units as 2
model%>%
  layer_dense(units=256,activation='relu',input_shape=c(2352))%>%
  layer_dense(units=128,activation='relu')%>%
  layer_dense(units=2,activation='softmax')
summary(model)

##(2352*256)+256=602112
#(602112*128)+128=32768
#(32768*2)+2=

##complie
model%>%
  compile(loss='binary_crossentropy',
          optimizer=optimizer_rmsprop(),
          metrics=c('accuracy'))

##fit model  ##20%used for validation and 80% for train
history<-model %>%
        fit(trainx,
            trainLabels,
            epochs=30,
            batch_size=32,
            validation_split=0.2)

##evaluation & prediction-train data 
#accuracy 0.9 means 9 pic has predicted properly one pic has miss classified
model %>% evaluate(trainx,trainLabels)
pred<-model %>% predict_classes(trainx)

##confusion matrix
table(Predicted=pred,Actual=trainy)

##find probability 
prob<-model %>% predict_proba(trainx)

##to see which is predicted wrong
cbind(prob,Predicted=pred,Actual=trainy)

##evaluation and predict test data
model %>%  evaluate(testx,testLabels)
pred<-model %>% predict_classes(testx)
table(Predicted=pred,Actual=testy)
