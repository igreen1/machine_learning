clear all
close all


%apparently, this is just within matlab ! yay, makes it easier
digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');


numTrainFiles = 750;
[imdsTrain,testds] = splitEachLabel(imds,numTrainFiles,'randomize');

layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer %??? idk, matlab answers said to put this in and it fixed a lot
    reluLayer
    
    fullyConnectedLayer(10) %number of catergories: 10 digits
    softmaxLayer
    classificationLayer];
 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    ...%'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
 
 trainedNet = trainNetwork(imdsTrain, layers, options);
 
 digitNet = trainedNet
%predictionss = classify(net, testds)
 save digitNet
 