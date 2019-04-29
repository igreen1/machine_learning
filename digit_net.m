%actually use the trained net
%apparently, this is just within matlab ! yay, makes it easier
clear all
close all

digitDatasetPath = fullfile(matlabroot,'toolbox','nnet','nndemos', ...
    'nndatasets','DigitDataset');
imds = imageDatastore(digitDatasetPath, ...
    'IncludeSubfolders',true,'LabelSource','foldernames');

numTrainFiles = 750;
[imdsTrain,testds] = splitEachLabel(imds,numTrainFiles,'randomize');


load digitNet
prediction = classify(digitNet, testds)
validation = testds.Labels;
accuracy = mean(prediction == validation)