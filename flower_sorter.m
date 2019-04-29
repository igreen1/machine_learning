

%%get the image folder (easy peasy)
imageFolder = fullfile("flower_photos")

imds = imageDatastore(imageFolder, 'LabelSource', 'foldernames', 'IncludeSubfolders',true);


%find number of images for each flower
amount_imgs_per_flower = countEachLabel(imds);

minImagesCount = min(amount_imgs_per_flower{:,2});

%now, lets get make the amount of images uniform for each dataset
imds = splitEachLabel(imds, minImagesCount, 'randomize');
 %for this data set, 633/flower
 
%because we have so many, we can make a trainingset and a test set
[trainingSet, testSet] = splitEachLabel(imds, 0.3, 'randomize')

imageSize = [224,224,3];
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
 
layers = [
    imageInputLayer([224 224 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];
 
options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',4, ...
    'Shuffle','every-epoch', ...
    'ValidationData',imdsValidation, ...
    'ValidationFrequency',30, ...
    'Verbose',false, ...
    'Plots','training-progress');
 
 %{
trainCascadeObjectDetector("daisyDetector.xml", imds.Labels(1), 

detector = vision.CascadeObjectDetector(
 
 
 %}