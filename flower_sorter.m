

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

imageSize = [250,320,3];
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, 'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, 'ColorPreprocessing', 'gray2rgb');
 
layers = [ ...
     imageInputLayer([250,320,3])
     convolution2dLayer(5,20)
     maxPooling2dLayer(2,'Stride', 2)
     fullyConnectedLayer(5)
     softmaxLayer
     classificationLayer];
 
 options = trainingOptions(...
            'sgdm', 'MaxEpochs',20, ...
            'Plots', 'training-progress');
 
 trainedNet = trainNetwork(augmentedTrainingSet, layers, options);
 
 %{
trainCascadeObjectDetector("daisyDetector.xml", imds.Labels(1), 

detector = vision.CascadeObjectDetector(
 
 
 %}