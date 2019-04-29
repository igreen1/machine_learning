
layers = [ ...
     imageInputLayer([32,32,3])
     convolution2dLayer(5,20)
     maxPooling2dLayer(2,'Stride', 2)
     fullyConnectedLayer(10)
     softmaxLayer
     classificationLayer];

 options = trainingOptions(...
            'sgdm', 'MaxEpochs',20, ...
            'Plots', 'training-progress');
 
 trainedNet = trainNetwork(train_images, layers, options);

 
 function data = loadmydata(filename)
    load(filename)
 end