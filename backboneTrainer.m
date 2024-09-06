clc
clear all
close all
warning off

tic

path = 'C:\Users\fta71\Desktop\BLS\BLS_DATA\notful\Training';


data = imageDatastore(path,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

% Resize images to 224x224
imageSize = [224, 224, 3];
data.ReadFcn = @(filename) imresize(imread(filename), imageSize(1:2));

[merchImagesTrain,merchImagesTest] = splitEachLabel(data,0.9,'randomized');

merchImagesTrain = shuffle(merchImagesTrain);

net = resnet50;

% resnet50;
% googlenet;
% xception;
% vgg16;
% efficientnetb0;
% densenet201

layersTransfer = net.Layers(1:end-3);

numClasses = 2;

newLearnableLayer = fullyConnectedLayer(numClasses, ...
'Name','fc1000', ...
'WeightLearnRateFactor',10, ...
'BiasLearnRateFactor',10);

newClassLayer = classificationLayer('Name','ClassificationLayer_fc1000');

layers = [...
    layersTransfer
    newLearnableLayer
    net.Layers(176)
    newClassLayer];

lgraph = layerGraph(layers);
nlgraph = lgraph;

for i = 1:176
    a = table2array(lgraph.Connections(i,1));
    b = table2array(lgraph.Connections(i,2));

    nlgraph = disconnectLayers(nlgraph,a{1},b{1});
end

lgraph = nlgraph;

for i = 1:192
    a = table2array(net.Connections(i,1));
    b = table2array(net.Connections(i,2));

    lgraph = connectLayers(lgraph,a{1},b{1});
end

options = trainingOptions('adam', ...
'InitialLearnRate',0.001, ...
'MaxEpochs',10, ...
'Shuffle','every-epoch', ...
'ValidationData',merchImagesTest, ...
'ValidationFrequency',40, ...
'MiniBatchSize',64, ...
'ExecutionEnvironment','gpu',...
'Verbose',true, ...
'Plots','training-progress');

classifier2 = trainNetwork(merchImagesTrain,lgraph,options);

save('featureExtractor.mat','classifier2');