%% Pretrained EfficientDet-D0 Network를 이용한 전이학습
% 아래에 이어지는 코드들은 객체인식을 위해 미리 학습된 EfficientDet-D0 네트워크의 전이학습을 수행하는 과정을 보여줍니다. 
% 아래 코드들은 "configureEfficientDetD0" 함수를 통해 미리 학습된 모델을 사용하는 것으로 커스텀 EfficientDet-D0 
% 네트워크를 구성합니다.
%% Setup
% 소스 디렉터리를 추가합니다.(데이터셋의 위치에 따라 해당 값을 조정해야 합니다.)

addpath('src');
%% Download Pretrained Model

model = helper.downloadPretrainedEfficientDetD0();
net = model.net;
%% Load Data
% 해당 코드에서는 DeepFashion2 데이터셋 중 validation의 이미지와 annos 데이터를 가져옵니다. Python에서와 
% 달리 matlab에서는 table 형태로 이미지의 위치와 annos 정보가 저장된 파일을 기반으로 데이터를 가져오기 때문에 데이터셋을 기반으로 
% .mat 파일을 생성해야 합니다. (별도 코드 참조)

data = load('DFvalTable.mat');
ClothDataset = data.CT;

% Add the full path to the local vehicle data folder.
% ClothDataset.imageFilename = fullfile(pwd, ClothDataset.imageFilename)
%%
image1= imread(ClothDataset.imageFilename{1});
imshow(image1)
%%
imDS = imageDatastore(ClothDataset.imageFilename);
boxDS = boxLabelDatastore(ClothDataset(:,2:end));
trainingData = combine(imDS,boxDS);
%%
% helper.validateInputData(trainingData);
%% Data Augmentation
% Data augmenation의 경우 원본 데이터를 학습과정에서 랜덤하게 변형시키는 것을 통해 학습 모델의 정확도를 향상시키는 방법입니다. 
% Data augmentation 사용을 통해 training 데이터셋의 샘플 수를 늘리지 않아도 학습 데이터의 다양성을 추가할 수 있습니다.
% 
% Use 'transform' function to apply custom data augmentations to the training 
% data. The 'augmentData' helper function, applies the following augmentations 
% to the input data.
% 
% 'transform'이라는 함수의 사용을 통해 학습 데이터에서 커스텀 data augmentation을 적용할 수 있습니다. 'helper.augmentData' 
% 함수를 통해 아래와 같은 augmentation들을 입력 데이터에 적용할 수 있습니다.
%% 
% * Color jitter augmentation in HSV space
% * Random horizontal flip
% * Random scaling by 10 percent

augmentedTrainingData = transform(trainingData, @helper.augmentData);

% Read the same image four times and display the augmented training data.
% Visualize the augmented images.
augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1,1}, 'Rectangle', data{1,2});
    reset(augmentedTrainingData);
end
figure
montage(augmentedData, 'BorderSize', 10)
%% Preprocess Training Data
% 위에서 진행된 augmented 학습 데이터를 학습 과정에 맞도록 preprocess하는 부분입니다.  'helper.preprocessTrainData 
% 함수를 통해 입력 데이터를 모델에 적합하도록 아래의 과정을 거치는 preprocessing을 하게 됩니다.
%% 
% * Normalizes input image using vgg mean and standard deviation.
% * Resizes the larger dimension of the input image to size 512.
% * Applies zero padding to the resized image to make its resolution [512 512]. 
% Input size of the pretrained network is [512 512 3]. Hence, all the input images 
% are preprocessed to have this size.
% * Rescales the bounding boxes as per the scale of resized image. 

preprocessedTrainingData = transform(augmentedTrainingData, @(data)helper.preprocessTrainData(data));
 
% Read the preprocessed training data.
data = read(preprocessedTrainingData);

% Display the image with the bounding boxes.
I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% Reset the datastore.
reset(preprocessedTrainingData);
%% Modify Pretrained EfficientDet-D0 Network
% 여기서는 검출할 객체의 크래스들을 정하고 검출할 클래스들의 갯수들을 입력함과 동시에 'generateAnchorBox'라는 함수를 통해 
% anchor 박스들을 생성하게 됩니다. 이후 미리 학습된 모델의 출력레이어를 'configureEfficientDetD0' 함수를 통해 
% 수정하게 됩니다.

% Specify classnames.
% classNames = {'SST' 'LST' 'SSO' 'LSO' 'vest' 'sling' 'shorts' 'trousers' 'skirt' 'SSD' 'LSD' 'vestDress' 'slingDress'};
classNames = {'SST'; 'LST'; 'SSO'; 'LSO' ; 'vest' ; 'sling' ; 'shorts'; 'trousers'; 'skirt'; 'SSD'; 'LSD'; 'vestDress'; 'slingDress'};
numClasses = 13
% size(classNames, 1);

% Add 'background' class to the existing class names.
classNames{end+1, 1} = 'background';

% Determine anchor boxes.
anchorBoxes = helper.generateAnchorBox;

% Configure pretrained model for transfer learning.
[lgraph, networkOutputs] = configureEfficientDetD0(net, numClasses);
%% Specify Training Options
% 트레이닝 옵션을 각각의 필요에 따라서 수정하면 됩니다.

numEpochs = 1;
miniBatchSize = 4;
learningRate = 0.001;
warmupPeriod = 1000;
l2Regularization = 0.0005;
penaltyThreshold = 0.5;
velocity = [];
%% Train Model
% 해당 코드에서는 매트랩에서 제공하는 Parallel Computing Toolbox를 통해 CUDA 사용이 가능한 로컬 NVDIA GPU를 
% 통해 학습합니다.
% 
% Use the 'minibatchqueue' function to split the preprocessed training data 
% into batches with function 'myMiniBatchFcn' which returns the batched images, 
% bounding boxes and the respective class labels. For faster extraction of the 
% batch data for training, 'dispatchInBackground' should be set to "true" which 
% ensures the usage of parallel pool.
% 
% 'minibatchqueue' 함수를 통해 전처리된 학습 데이터를 batch 크기에 맞게 분리하게 됩니다. 이때, 'myMiniBatchFcn' 
% 함수를 같이 사용해서 batch로 분리된 이미지들, 해당하는 bounding box들과 레이블을 받게 됩니다. Batch 과정을 더 빠르게 
% 진행하고 싶다면 'dispatchBackground'가 true로 설정되어야 병렬풀의 사용이 보장됩니다. 

if canUseParallelPool
   dispatchInBackground = true;
else
   dispatchInBackground = false;
end

if canUseGPU
    executionEnvironment = "gpu";
else
    executionEnvironment = "cpu";
end

myMiniBatchFcn = @(img, boxes, labels) deal(cat(4, img{:}), boxes, labels);

mbqTrain = minibatchqueue(preprocessedTrainingData, 3, "MiniBatchFormat", ["SSCB", "", ""],...
                            "MiniBatchSize", miniBatchSize,...
                            "OutputCast", ["single","",""],...
                            "OutputAsDlArray", [true, false, false],...
                            "DispatchInBackground", dispatchInBackground,...
                            "MiniBatchFcn", myMiniBatchFcn,...
                            "OutputEnvironment", [executionEnvironment,"cpu","cpu"]);

% To train the network with a custom training loop and enable automatic differentiation, 
% convert the layer graph to a dlnetwork object. Then create the training progress 
% plotter using helping function configureTrainingProgressPlotter. 
% 
% Finally, specify the custom training loop. For each iteration:
%
% * Read data from the 'minibatchqueue'. If it doesn't have any more data, reset 
%   the 'minibatchqueue' and shuffle. 
% * Evaluate the model gradients using 'dlfeval' and the 'modelGradients' function. 
%   The function 'modelGradients', listed as a supporting function, returns the 
%   gradients of the loss with respect to the learnable parameters in 'net', the 
%   corresponding mini-batch loss, and the state of the current batch.
% * Apply a weight decay factor to the gradients to regularization for more 
%   robust training.
% * Update the network parameters using the 'sgdmupdate' function.
% * Update the 'state' parameters of 'net' with the moving average.
% * Display the learning rate, total loss, and the individual losses for every 
%   iteration. These can be used to interpret how the respective losses are changing 
%   in each iteration. For example, a sudden spike in the box loss after few iterations 
%   implies that there are Inf or NaNs in the predictions.
% * Update the training progress plot.  
%
% The training can also be terminated if the loss has saturated for few epochs. 
rng('default');
modelName = "trainedNet";

start = tic;

% Convert layer graph to dlnetwork.
net = dlnetwork(lgraph);

% Create subplots for the learning rate and mini-batch loss.
fig = figure;
[lossPlotter, learningRatePlotter] = helper.configureTrainingProgressPlotter(fig);

iteration = 0;

% Custom training loop.
for epoch = 1:numEpochs
    reset(mbqTrain);
    shuffle(mbqTrain);
    
    while(hasdata(mbqTrain))
        iteration = iteration + 1;
        
        [imgTrain, bboxTrain, labelTrain] = next(mbqTrain);
        
        % Evaluate the model gradients and loss using dlfeval and the modelGradients function.
        [gradients, state, lossInfo] = dlfeval(@modelGradients, net, imgTrain, bboxTrain, labelTrain, anchorBoxes, penaltyThreshold, classNames, networkOutputs);
        
        % Apply L2 regularization.matlab:matlab.internal.language.introspective.errorDocCallback('focalCrossEntropy>validateTrueValues')
        gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, net.Learnables);
        
        % Determine the current learning rate value.
        currentLR = helper.piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
       
        % Update the network learnable parameters using the SGDM optimizer.
        [net, velocity] = sgdmupdate(net, gradients, velocity, currentLR);
        
        % Update the state parameters of dlnetwork.
        net.State = state;
        
        % Display progress.
        if mod(iteration,100) == 1
            helper.displayLossInfo(epoch, iteration, currentLR, lossInfo);
        end
        
        % Update training plot with new points.
        helper.updatePlots(lossPlotter, learningRatePlotter, iteration, currentLR, lossInfo.totalLoss);
    end
end

save(modelName,"net");
%% Evaluate Model
% Computer Vision System Toolbox에서는 객체 인식 평가 함수들을 제공합니다. 이를 통해 average precision 
% 과 로그 평균 오차율에 대해 측정할 수 있습니다. 여기서는 average precision을 사용하여 평가합니다. 

% load trainedNet.mat
load DeepFashionNet2.mat
%%
% timg=imread('C:\Users\yoons\DeepLearning\Capstone\Detectron2\test-002\test\image\000005.jpg');
% timg = imread('testImg2.jpg')
timg = imread("testImg.jpg")
figure
imshow(timg)
%%
classNames = {'SST'; 'LST'; 'SSO'; 'LSO' ; 'vest' ; 'sling' ; 'shorts'; 'trousers'; 'skirt'; 'SSD'; 'LSD'; 'vestDress'; 'slingDress'};
classNames{end+1, 1} = 'background';
executionEnvironment = 'auto';
[bboxes,scores,labels] = detectEfficientDetD0(net, timg, classNames, executionEnvironment);
labels
% true_labels = (labels ~= 'background1')
% new_labels = labels(true_labels)
% new_scores = scores(true_labels)
% new_bboxes = bboxes(true_labels,:)
if ~isempty(scores)
    Iout = insertObjectAnnotation(timg, 'rectangle', gather(bboxes), gather(scores));
else
    Iout = im2uint8(timg);
end
figure
imshow(Iout)

labelScore = cell(size(labels));
for k = 1:length(labels)
    labelScore{k} = [char(labels(k)),':',num2str(scores(k))];
end

detect = insertObjectAnnotation(timg,"rectangle",bboxes,labelScore);
figure
imshow(detect)
%% 
%%