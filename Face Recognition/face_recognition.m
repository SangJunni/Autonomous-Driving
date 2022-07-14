%% *Alexnet face recognition*

cam=webcam;
faceDetector=vision.CascadeObjectDetector;
c=100;
temp=0;
while true
    video_frame=cam.snapshot;
    bboxes =step(faceDetector,video_frame);
    if(sum(sum(bboxes))~=0)
    if(temp>=c)
        break;
    else
    video_face=imcrop(video_frame,bboxes(1,:));
    video_face=imresize(video_face,[224 224]);
    filename=strcat(num2str(temp),'.bmp');
    imwrite(video_face,filename);
    temp=temp+1;
    imshow(video_face);
    drawnow;
    end
    else
        imshow(video_frame);
        drawnow;
    end
end
%%
model=alexnet;
layers=model.Layers;
layers(23)=fullyConnectedLayer(1);
layers(25)=classificationLayer;
allImages=imageDatastore('Face_Recongnition_Database','IncludeSubfolders',true, 'LabelSource','foldernames');
opts=trainingOptions('sgdm','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',64);
FaceNet=trainNetwork(allImages,layers,opts);
save FaceNet;
%%
model2=resnet18;
model3 = googlenet
numClasses = 1
lgraph = layerGraph(model3);
lgraph
newFCLayer = fullyConnectedLayer(numClasses,'Name','new_fc','WeightLearnRateFactor',10,'BiasLearnRateFactor',10);
lgraph = replaceLayer(lgraph,'loss3-classifier',newFCLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,'output',newClassLayer);
allImages=imageDatastore('Face_Recognition_Database_Resnet','IncludeSubfolders',true, 'LabelSource','foldernames');
opts=trainingOptions('adam','InitialLearnRate',0.001,'MaxEpochs',20,'MiniBatchSize',32);
FaceResNet=trainNetwork(allImages,lgraph,opts);
save FaceResNet;
%%
cam=webcam;
load FaceResNet;
faceDetector=vision.CascadeObjectDetector;
while true
    e=cam.snapshot;
    bboxes =step(faceDetector,e);
    if(sum(sum(bboxes))~=0)
      es=imcrop(e,bboxes(1,:));
      es=imresize(es,[224 224]);
      label=classify(FaceResNet,es);
      image(e);
      title(char(label));
      drawnow;
    else
        image(e);
        title('No Face Detected');
    end
end