%% Object Detection Using Transfer Learning YOLO v4 Network
% 해당 코드는 DeepFashion2로 전이 학습된 모델 데이터를 불러온 후 객체 인식 진행 후 해당 정보를 웹으로 전송하는 일련의 과정이 
% 작성된 코드입니다.
%% Prerequisites
% To run this example you need the following prerequisites -
%% 
% # 전이학습된 모델데이테
% # 웹캠 및 데이터를 전송받을 서버.
%% Setup
% 소스 디렉터리를 추가합니다.(데이터셋의 위치에 따라 해당 값을 조정해야 합니다.

clear
addpath('src');
ftpobj = ftp("capstone5.dothome.co.kr/","capstone5","scoutmini5!");
%%
cd(ftpobj)
dir(ftpobj)
%%
% webread("http://capstone5.dothome.co.kr/insertData.php?x=10&y=10&id=99")
%%
data = webread("http://capstone5.dothome.co.kr/getData.php")
data = erase(data, '["')
data = erase(data, '"]')
s_data = strsplit(data, '","')
% strcmp(sample_data,'1')
length(s_data)
%% Download the pre-trained network
% 로컬환경에서 사용하는 GPU의 메모리가 4GB인 관계로 yoloV4 기본 모델은 사용이 불가하여 tiny 모델로 진행합니다.

% 학습된 모델 불러오기
load yolov4_trained_tiny_epoch5.mat
load FaceNet.mat
% load DeepFashionNet2.mat
classNames = {'SST','LST','SSO','LSO','vest','sling','shorts','trousers','skirt','SSD','LSD','SSO','vestDress','slingDress'};
% classNames = {'SST'; 'LST'; 'SSO'; 'LSO' ; 'vest' ; 'sling' ; 'shorts'; 'trousers'; 'skirt'; 'SSD'; 'LSD'; 'vestDress'; 'slingDress'};
% classNames{end+1, 1} = 'background';
numClasses = 13
executionEnvironment = 'auto';
%%
cam = webcam("Intel(R) RealSense(TM) Depth Camera 415  RGB")
cam.Resolution='1280x720'
% cam = webcam("Integrated Webcam")

video_player = vision.VideoPlayer('Position', [50 50 1100 700])
faceDetector=vision.CascadeObjectDetector;

run_loop = true;
frame_count = 0;
missing_person = 1;
cloth1 = 'SST'
cloth2 = 'trousers'
%% Perform Object Detection Using YOLOv4 Network

while run_loop
    video_frame=snapshot(cam);
    frame_count=frame_count+1;
%     [bboxes,scores,labels] = detectEfficientDetD0(net, video_frame, classNames, executionEnvironment);
    [bboxes, scores, labels] = detectYOLOv4(net, video_frame, anchors, classNames, executionEnvironment);
    face_bboxes = faceDetector(video_frame);
    true_scores = (scores >= 0.5);
    new_labels = labels(true_scores);
    new_scores = scores(true_scores);
    new_bboxes = bboxes(true_scores,:);
    if ~isempty(face_bboxes)
        video_crop=imcrop(video_frame,face_bboxes(1,:));
        video_crop=imresize(video_crop,[227 227]);
        face_label=classify(FaceNet,video_crop);
        video_frame = insertObjectAnnotation(video_frame, 'rectangle', face_bboxes, face_label);
    end
    if ~isempty(new_scores)
        annotations = string(new_labels) + ": " + string(new_scores);
        video_frame = insertObjectAnnotation(video_frame, 'rectangle', new_bboxes, annotations);
        if ismember(cloth1, new_labels) && missing_person && ismember(cloth2, new_labels) && ~isempty(face_bboxes)
            image_name = ['image' int2str(frame_count) '.jpg']
            imwrite(video_frame, image_name);
            mput(ftpobj, image_name);
            webread("http://capstone5.dothome.co.kr/deleteId.php?id=99");
            missing_person = 0;
        end
    end
    % Visualize detection results.

    step(video_player,video_frame);
    run_loop=isOpen(video_player);
end
%%
clear camObject;
%%
dir(ftpobj)
%%
delete(ftpobj, 'image48.jpg')
dir(ftpobj)
%%
mget(ftpobj,'image1.jpg')