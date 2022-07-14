%% Yolov4를 활용한 객체인식 및 Resnet을 사용한 얼굴 인식 동시 검출
% 해당 코드에서는 Yolov4와 Resnet을 사용한 객체인식 및 얼굴 인식을 동시에 진행하게 됩니다.
%% Prerequisites
%% 
% # 객체 인식과 얼굴인식을 진행하는 두 개의 모델 정보 파일
% # 웹캠
%% Setup
% |소스 디렉터리를 추가합니다.(데이터셋의 위치에 따라 해당 값을 조정해야 합니다.)|

clear
%%
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
%% Loading Models and Basic Settings

% 학습된 모델 불러오기
load yolov4_trained_tiny_epoch5.mat
net1=net
anchor1=anchors
load yolov4_trained_face_recognition.mat
net2=net
anchor2=anchors
% load DeepFashionNet2.mat
classNames = {'SST','LST','SSO','LSO','vest','sling','shorts','trousers','skirt','SSD','LSD','SSO','vestDress','slingDress'};
classNames2 = {'Missing_Person'};
% classNames = {'SST'; 'LST'; 'SSO'; 'LSO' ; 'vest' ; 'sling' ; 'shorts'; 'trousers'; 'skirt'; 'SSD'; 'LSD'; 'vestDress'; 'slingDress'};
% classNames{end+1, 1} = 'background';
numClasses = 13
executionEnvironment = 'auto';
%%
% cam = webcam("Intel(R) RealSense(TM) Depth Camera 415  RGB")
% cam.Resolution='1280x720'
cam = webcam("Integrated Webcam")

video_player = vision.VideoPlayer('Position', [50 50 1100 700])

run_loop = true;
frame_count = 0;
missing_person = 1;
cloth1 = 'SST'
cloth2 = 'SST'
%% Perform Realtime Object Detection and Face Recognition
% 매트랩에서 제공하는 웹캠 관련 기본 함수들을 통해 이미지를 받아온 다음에 객체인식 및 얼굴인식을 동시에 진행하게 됩니다.

while run_loop
    video_frame=snapshot(cam);
    frame_count=frame_count+1;
%     [bboxes,scores,labels] = detectEfficientDetD0(net, video_frame, classNames, executionEnvironment);
    [bboxes, scores, labels] = detectYOLOv4(net1, video_frame, anchor1, classNames, executionEnvironment);
    [f_bboxes, f_scores, f_labels] = detectYOLOv4(net2, video_frame, anchor2, classNames2, executionEnvironment);
    true_scores = (scores >= 0.5);
    new_labels = labels(true_scores);
    new_scores = scores(true_scores);
    new_bboxes = bboxes(true_scores,:);
    true_fscores = (f_scores >= 0.001)
    new_flabels = f_labels(true_fscores);
    new_fscores = f_scores(true_fscores);
    new_fbboxes = f_bboxes(true_fscores,:);
    if ~isempty(new_scores)
        annotations = string(new_labels) + ": " + string(new_scores);
        video_frame = insertObjectAnnotation(video_frame, 'rectangle', new_bboxes, annotations);
        if ~isempty(new_fscores)
            video_frame = insertObjectAnnotation(video_frame, 'rectangle', new_fbboxes, new_flabels);
        end
        if ismember(cloth1, new_labels) && missing_person && ismember(cloth2, new_labels) && ~isempty(new_fbboxes)
            image_name = 'img.jpg'
            imwrite(video_frame, image_name);
            cd(ftpobj, "html/images")
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
mget(ftpobj,'image1.jpg')`