# 객체 인식 프로젝트

# main_project

## 활용 데이터셋
[DeepFashion2 다운로드 받기](https://drive.google.com/drive/folders/125F48fsMBz2EF0Cpqk6aaHet5VH399Ok).  
데이터셋 다운로드 후 압축 해제 과정에서 비밀번호가 필요합니다.[암호 획득하기](https://docs.google.com/forms/d/e/1FAIpQLSeIoGaFfCQILrtIZPykkr8q_h9qQ5BoTYbjvf95aXbid0v2Bw/alreadyresponded).  
해당 데이터셋 사용 시 사용하는 개발 환경 및 사용 모델에 따른 데이터 형식 변환이 필요.  
Detectron2 활용시에는 데이터셋을 Coco에 맞게 변환해야 합니다.  
Matlab에서 사용하기 위해서는 데이터셋을 열어서 학습시키기 위해 새로운 데이터 테이플을 생성해야 합니다.  
관련 코드들은 각각의 폴더에 존재합니다.  

## 1.파이썬(코랩 활용)  

1. Detectron2를 이용한 DeepFashion2 데이터셋 학습 및 테스트

## 2.Matlab(로컬 GPU 활용)  

2. Pretrained Efficientnet을 활용한 DeepFashion2 재학습  
* 코드1: 전이학습을 위한 일련의 코드

3. Pretrained Yolov4를 활용한 DeepFashon2 재학습
* 코드1: 전이학습을 위한 일련의 코드
* 코드2: 실시간 객체인식 확인
--- 
