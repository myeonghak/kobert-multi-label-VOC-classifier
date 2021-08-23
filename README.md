KoBERT multi-label VOC classifier
==============================================

<br/>


**[TL-DR]**   
KoBERT를 사용한 PyTorch text multi-label classification 모델입니다. SKT에서 공개한 한국어 pretrain embedding model인 [KoBERT](https://github.com/SKTBrain/KoBERT)를 사용했습니다.  

----

<br>

Contents
--------

1.	[Intro](#intro)
2.	[Structure](#structure)
3.	[Embedding Visualization](#embedding)
4.  [XAI using PyTorch Captum](#captum)
5.  [Streamlit Demo](#demo)

<br>

<a id="intro"></a>
## Intro  

큰 기업에서는 매일 수 천건, 통화로 생성되는 STT 결과 데이터를 포함하면 수 만건에 달하는 VoC(Voice of Customers)가 발생합니다. 이 때 고객의 불만 사항을 실시간으로 분류/관리하여 트렌드를 추적해주는 시스템이 있다면, 운영 부서에서 미처 예상치 못한 서비스 장애를 진단하여 조기에 대응할 수도 있고, 나아가 고객 불만을 보상하는 프로모션을 제공한다면 고객 이탈을 방지함과 동시에 세일즈 KPI에 직접적인 효과를 얻을 수 있겠죠. 이러한 맥락에서 고안된 VOC 자동 분류 모델입니다.  

사용된 데이터는 모 항공사의 VOC 데이터이고, 약 2,000건의 raw 데이터를 가지고 있습니다. 전체 레이블의 수는 70여개이며, 모델링에 사용된 수는 17개 입니다. 극히 소수의 데이터에서 실제 현업에 적용 가능한 수준의 성능을 검증하는 것이 해당 모델의 구현 목적이었습니다. 라벨 데이터는 현업 전문가에 의해 태깅되었고, 상담사와 고객의 대화를 보고 관련되었다고 판단되는 태그를 달아 주었습니다. 데이터와 학습된 weight file은 보안상의 이유로 공개할 수 없음을 양해바랍니다.  


<br>
<a id="structure"></a>
## Structure  
--------

<br>

```
.
├── data                                          # 모델 학습 및 데모에 사용되는 데이터셋
├── KoBERT                                        # 과거 버전의 KoBERT 레포지터리를 클론한 폴더
├── model
│   ├── bert_model.py                             # dataloader, bert 모델 및 학습 관련 util
│   ├── captum_tools_vocvis.py                    # PyTorch XAI를 위한 Captum 관련 util
│   ├── config.py
│   ├── demo.py                                   # 텍스트 입력을 넣으면 모델 예측 결과를 반환하는 streamlit web app
│   ├── inference.py                              # 추론 모듈
│   ├── kobert_multilabel_text_classifier.ipynb
│   ├── metrics_for_multilabel.py                 # multilabel 모델 평가를 위한 metrics
│   └── preprocess.py                             # 전처리 모듈
└── weights                                       # 학습 모델 가중치
```

Python 3.7.11 버전에서 구현되었습니다. conda 가상환경을 권장합니다. 사용 패키지는 requirements.txt를 참조해주세요.  



### Model Performance  

<br>

해당 모델은 Multi-Label classification 모델로, 전체 2,000여 건의 샘플 데이터를 train 85%, test 15%로 분할하여 테스트했습니다.  


| methods                 | NDCG@17| Micro f1 score| Macro f1 score |
|-------------------------|:---------:|:---:|:---:|
| KoBERT              |  **0.841**   | **0.615** | **0.534**|


<br>


<a id="embedding"></a>
### Embedding visualization  



프로젝트의 초기에는 multi-class task로 접근하여 모델링을 진행했습니다. 그런데, 500여개의 데이터 셋으로 나왔던 1차 성능에 비해 샘플이 더 추가된 데이터 셋으로 만든 2차 모델의 성능이 더 떨어지는 현상이 발생했고 (8개 클래스 77% -> 73%), 원인 파악을 위해 오분류 샘플을 조사했습니다.  

|                 |  카테고리 명|
|-------------------------|:---------:|
| 예측 정확도 70% 이상 | 무상 변경/취소, 유상 변경/취소, 기내 서비스 등  |
| 예측 정확도 40% 이하 | **예약 기타**, **변경/취소** |   

위와 같이, 기타 클래스의 특징을 모호하게 포함하고 있는 클래스의 성능이 매우 낮은 것을 확인할 수 있었습니다. 사람이 직접 의미적으로 판단해도 모호한 경우가 많았습니다. 이 클래스에 포함된 샘플들은 모델의 최적화 과정에서 모호한 시그널을 제공함으로써 파라미터 최적화에 악영향을 미칠 것이라고 직관적으로 생각했고, 이와 같은 내용이 버트 분류 모델의 예측에 사용되는 마지막 CLS 토큰의 representation을 low dimension에 mapping 했을 때 확인 가능할 것이라고 가정했습니다.  

아래는 실제 CLS 토큰의 임베딩에 T-SNE를 적용한 결과입니다. 모호한 라벨을 가진 샘플들에 의해 임베딩 스페이스가 다소 entangled된 형태를 보이는 것을 알 수 있습니다.    


<br>
<center><img src="/img/entangled.png" align="center" alt="drawing" width="500"/></center>    
<br/>

그렇다면 이 라벨들을 제거해 준다면, 버트 representation 이후의 레이어가 결정 경계를 손쉽게 그을 수 있도록 임베딩이 학습되지 않을까요?  그러한 질문에 답한 것이 다음과 같은 이미지였습니다.  


<br>
<center><img src="/img/seperated.png" align="center" alt="drawing" width="500"/></center>    
<br/>  

예쁘게 잘 정리 됐네요. 이와 같은 결과가 말해주듯이, 데이터셋을 수작업으로 레이블링할 때 모델이 혼동하지 않는 기준을 세우는 것이 중요하다는 결론을 내릴 수 있었습니다. 아래는 수정 후 모델의 confusion matrix입니다. 85:15로 stratified sampling을 해 주었습니다.  


<br>
<center><img src="/img/confusion_matrix.png" align="center" alt="drawing" width="500"/></center>    
<br/>  



<a id="captum"></a>
### XAI using pytorch Captum   


[Captum](https://captum.ai/)은 PyTorch 모델의 interpretability를 위한 라이브러리입니다. 이 중 자연어 분류 모델의 판단에 긍정적, 부정적으로 영향을 미친 토큰을 시각화해주는 [예제](https://github.com/pytorch/captum/blob/master/tutorials/IMDB_TorchText_Interpret.ipynb)가 있어 본 문제에 적용해 보았습니다. 아래는 시각화 결과입니다.  



<br>
<center><img src="/img/captum_example.png" align="center" alt="drawing" width="400"/></center>    
<br/>

"기내 서비스" 라는 레이블을 예측하는 데 positive한 영향을 준 토큰은 녹색으로, negative한 영향을 준 (즉 라벨 예측에 혼동을 준) 토큰은 붉은 색으로 시각화해 줍니다. 우리의 경우에서는 토큰 시각화가 직관에 다소 부합하지 않는 결과를 보이기도 했으나, 이는 소수 샘플로 인한 특정 토큰의 영향에 의한 것일수도, 한글 토큰의 인코딩의 문제일 수도 있습니다.  




<a id="demo"></a>
### Streamlit Demo  

[streamlit](https://streamlit.io/)은 웹/앱 개발에 익숙치 않은 데이터 사이언티스트들이 손쉽게 웹앱 데모를 구현할 수 있도록 도와주는 high-level data app 라이브러리입니다. 입출력을 현업에게 빠르게 보여주기 위해 다음과 같은 데모를 만들었습니다. 불과 몇 분의 투자로 모델의 I/O를 보여줄 수 있는 매우 간편한 기능을 제공합니다.  

<br>
<center><img src="/img/demo_view.png" align="center" alt="drawing" width="700"/></center>    
<br/>

<br>
<center><img src="/img/demo_example.png" align="center" alt="drawing" width="600"/></center>    
<br/>


다음 커맨드로 간단하게 실행할 수 있습니다.  

```
streamlit run demo.py
```
