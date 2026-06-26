# Kaggle Portfolio

Kaggle 대회에서 진행한 프로젝트 중, 제가 직접 맡았던 모델링 작업을 포트폴리오 관점에서 다시 정리한 저장소입니다.

![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat-square&logo=kaggle&logoColor=white)
![Computer Vision](https://img.shields.io/badge/Computer%20Vision-YOLO11m--cls-0F172A?style=flat-square)
![NLP](https://img.shields.io/badge/NLP-French%20Back--Translation%20%2B%20LR-0B6E4F?style=flat-square)
![Pig F1](https://img.shields.io/badge/Pig%20Posture%20Public%20LB-0.916-F59E0B?style=flat-square)
![Disaster Macro F1](https://img.shields.io/badge/Disaster%20Tweets%20Macro%20F1-0.77-2563EB?style=flat-square)

- `Pig Posture Recognition`: YOLO 기반 돼지 자세 분류 모델 담당
- `Natural Language Processing with Disaster Tweets`: 프랑스어 경유 이중번역 데이터 증강과 Logistic Regression 기반 NLP 분류 모델 담당

저장소에는 대회 보고서 PDF, Kaggle Notebook 링크, 그리고 제가 맡은 파트의 실험 흐름과 성과를 한눈에 볼 수 있도록 정리한 README가 포함되어 있습니다.

## Summary

| Project | Domain | My Main Contribution | Core Strategy | Verified Result |
| --- | --- | --- | --- | --- |
| Pig Posture Recognition | Computer Vision | 객체 중심 crop 입력과 YOLO 학습 전략 설계 | YOLO11m-cls + K-Fold + Multi-crop TTA | Public LB F1 `0.916` |
| NLP with Disaster Tweets | NLP / Text Classification | 직접 설계한 프랑스어 경유 이중번역으로 재난 클래스 데이터 증강 | French Back-Translation + TF-IDF + Logistic Regression | Accuracy `0.78`, Macro F1 `0.77` |

## 1. Pig Posture Recognition

![Pig competition header](assets/competition-header-pig.png)

### Official Competition Snapshot

- Competition: [Pig Posture Recognition](https://www.kaggle.com/competitions/pig-posture-recognition)
- Official description: `A Computer Vision Challenge in Precision Livestock Farming (PLF)`
- Task: 돼지의 자세를 이미지 단위로 분류하는 computer vision 문제
- My role: `YOLO 기반 classification pipeline` 설계 및 성능 개선

### Dataset Snapshot

보고서와 대회 데이터 구조를 기준으로 보면, 이 프로젝트는 bbox 기반 객체 중심 분류가 핵심인 과제였습니다.

| Item | Details |
| --- | --- |
| Input | 돼지 이미지와 `train.csv`의 bbox 정보 |
| Sample unit | `row_id` 단위 crop 이미지 |
| Train split | 전체 이미지의 약 70%, `2,164`장 |
| Instances in train split | `16,062`개 인스턴스 |
| Label structure | 자세 분류용 다중 클래스 |
| Notable classes in my work | `Lateral_left`, `Lateral_right`, `Sitting` 등 |

### Evaluation

대회 평가는 각 자세 클래스별 `F1 score`를 독립적으로 계산한 뒤, 클래스 빈도와 관계없이 동일 가중치로 평균하는 방식입니다. 클래스 불균형이 존재하는 데이터셋이라, 단순 정확도보다 클래스별 균형 잡힌 분류 성능이 더 중요했습니다.

### My Approach

핵심 아이디어는 "원본 전체 이미지"보다 "돼지 객체 중심 입력"에 집중하도록 만드는 것이었습니다.

- `YOLO11m-cls` 기반 전이학습 사용
- `train.csv`의 bbox 정보를 활용해 row 단위 crop 이미지 생성
- bbox 주변 정보를 일부 살리기 위해 `PAD=0.10` 확장 적용
- 비율 왜곡을 줄이기 위해 `letterbox + 평균색 padding` 적용
- `Lateral_left / Lateral_right` 같은 방향성 클래스 보호를 위해 `flip` 비활성화
- `K-Fold` 학습과 `Multi-crop TTA` 적용
- Sitting 클래스 약세 보완을 위한 약한 oversampling / augmentation 적용
- 이후 `pseudo-labeling`, `hyperparameter tuning`까지 확장

### Performance

아래 수치는 현재 저장소에 포함된 보고서에서 확인 가능한 `Public Leaderboard` 기준입니다.

| Setting | Public LB |
| --- | --- |
| YOLOv11s baseline | `0.723` |
| + Image size increase | `0.820` |
| + TTA (Multi-crop inference) | `0.857` |
| + Pseudo-labeling | `0.878` |
| + Hyperparameter tuning | `0.916` |

### Result Takeaways

| Conclusion | Evidence | Portfolio Signal |
| --- | --- | --- |
| 객체 중심 입력 구성이 성능 개선의 출발점이 됨 | bbox crop, `PAD=0.10`, `letterbox + 평균색 padding` 적용 | 이미지 전체가 아니라 분류에 필요한 객체 영역에 모델이 집중하도록 설계 |
| 추론 안정화 전략이 leaderboard 성능을 끌어올림 | TTA 적용 후 Public LB `0.857` | 단일 예측보다 여러 crop 기반 추론을 결합해 예측 변동성 완화 |
| 최종 고도화로 목표 성능을 달성 | pseudo-labeling과 hyperparameter tuning 이후 Public LB `0.916` | 실험 결과를 바탕으로 단계적으로 성능을 개선 |

### Why This Work Matters

이 프로젝트에서는 단순히 모델만 바꾼 것이 아니라, 실제 성능에 더 직접적인 영향을 주는 `입력 구성`, `클래스 특성에 맞는 augmentation 설계`, `추론 안정화 전략`을 주도적으로 다뤘습니다. 특히 좌우 방향이 의미를 가지는 클래스에서 일반적인 flip augmentation이 오히려 label noise를 만들 수 있다는 점을 고려해 설정을 조정한 부분이 실무적으로도 의미 있는 판단이었습니다.

### Links

- Kaggle Competition: [Pig Posture Recognition](https://www.kaggle.com/competitions/pig-posture-recognition)
- Kaggle Notebook: [Pig Posture Recognition YOLO](https://www.kaggle.com/code/byeongsunmoon/pig-posture-recognition-yolo)
- Report: [Kaggle Pig Posture Recognition대회 보고서.pdf](<./Kaggle Pig Posture Recognition대회 보고서.pdf>)

## 2. Natural Language Processing with Disaster Tweets

![Disaster competition header](assets/competition-header-disaster.png)

### Official Competition Snapshot

- Competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- Official description: `Predict which Tweets are about real disasters and which ones are not`
- Task: 재난 관련 트윗 여부를 분류하는 binary text classification 문제
- My role: `프랑스어 경유 이중번역 데이터 증강 + Logistic Regression 기반 NLP pipeline` 설계 및 고도화

### Dataset Snapshot

보고서 기준 이 대회 데이터는 짧은 트윗 텍스트를 중심으로 구성되어 있고, 일부 보조 컬럼과 이진 레이블을 함께 제공합니다.

| Item | Details |
| --- | --- |
| Train samples | `7,613` |
| Test samples | `3,263` |
| Input columns | `id`, `keyword`, `location`, `text` |
| Target | `target` (`1`: 재난, `0`: 비재난) |
| Data characteristics | 축약어, 해시태그, 중의적 표현, 위치 정보 결측 등 |

### Evaluation

대회 평가지표는 `F1 Score`입니다. 재난 트윗 탐지에서는 false negative와 false positive를 함께 관리해야 해서, 단순 accuracy보다 precision과 recall의 균형이 중요한 문제였습니다.

### My Approach

딥러닝 모델을 무작정 키우기보다, 데이터 품질을 먼저 개선하는 방향으로 접근했습니다. 특히 재난 트윗 클래스의 표현 다양성을 늘리기 위해 `영어 원문 -> 프랑스어 -> 영어` 흐름의 이중번역(back-translation)을 직접 설계해 데이터 증강에 적용했습니다.

| Step | What I Did | Why It Mattered |
| --- | --- | --- |
| 데이터 증강 설계 | 재난 클래스 문장을 프랑스어로 번역한 뒤 다시 영어로 되돌리는 back-translation 적용 | 원문 의미는 유지하면서 표현만 달라진 고품질 학습 샘플 확보 |
| 증강 품질 관리 | 단순 복제가 아니라 의미가 크게 훼손되지 않는 문장을 중심으로 학습 데이터에 반영 | 노이즈를 줄이고 재난 클래스의 표현 다양성 보강 |
| 특징 추출 | `TF-IDF` 기반 텍스트 벡터화와 보조 feature 사용 | Logistic Regression이 짧은 트윗에서도 핵심 단어 패턴을 활용할 수 있도록 구성 |
| 분류 모델 | `scikit-learn LogisticRegression` + `L2 regularization` 사용 | 비용이 낮고 해석 가능한 baseline을 안정적으로 고도화 |

### Performance

아래 수치는 현재 저장소에 포함된 보고서에서 확인 가능한 Logistic Regression 파트 성능입니다.

| Metric | Result |
| --- | --- |
| Accuracy | `0.78` |
| Macro F1 | `0.77` |
| Weighted F1 | `0.77` |
| Disaster class F1 | `0.74` |
| Disaster class Recall | `0.72` |

### Result Takeaways

| Conclusion | Evidence | Portfolio Signal |
| --- | --- | --- |
| 핵심 기여는 모델 복잡도보다 데이터 증강 전략에 있음 | 프랑스어 경유 back-translation으로 재난 클래스 학습 샘플 보강 | 문제 특성에 맞는 데이터 중심 개선을 직접 설계 |
| 고품질 증강이 짧은 트윗 분류의 한계를 보완 | Accuracy `0.78`, Macro F1 `0.77`, Disaster class F1 `0.74` | 단순 feature 튜닝이 아니라 클래스 표현 다양성 자체를 개선 |
| 해석 가능한 모델로 실용적인 baseline 구축 | TF-IDF + Logistic Regression 기반으로 성능 확인 | 빠르게 반복 실험하고 결과를 설명할 수 있는 파이프라인 구성 |

### Why This Work Matters

이 작업의 강점은 단순히 Logistic Regression을 사용한 것이 아니라, 데이터가 부족한 재난 클래스에 대해 `프랑스어 경유 이중번역`을 직접 적용해 의미는 유지하면서 표현을 다양화했다는 점입니다. 결과적으로 복잡한 딥러닝 모델 대비 비용이 낮고 해석 가능성이 높은 baseline을 만들면서도, 데이터 중심의 성능 개선 경험을 보여줄 수 있었습니다.

### Links

- Kaggle Competition: [Natural Language Processing with Disaster Tweets](https://www.kaggle.com/competitions/nlp-getting-started)
- Kaggle Notebook: [Basic NLP on Disaster Tweets](https://www.kaggle.com/code/byeongsunmoon/basic-nlp-on-disaster-tweets)
- Report: [Kaggle Natural Language Processing with Disaster Tweets 대회 보고서(11.30.18h).pdf](<./Kaggle Natural Language Processing with Disaster Tweets 대회 보고서(11.30.18h).pdf>)

## Portfolio Notes

- 이 저장소 설명은 `제가 직접 담당한 파트`를 중심으로 정리했습니다.
- 공식 대회 설명과 헤더 이미지는 각 Kaggle competition 페이지를 참고했습니다.
- 데이터셋 구조와 평가 지표 설명은 현재 저장소의 보고서와 대회 페이지 정보를 바탕으로 요약했습니다.
- 점수는 현재 저장소에 포함된 보고서에서 확인 가능한 범위만 반영했습니다.

## Files

- [README.md](./README.md)
- [Kaggle Pig Posture Recognition대회 보고서.pdf](<./Kaggle Pig Posture Recognition대회 보고서.pdf>)
- [Kaggle Natural Language Processing with Disaster Tweets 대회 보고서(11.30.18h).pdf](<./Kaggle Natural Language Processing with Disaster Tweets 대회 보고서(11.30.18h).pdf>)
