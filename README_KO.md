# 조명 변화에 강건한 ResNet-34 U-Net 기반 균열 분할 및 ONNX Runtime 최적화

## 1. 프로젝트 개요

이 프로젝트는 다양한 조명 환경에서 도로 및 콘크리트 표면의 얇은 균열을 검출하기 위한 딥러닝 기반 균열 분할 파이프라인입니다.

초기 구현에서는 기본 U-Net과 CLAHE 전처리, Morphology 후처리를 중심으로 실험했습니다. 이후 모델 정확도와 조명 강건성, 평가 과정의 신뢰성을 높이기 위해 전체 실험 구조를 다음과 같이 개선했습니다.

- Train / Validation / Test 데이터 분리
- BCE와 Dice 결합 손실 적용
- 조명 변화 기반 데이터 증강
- ImageNet 사전학습 ResNet-34 Encoder 적용
- Encoder Freeze–Unfreeze 기반 2단계 Fine-tuning
- Validation 기반 Threshold 선정
- Morphology 후처리 비교 실험
- ONNX Runtime 기반 배포 성능 평가

최종 목표는 일반 조명뿐 아니라 저조도와 과노출 환경에서도 안정적인 균열 분할 성능을 유지하는 모델을 구축하는 것입니다.

---

# 2. 문제 정의

균열 분할에는 다음과 같은 어려움이 있습니다.

- 균열은 얇고 전체 이미지에서 차지하는 영역이 작음
- 저조도에서는 균열의 가시성이 감소함
- 과노출에서는 표면 질감과 경계가 약해짐
- 배경의 선형 질감이 균열로 오검출될 수 있음
- 과도한 후처리는 실제 얇은 균열까지 제거할 수 있음

따라서 이 프로젝트는 다음 목표에 집중했습니다.

> 균열 분할 정확도를 개선하면서 큰 조명 변화에서도 안정적인 성능을 유지하는 것

---

# 3. 데이터셋

## DeepCrack 데이터셋

표면 이미지와 이에 대응하는 이진 균열 마스크를 제공하는 DeepCrack 데이터셋을 사용했습니다.

데이터 구조:

```text
DeepCrack/
├── train_img
├── train_lab
├── test_img
└── test_lab
```

## 데이터 분할

기존 Train 데이터는 고정된 Random Seed를 사용해 Train과 Validation으로 분리했습니다.

| 구분 | 이미지 수 | 사용 목적 |
| --- | ---: | --- |
| Train | 240 | 모델 학습 |
| Validation | 60 | 모델 및 Threshold 선정 |
| Test | 237 | 최종 성능 평가 |

Test 데이터는 모델 선택이나 Threshold 조정에 사용하지 않았습니다.

---

# 4. 최종 파이프라인

```text
RGB 입력 이미지
→ 256 × 256 크기 조정
→ ResNet-34 U-Net 균열 분할
→ Sigmoid 확률 맵
→ Threshold 0.55 적용
→ 이진 균열 마스크
→ ONNX Runtime 추론
```

![파이프라인 개요](assets/pipeline_overview.png)

최종 파이프라인에는 CLAHE 전처리와 Morphology 후처리를 적용하지 않았습니다.

고정된 이미지 전처리에 의존하는 대신, 학습 단계에서 조명 변화 증강을 적용해 다양한 밝기 환경에 대응했습니다.

---

# 5. 베이스라인 모델

## 5.1 모델 구조

Semantic Segmentation의 베이스라인으로 기본 U-Net을 사용했습니다.

### 입력

- RGB 이미지
- 해상도: 256 × 256

### 출력

- 이진 균열 분할 마스크

### 베이스라인 학습 설정

- Loss: BCEWithLogitsLoss
- Optimizer: Adam
- Batch Size: 4
- 데이터 분할: Train 240장 / Validation 60장
- Validation Dice를 기준으로 Best Checkpoint 저장

## 5.2 베이스라인 결과

Validation 데이터를 별도로 분리하고 평가 구조를 재정비한 뒤 학습한 베이스라인 모델의 성능은 다음과 같습니다.

| 모델 | Validation Dice | Validation IoU |
| --- | ---: | ---: |
| BCE 베이스라인 | 0.7644 | 0.6434 |

이 결과를 이후 손실함수, 데이터 증강, 모델 구조 개선 실험의 재현 가능한 기준 성능으로 사용했습니다.

---

# 6. BCE와 Dice 결합 손실

## 6.1 적용 배경

균열 픽셀은 배경 픽셀보다 훨씬 적은 영역을 차지합니다.

BCE Loss는 각 픽셀을 균열과 배경으로 분류하는 데 효과적이지만, 얇은 균열 영역의 전체적인 겹침 정도를 직접적으로 최적화하는 데에는 한계가 있을 수 있습니다.

이 문제를 보완하기 위해 다음 두 손실함수를 결합했습니다.

- 픽셀 단위 분류를 위한 BCE Loss
- 균열 영역의 겹침을 높이기 위한 Dice Loss

## 6.2 결과

| 모델 | Validation Dice | Validation IoU |
| --- | ---: | ---: |
| BCE 베이스라인 | 0.7644 | 0.6434 |
| BCE + Dice | 0.7871 | 0.6668 |

결합 손실을 적용한 결과 Dice와 IoU가 모두 개선됐으며, 얇은 균열 구조를 더욱 안정적으로 보존했습니다.

---

# 7. 조명 변화 기반 데이터 증강

## 7.1 적용 배경

베이스라인 모델은 일반 이미지에서는 안정적인 성능을 보였지만, 큰 조명 변화에서는 성능이 크게 감소했습니다.

특히 저조도 환경에서는 얇은 균열이 사라지고, 과노출 환경에서는 표면 대비와 경계가 약해지는 문제가 발생했습니다.

조명 변화에 대한 강건성을 높이기 위해 학습 데이터를 다음 세 조건으로 구성했습니다.

| 학습 조건 | 적용 비율 |
| --- | ---: |
| 원본 이미지 | 40% |
| 저조도 증강 | 30% |
| 과노출 증강 | 30% |

## 7.2 적용한 증강

### 저조도 증강

- 밝기 감소
- Gamma 조정
- Gaussian Noise 추가

### 과노출 증강

- Contrast 증가
- 양의 밝기 값 추가

증강은 Train 데이터에만 적용했으며, 모델과 Threshold를 선정할 때 Validation과 Test 이미지는 원본 상태를 유지했습니다.

---

# 8. 조명 강건성 평가

동일한 Validation 데이터와 Threshold 0.50을 사용해 세 모델을 비교했습니다.

| 조건 | BCE + Dice | 무작위 조명 증강 | 균형 조명 증강 |
| --- | ---: | ---: | ---: |
| 원본 | 0.7862 | **0.7908** | 0.7804 |
| 저조도 50% | 0.6203 | 0.6928 | **0.7487** |
| 저조도 35% | 0.3707 | 0.4498 | **0.7193** |
| 저조도 25% | 0.2794 | 0.1972 | **0.6571** |
| 심한 과노출 | 0.4736 | 0.6159 | **0.6985** |

균형 조명 증강 모델은 원본 Validation 이미지에서는 성능이 소폭 감소했지만, 저조도와 과노출 환경에서는 훨씬 높은 성능을 유지했습니다.

이 실험을 통해 다음 두 목표 사이의 Trade-off를 확인했습니다.

- 일반 환경에서의 최고 성능
- 조명 변화 환경에서의 안정적인 성능

실제 균열 점검 환경에서는 조명 변화에 대한 안정성이 더 중요하다고 판단해 균형 조명 증강을 최종 학습 방식으로 선정했습니다.

![조명 강건성](assets/lighting_robustness.png)

---

# 9. ResNet-34 U-Net과 2단계 Fine-tuning

## 9.1 적용 배경

학습 데이터가 240장으로 제한돼 있어, 랜덤 초기화된 기본 U-Net만으로는 충분한 특징 표현력을 확보하는 데 한계가 있었습니다.

적은 데이터에서도 표면 질감과 균열 경계를 효과적으로 추출하기 위해 기본 U-Net의 Encoder를 ImageNet으로 사전학습된 ResNet-34로 교체했습니다.

최종 모델은 다음과 같이 구성했습니다.

- ImageNet 사전학습 ResNet-34 Encoder
- Skip Connection을 사용하는 U-Net 형태의 Decoder
- BCE와 Dice 결합 손실
- 균형 조명 증강
- Encoder Freeze–Unfreeze 기반 2단계 Fine-tuning

## 9.2 2단계 학습

### Stage 1: Encoder 고정

사전학습된 ResNet-34 Encoder를 고정하고 Decoder를 12 Epoch 동안 학습했습니다.

| 학습 단계 | 최고 Validation Dice | 최고 Validation IoU |
| --- | ---: | ---: |
| Encoder 고정 | 0.8052 | 0.6835 |

### Stage 2: 전체 Fine-tuning

이후 Encoder를 해제하고 전체 네트워크를 30 Epoch 동안 추가 학습했습니다.

Encoder와 Decoder에는 서로 다른 Learning Rate를 적용했습니다.

- Encoder Learning Rate: 1e-5
- Decoder Learning Rate: 1e-4

최고 성능의 Checkpoint는 Stage 2의 24 Epoch에서 저장됐습니다.

| 모델 | Validation Dice | Validation IoU |
| --- | ---: | ---: |
| 기본 U-Net | 0.7816 | 0.6590 |
| ResNet-34 U-Net | **0.8259** | **0.7108** |

사전학습 Encoder와 2단계 Fine-tuning을 적용해 Validation Dice는 0.0443, IoU는 0.0518 향상됐습니다.

---

# 10. Threshold 선정

최종 Threshold는 Validation 데이터만 사용해 선정했습니다.

| Threshold | Dice | IoU |
| ---: | ---: | ---: |
| 0.30 | 0.8242 | 0.7084 |
| 0.40 | 0.8253 | 0.7101 |
| 0.50 | 0.8259 | 0.7108 |
| **0.55** | **0.8262** | **0.7112** |
| 0.60 | 0.8259 | 0.7108 |
| 0.70 | 0.8252 | 0.7099 |
| 0.80 | 0.8235 | 0.7076 |

최종 Threshold는 다음 값으로 고정했습니다.

```text
0.55
```

Threshold를 선정한 이후에는 Test 결과를 이용해 모델이나 설정을 변경하지 않았습니다.

---

# 11. Morphology 후처리 실험

## 11.1 실험 목적

기본 U-Net 개발 단계에서 작은 노이즈 제거와 끊어진 균열 연결이 예측 마스크를 개선하는지 확인하기 위해 Morphology 후처리를 비교했습니다.

Validation 데이터에서 다음 조건을 평가했습니다.

- 후처리 없음
- Connected Component 기반 작은 영역 제거
- 3 × 3 Morphology Closing 후 작은 영역 제거

## 11.2 결과

| 방법 | Validation Dice | Validation IoU |
| --- | ---: | ---: |
| 후처리 없음 | 0.7816 | 0.6590 |
| 작은 영역 제거, min area 20 | **0.7822** | 0.6598 |
| Closing + 작은 영역 제거, min area 20 | 0.7820 | **0.6599** |

최대 개선 폭은 0.001 미만이었습니다.

Morphology 후처리는 성능 개선이 매우 작았고, 짧거나 얇은 실제 균열까지 제거할 가능성이 있다고 판단해 기본 U-Net과 ResNet-34 U-Net의 최종 파이프라인에서 모두 제외했습니다.

이 실험을 통해 다음을 확인했습니다.

> 모델 자체의 예측이 충분히 안정적이라면 추가적인 후처리가 반드시 필요한 것은 아니다.

---

# 12. 최종 Test 평가

최종 ResNet-34 U-Net과 Threshold 0.55를 고정한 뒤, 사용하지 않았던 Test 데이터에서 평가했습니다.

## 공식 Test 결과

| 모델 | Dice | IoU |
| --- | ---: | ---: |
| 기본 U-Net | 0.7743 | 0.6625 |
| ResNet-34 U-Net | **0.7902** | **0.6749** |
| 개선 폭 | **+0.0159** | **+0.0124** |

최종 모델은 Test 이미지 237장에서 Dice와 IoU를 모두 개선했습니다.

Validation Dice는 0.8262, Test Dice는 0.7902였습니다. Test에서의 개선 폭은 Validation보다 작았지만, 학습에 사용하지 않은 이미지에서도 모델 구조 변경에 따른 성능 향상을 확인했습니다.

![모델 개선 비교](assets/model_improvement_comparison.png)

---

# 13. Test 데이터 조명 강건성 평가

Test 데이터에도 조명 변화를 적용해 참고용 강건성 평가를 진행했습니다.

공식 Test 성능은 조명 변환을 적용하지 않은 원본 Test 이미지의 결과입니다.

| Test 조건 | Dice | IoU | 원본 대비 Dice 감소 |
| --- | ---: | ---: | ---: |
| 원본 | **0.7902** | **0.6749** | 0.0000 |
| 저조도 50% | 0.7843 | 0.6657 | 0.0059 |
| 저조도 35% | 0.7843 | 0.6649 | 0.0059 |
| 저조도 25% | 0.7814 | 0.6606 | 0.0088 |
| 심한 과노출 | 0.7469 | 0.6224 | 0.0434 |

이미지 밝기를 원본의 25%까지 낮춘 조건에서도 Dice 감소 폭은 0.0088에 그쳤습니다.

심한 과노출 환경에서도 Dice 0.7469를 유지했습니다.

---

# 14. ONNX 변환 및 추론 성능 평가

## 14.1 실험 목적

최종 모델을 ONNX로 변환한 뒤 다음 항목을 검증했습니다.

- 모델 변환 후 출력 일치 여부
- CPU 기반 추론 가능성
- Runtime 변환에 따른 속도 개선

## 14.2 정확도 유지 여부

PyTorch와 ONNX 모델을 동일한 Test 이미지 237장에서 평가했습니다.

| Runtime | Dice | IoU |
| --- | ---: | ---: |
| PyTorch CPU | 0.7902 | 0.6749 |
| ONNX Runtime CPU | 0.7902 | 0.6749 |

출력 차이는 다음과 같습니다.

| 평가 항목 | 결과 |
| --- | ---: |
| Dice 차이 | 0.00000044 |
| IoU 차이 | 0.00000063 |
| 최대 확률값 차이 | 0.00005490 |

ONNX 변환 후에도 기존 PyTorch 모델과 사실상 동일한 균열 분할 성능을 유지했습니다.

## 14.3 CPU 추론 속도 비교

평가 조건:

- CPU 추론
- Batch Size: 1
- 입력 해상도: 256 × 256

| Runtime | 평균 추론 시간 | FPS |
| --- | ---: | ---: |
| PyTorch CPU | 72.59ms | 13.78 |
| ONNX Runtime CPU | **41.46ms** | **24.12** |

ONNX Runtime을 적용해 CPU 추론 속도가 다음과 같이 개선됐습니다.

```text
1.75배
```

현재 처리 속도는 30 FPS에는 미치지 못하지만, 정확도 손실 없이 배포 형식으로 변환하고 실시간 처리 가능성에 가까운 수준까지 추론 속도를 개선했습니다.

---

# 15. 시각화 결과

## 모델 개선 비교

```text
입력 이미지
→ 정답 마스크
→ 기본 U-Net 예측
→ ResNet-34 U-Net 예측
```

최종 ResNet-34 U-Net은 기본 U-Net과 비교해 균열의 연속성을 개선하고 미검출 영역을 줄였습니다.

![모델 개선 결과](assets/model_improvement_comparison.png)

## 조명 강건성

다음 조건에서 최종 모델의 예측 결과를 비교했습니다.

- 원본 조명
- 저조도 50%
- 저조도 25%
- 심한 과노출

![조명 강건성 결과](assets/lighting_robustness.png)

## 대표 예측 결과

성능이 높은 대표 사례와 일반적인 예측 사례를 함께 확인했습니다.

![Best Case](assets/best_case.png)

![Typical Case](assets/typical_case.png)

---

# 16. 핵심 인사이트

1. 신뢰할 수 있는 모델과 Threshold 선정을 위해 Validation 데이터를 별도로 분리해야 했습니다.
2. BCE와 Dice 결합 손실은 BCE 단독 사용보다 얇은 균열 영역의 겹침 성능을 개선했습니다.
3. 저조도와 과노출 조건을 명시적으로 구성한 균형 조명 증강이 큰 조명 변화에 대한 강건성을 높였습니다.
4. ImageNet 사전학습 ResNet-34 Encoder는 Train 이미지 240장의 제한된 데이터에서도 특징 추출 성능을 개선했습니다.
5. Freeze–Unfreeze 기반 2단계 Fine-tuning으로 Validation Dice를 0.7816에서 0.8262로 개선했습니다.
6. Morphology 후처리는 개선 폭이 매우 작아 최종 파이프라인에서 제외했습니다.
7. 최종 ResNet-34 U-Net은 Test 237장에서 Dice 0.7902, IoU 0.6749를 기록했습니다.
8. 저조도 25% Test Dice는 0.7814로, 원본 Test보다 0.0088만 감소했습니다.
9. ONNX Runtime은 모델 정확도를 유지하면서 CPU 추론 속도를 1.75배 개선했습니다.

---

# 17. 기술 스택

## Deep Learning

- Python
- PyTorch
- U-Net
- ResNet-34
- Transfer Learning
- BCE Loss
- Dice Loss

## Computer Vision

- OpenCV
- NumPy
- 조명 변화 데이터 증강
- Connected Component 분석
- Morphology 비교 실험

## 배포 및 평가

- ONNX
- ONNX Runtime
- Matplotlib
- CSV / JSON 결과 저장

---

# 18. 결론

이 프로젝트는 초기 U-Net 균열 분할 모델의 학습 과정과 평가 방법을 함께 개선했습니다.

최종 ResNet-34 U-Net 파이프라인의 결과는 다음과 같습니다.

- Validation Dice 0.8262, IoU 0.7112
- Test 237장에서 Dice 0.7902, IoU 0.6749
- 저조도 25% 조건에서 Test Dice 0.7814
- 심한 과노출 조건에서 Test Dice 0.7469
- PyTorch와 ONNX 모델의 사실상 동일한 분할 정확도 유지
- CPU 추론 시간 72.59ms에서 41.46ms로 단축
- ONNX Runtime 기반 CPU 추론 속도 1.75배 개선

이 프로젝트의 핵심은 단순히 전처리나 후처리를 추가하는 것이 아니라 다음 과정을 통해 모델 자체의 안정성과 실용성을 함께 높인 점입니다.

> 신뢰할 수 있는 Validation 구조, 손실함수 개선, 조명 변화 기반 학습, 사전학습 특징 추출, 2단계 Fine-tuning, 배포 환경 평가
