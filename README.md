# Removing Motion Blur from Text Images and Enhancing OCR Performance Using Mobile IMU Sensors

모바일 **IMU(자이로) 센서**를 이용해 텍스트 이미지의 모션 블러를 제거하고 OCR 성능을
높이는 연구의 소스코드입니다.

**핵심 목표:** 디블러링 네트워크에 **IMU(자이로) 신호**를 추가하면 흐릿한 텍스트가 더
잘 복원되어 **OCR이 더 정확하게 읽는다**는 것을 표 하나로 보이는 것. SOTA를 노리지
않고, 깨끗하고 정직한 신호만 확인합니다.

```
Blur → (디블러 네트워크) → 복원 → OCR → CER
                 ▲
                 └── IMU (블러를 만든 자이로)   ← 이게 도움이 되나?
```

**왜 IMU가 도움이 되는가:** 노출 중 카메라 흔들림은 하나의 움직임 궤적을 적분한
결과이고, **블러 커널이 곧 그 궤적**입니다. 자이로스코프는 같은 움직임의 각속도를
측정하므로 IMU 신호는 블러 커널 정보를 담고 있습니다. 본 생성기는 각 블러를 만든
자이로 시계열을 그대로 저장하기 때문에 융합(fusion)이 물리적으로 타당합니다.

## 결과 (합성 테스트셋 — 300장, 20 epoch)

| 조건 | PSNR (dB) ↑ | CER ↓ |
|---|---|---|
| Blur (입력) | 15.50 | 0.26 |
| Baseline (디블러만) | 21.67 | 0.17 |
| **+ IMU (제안)** | **22.91** | **0.15** |
| Sharp (정답 / 상한) | — | 0.07 |

IMU(자이로) 신호를 더하면 복원 품질(PSNR)과 OCR 정확도(CER)가 모두 디블러 전용
베이스라인보다 좋아집니다. 합성 영문 텍스트 기준 예비 결과이며, 실제 촬영 데이터와
다중 시드 실험은 진행 중입니다.

## 실행 순서 (순서대로)

| # | 내용 | 명령어 | 필요 |
|---|------|--------|------|
| 2 | 데이터 생성 (train 2000 / test 300) | `python data_gen.py --config configs/default.yaml` | CPU |
| 1 | ⭐ **문제 측정**: OCR로 sharp vs blur CER 비교 | `python step1_ocr_problem.py` | OCR |
| 3 | 베이스라인: 디블러 전용 학습 | `python train.py --model baseline` | GPU |
| 4 | 제안: 디블러 + IMU 학습 | `python train.py --model imu` | GPU |
| 5 | 복원 + 최종 표 생성 | `python infer.py --model baseline`<br>`python infer.py --model imu`<br>`python make_table.py` | GPU + OCR |

> **step 1**이 진입점입니다(학습 없음, 실패할 게 없음). 테스트 데이터가 필요하므로
> `data_gen.py`를 먼저 실행하세요. 이후 단계는 같은 데이터와 step 1의 CER 도구를
> 재사용합니다. `bash run_all.sh` 로 3~5단계를 한 번에 실행할 수 있습니다.

## 환경

- **권장: Google Colab (무료 T4 GPU).** torch/torchvision이 이미 설치돼 있어
  `pip install easyocr`만 하면 됩니다. 전체 파이프라인은 `notebooks/colab_demo.ipynb`
  참고.
- **로컬 (Mac/CPU):** step 1~2는 CPU로 충분합니다. 코어만 설치:
  ```bash
  python3 -m venv .venv && source .venv/bin/activate
  pip install numpy pillow scipy matplotlib pyyaml tqdm   # 코어 (데이터 + 메트릭)
  pip install easyocr                                     # OCR 단계 (torch 포함)
  ```
  학습(3~4)도 `train.device: auto`로 Apple Silicon MPS에서 로컬 실행 가능하며,
  GPU보다 느립니다.

## 언어

기본값은 **영문**(`configs/default.yaml`의 `language: en`) — CER 신호가 가장
깨끗합니다. 한글을 시도하려면 `language: ko`(또는 `both`)로 바꾸고 CJK 폰트를
설치하세요(Colab: `!apt-get install -y fonts-nanum`). 생성기가 폰트를 자동 감지합니다.

## 파일 구성

| 파일 | 역할 |
|------|------|
| `configs/default.yaml` | 모든 설정 (이미지 크기, 블러 세기, 데이터 수, 학습) |
| `data_gen.py` | 합성 sharp/blur/IMU 생성기 (step 2) |
| `metrics.py` | 의존성 없는 CER + PSNR |
| `ocr_eval.py` | EasyOCR 래퍼 + split 단위 CER |
| `step1_ocr_problem.py` | **step 1** — sharp vs blur CER |
| `models.py` | 디블러 U-Net(베이스라인) + 자이로 FiLM 융합(제안) |
| `dataset.py`, `train.py`, `infer.py` | 3~4단계 학습 / 복원 |
| `make_table.py` | **step 5** — 최종 비교 표 |
| `run_all.sh` | 3~5단계 일괄 실행 |

결과물은 `results/`에 생성됩니다 (`step1_ocr_problem.csv`, `final_table.md/.csv`,
`restored/`, `checkpoints/`) — 모두 git에서 제외됨.

## 효과가 약할 때 튜닝

- 블러가 약하면 `blur.max_shift_px`를 키우세요.
- IMU 효과가 약하면 `train.epochs`를 늘리거나 `train.base_channels`를 키우세요.
- 짧은 문자열에서 OCR이 불안정하면 `text.min_chars`를 올리세요.

## 인용

```bibtex
@misc{imudeblur2026,
  title  = {Removing Motion Blur from Text Images and Enhancing OCR Performance Using Mobile IMU Sensors},
  year   = {2026},
  note   = {Work in progress}
}
```
