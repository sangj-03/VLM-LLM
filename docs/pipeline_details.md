# 파이프라인 세부 사항

README의 [시스템 구조](../README.md#시스템-구조)를 코드·설정 단위로 풀어 쓴 문서입니다.

## 1. 논문 표기와 코드 위치

| 논문 내용 | 값 | 코드·설정 |
| --- | --- | --- |
| 운전자 검출·추적 | YOLO26n + ByteTrack, 매 프레임 | `--driver-yolo-model`, `--driver-yolo-interval 1`, `src/features/driver_selector.py` |
| 특징 추출 | 32차원, 10 Hz | `TCN_FEATURE_NAMES` (`src/pipeline/run_tcn_vlm_yolo.py`), `configs/tcn/tcn_1p5s_32d.json` |
| 주 TCN | 인과적 TCN 5개 앙상블, 1.5 s(15 시점), dilation 1·2·4 | `configs/ensembles/observable_state_tcn_v38.json`, `src/models/observable_state_tcn.py` |
| 후보 선별 | 위험 점수 ≥ 0.50, 0.4 s 지속 | `--event-threshold 0.50`, `--combined-risk-threshold 0.50`, `--trigger-persist-sec` (기본 0.4) |
| 보조 조건: 지속적 눈 감김 | 닫힘 확률 ≥ 0.70, 0.6 s 지속, 눈 가시 비율 ≥ 0.60, 가중치 0.45 | `--eye-*` 인자, `src/models/eye_state_model.py` |
| 보조 조건: 눈 관측 불확실성 | 눈 관측 불량이 지속되고 TCN 점수 ≥ 0.59 × 임계값이면 1 s 간격으로 VLM 확인 | `--eye-uncertain-tcn-ratio 0.59`, `--eye-uncertain-watchdog-sec 1.0` |
| 보조 조건: 급격한 머리 하강 | 1 s 보조 TCN + 최근 0.5 s 하강 동작 | `--fast-onset-risk`, `configs/ensembles/fall_onset_tcn_v32_1s.json`, `src/features/fast_onset.py` |
| 보조 조건: 조향 관여 이탈 | 손목 특징 13차원 TCN, 5회 연속 이탈 시 후보 | `configs/ensembles/control_engagement_tcn_1p5s.json`, `src/models/control_engagement_tcn.py` |
| VLM | Qwen3-VL-2B-Instruct, temperature 0, vLLM | `scripts/start_vllm.sh`, `src/bridge/vlm_bridge_server.py` |
| VLM 입력 | 후보 직전 1.5 s 운전자 crop 영상 + TCN 점수 추이 | `--candidate-seconds 1.5`, `--event-media-format video`, `prompt_with_tcn_context()` |
| 비동기·중복 억제 | VLM은 별도 스레드 1개에서 실행, 추론 중에는 새 후보를 보내지 않음, 같은 사건은 1 s 간격으로 재확인 | `ThreadPoolExecutor(max_workers=1)`, `--qwen-recheck-sec 1.0` |

## 2. 32차원 입력 특징

| 그룹 | 특징 |
| --- | --- |
| 눈 종횡비(EAR) 9 | `ear_left`, `ear_right`, `ear_mean`, `eye_valid`, `ear_asymmetry`, `ear_velocity`, `eye_closed`, `eye_closed_duration`, `perclos_1s` |
| 머리 자세 7 | `head_pitch`, `head_yaw`, `head_roll`, `pitch_velocity`, `yaw_velocity`, `roll_velocity`, `head_angular_speed` |
| 머리–어깨·상체 5 | `head_to_shoulder_dx`, `head_to_shoulder_dy`, `head_relative_vx`, `head_relative_vy`, `upper_body_motion` |
| 관측 신뢰도 3 | `face_valid`, `face_reliability`, `pose_reliability` |
| 눈 깜빡임 blendshape 4 | `eye_blink_left`, `eye_blink_right`, `eye_blink_mean`, `eye_blendshape_valid` |
| 좌석 기준 머리 위치 4 | `seat_head_dx`, `seat_head_dy`, `seat_head_vx`, `seat_head_vy` |

특징 순서는 학습된 가중치와의 계약입니다. 순서가 다른 체크포인트를 넣으면 실행기가 즉시 오류로 중단합니다.

## 3. VLM 후보 종류

`events.jsonl`의 `trigger` 필드에 기록됩니다.

| trigger | 발생 조건 |
| --- | --- |
| `main_tcn` | 주 TCN과 눈 상태를 합친 위험 점수가 임계값 이상으로 지속 |
| `control_reduced_tcn` | 조향 관여 TCN이 손 이탈을 연속 확인 (반응 감소 후보) |
| `fall_onset_tcn` | 보조 TCN과 머리 하강 동작이 함께 감지 |
| `eye_uncertain_watchdog` | 눈 관측이 불확실한 상태에서 TCN 점수가 경계 수준 |

## 4. VLM 출력

브리지는 Qwen3-VL 응답을 다음 필드로 정규화하고, 마지막 필드를 최종 판정으로 씁니다.

`eye_state`, `head_pose`, `upper_body_posture`, `hand_on_wheel`, `voluntary_motion`, `recovery_status`, `driver_response` (`active` / `reduced` / `no_visible_response`)

## 5. 실행 결과 파일

| 시스템 | 파일 | 내용 |
| --- | --- | --- |
| 공통 | `outputs/<시스템>/<영상>/tcn_timeline_*.csv` | 10 Hz 타임라인: TCN 확률, 앙상블 멤버별 확률, 눈 상태, 보조 TCN, 추론 시간 |
| TCN–VLM–YOLO | `outputs/tcn_vlm_yolo/<영상>/events.jsonl` | 후보마다 trigger, TCN 확률, VLM 판정, 모델 호출·전체 지연 |
| | `outputs/tcn_vlm_yolo/tcn_vlm_yolo_accuracy{.png,_matched.csv}` | 후보 단위 정답 비교 |
| VLM 단독 | `outputs/vlm_only/<영상>/vlm_clips.jsonl` | 1.5 s 구간마다 VLM 판정과 지연 |
| TCN 단독 | `outputs/tcn_only/<영상>/tcn_driver_states_*.jsonl` | 10 Hz TCN 3상태 판정 |
| 평가 | `outputs/evaluation/system_comparison{.png,.csv}`, `system_comparison_audit.json` | 세 시스템 공통 구간 비교 (`./scripts/evaluate.sh --latest`) |

## 6. 평가 방식

`src/evaluation/compare_systems.py`

- **공통 구간**: VLM 단독 실행의 1.5 s 비중첩 구간(192개)을 기준으로 삼고, 구간 중앙 시각의 정답 라벨과 비교합니다.
- **이진 판정**: 반응 부재(`no_visible_response`)를 양성, 나머지를 음성으로 봅니다. TCN 단독은 구간 끝의 최신 판정, 제안 구조는 구간 안에 반응 부재 VLM 판정이 하나라도 있으면 양성이며 VLM 호출이 없는 구간은 음성입니다.
- **사건 단위**: 연속된 양성 구간을 하나의 사건으로 묶고, 정답 사건과 0.5 s 이상 겹치면 일대일로 대응시킵니다.
- **지연시간**: 각 시스템의 전체 모델 호출(TCN 순전파, VLM 단독 호출, 후보 VLM 호출)의 평균입니다. YOLO·MediaPipe·영상 디코딩은 포함하지 않습니다.

## 7. 학습 코드

`src/models/`의 파일은 실행기가 모델 구조를 불러오는 모듈이자 학습 스크립트입니다(`python src/models/<file>.py --help`). 학습에 쓴 window 데이터(`data/windows/*.npz`)는 원본 영상에서 만든 것이라 저장소에 포함하지 않았습니다. 각 앙상블 JSON의 `validation_metrics`와 `limitations` 항목에 검증 결과와 데이터 한계를 기록해 두었습니다.
