# LLM control bridge

이 디렉터리는 CARLA `Town04_Opt`의 `town06_ego` telemetry와 DGX SPARK의
Qwen3-VL-2B 운전자 상태 추론을 연결한다.

프로젝트 전체 구조, CARLA 실행 순서, SSH tunnel, Qwen/vLLM, Safety Supervisor,
갓길 정차 조건, 제어권 규칙, 테스트와 문제 해결은 상위 문서를 참고한다.

→ [프로젝트 전체 README](../README.md)

핵심 실행 파일:

- `start_dgx_qwen3_vl_vllm.sh`: DGX Qwen vLLM
- `start_dgx_qwen3_vl_bridge.sh`: DGX monitor bridge
- `run_driver_safety_pipeline.py`: cabin image → `/monitor`
- `carla_telemetry_publisher.py`: CARLA telemetry → `/telemetry`
- `carla_safety_supervisor.py`: driver state → MRM/recovery override
- `test_bridge.py`: bridge와 planner 테스트

표준 CARLA 주행 스택에서는 `run_behavior_autopilot.py`만 차량에
`apply_control()`을 호출한다. `carla_llm_overlay.py`와 PCLA/SimLingo runner는
legacy 실험용이므로 표준 스택과 동시에 실행하지 않는다.
