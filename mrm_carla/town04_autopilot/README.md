# Town04_Opt single-actuator safety autopilot

기존 셸 습관을 유지하기 위해 `carla06*` 명령 이름은 호환용으로 남겨 두었지만,
네 명령은 모두 이 디렉터리의 `Town04_Opt` 구성만 실행한다. Ego role은
`town06_ego`, safety handoff 파일은
`/tmp/carla_town04_ego_safety_override.json`이다.

Open three terminals together:

```bash
carla06all
```

Or run each role manually in a separate terminal:

```bash
carla06
carla06spawn
carla06auto
```

정상 주행만 실행:

```bash
carla06auto
```

빨간색/노란색 신호는 정지선 또는 횡단보도 바로 앞에 정지한다. 해당 표시가
없으면 현재 차선의 끝 바로 앞을 정지 목표로 사용한다.

DGX/VLM을 완전히 우회하고 결정론적 갓길 MRM 시험:

```bash
carla06auto --force-mrm
```

DGX의 `driver_response`를 사용한 통합 주행:

```bash
export LLM_BRIDGE_TOKEN='DGX와 동일한 토큰'
carla06auto --with-driver-monitor
```

표준 `carla06auto` 스택에서는 `run_behavior_autopilot.py` 하나만 매 tick
`vehicle.apply_control()`을 호출한다. Supervisor는 원자적으로 safety command
파일을 갱신할 뿐 차량 제어를 직접 적용하지 않는다. Spawner의 1회 초기
핸드브레이크는 주행 actuator가 시작되기 전의 초기화다. Legacy
`run_pcla_simlingo.py`와 `carla_llm_overlay.py`는 독립 실험 runner이므로 표준
스택이 실행 중일 때 함께 실행하지 않는다. 1초보다 오래된 override는 무시한다.

The second terminal owns only the `town06_ego` vehicle. The third terminal
imports CARLA's original `BehaviorAgent` and only applies its controls to that
vehicle. NPC vehicles and pedestrians are not spawned by default; pass
`--npc-vehicles` or `--npc-walkers` to `spawn_ego_vehicle.py` when they are
needed.
Stop in reverse order with Ctrl+C: autopilot, vehicle, then CARLA.

Spawner는 기본적으로 차량 폭과 여유 폭을 만족하는 Shoulder로 연결되는 spawn을
우선 선택한다. 특정 위치를 재현하려면 `carla06spawn --spawn-index N`을 사용한다.
NPC는 MRM이 빈 도로에서 성공한 뒤 `--npc-vehicles`와 `--npc-walkers`로 추가한다.
