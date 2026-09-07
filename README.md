# TCN–VLM Driver Response Monitoring and CARLA MRM

This repository contains the reproducibility package for a two-stage, in-cabin driver-response monitoring prototype and its CARLA minimum-risk manoeuvre (MRM) integration.

SEA:ME @ Korea 3기 VLM/LLM project.

> **Research prototype only.** This code is not a medical diagnostic system and is not suitable for use on public roads. It classifies visual evidence of driver response as `active`, `reduced`, or `no_visible_response`.

## What the system does

1. **TCN fast path**: extracts a time series of driver features and flags a possible non-active event.
2. **VLM verification path**: sends short, driver-centred visual evidence to a Qwen vision-language model, which produces one of the three response labels.
3. **Deterministic safety layer**: validates freshness and response observations, displays a warning for `reduced`, and begins a conservative CARLA MRM only after the configured `no_visible_response` condition.

The VLM never generates steering, throttle, or braking commands. In the CARLA integration, only `run_behavior_autopilot.py` calls `vehicle.apply_control()`; the supervisor supplies a time-limited override.

```text
Cabin video
  -> driver ROI / landmarks / eye-state evidence
  -> causal TCN candidate detector
  -> short evidence clip -> Qwen3-VL response verification
  -> deterministic safety FSM
  -> CARLA warning / safe shoulder search / controlled stop
```

## Repository layout

| Path | Contents |
| --- | --- |
| `driver_monitoring/feature_extraction` | Driver selection and MediaPipe-derived feature extraction utilities. |
| `driver_monitoring/training` | Causal binary TCN training implementation. |
| `driver_monitoring/streaming` | Experimental real-time TCN/VLM runtime. |
| `driver_monitoring/verification` | Rule-based conversion of TCN/VLM evidence into the three response labels. |
| `mrm_carla/bridge` | DGX bridge, monitor worker, CARLA telemetry publisher, safety supervisor, and tests. |
| `mrm_carla/town04_autopilot` | CARLA Town04_Opt ego spawning and single-actuator behavior-autopilot stack. |
| `data/labels` | Small ground-truth label example only. |
| `docs` | Reproduction notes, asset policy, and presentation script. |
| `results` | Place locally generated evaluation outputs here; it is ignored by Git except for documentation. |

## Quick start: static checks

The following check needs no CARLA server, DGX, model weights, or video data.

```bash
git clone <YOUR-GITHUB-URL>/VLM-LLM.git
cd VLM-LLM
python3 -m py_compile \
  driver_monitoring/feature_extraction/*.py \
  driver_monitoring/training/*.py \
  driver_monitoring/verification/*.py \
  mrm_carla/bridge/*.py \
  mrm_carla/town04_autopilot/*.py
```

For the CARLA bridge/safety logic tests, install a compatible CARLA Python API and expose its `PythonAPI` directory on `PYTHONPATH`, then run:

```bash
PYTHONPATH=/path/to/CARLA/PythonAPI \
python3 -m unittest -v mrm_carla/bridge/test_bridge.py
```

## Full CARLA + DGX reproduction

The full run requires two machines: a CARLA host and a DGX host running Qwen3-VL through vLLM. See [docs/REPRODUCTION.md](docs/REPRODUCTION.md) for the ordered commands and [docs/ASSET_POLICY.md](docs/ASSET_POLICY.md) for the intentionally excluded videos and checkpoints.

## Experimental results and interpretation

The current presentation materials report a 10-model, 2.5 s TCN ensemble with 32 visual features. On the reported held-out window set, it obtained 98.66% accuracy and F1 95.30% for binary candidate detection. The reported event-triggered VLM evaluation obtained 44/53 correct three-class decisions (83.02%) with mean VLM inference time about 1.31 s.

Those scores describe different tasks and **must not be compared as one accuracy**: the TCN number is window-level binary candidate detection; the VLM number is final three-class classification on TCN-triggered events. The data are limited and include staged behaviour, so external validation across drivers, camera positions, lighting, and normal actions remains necessary.

## Safety boundaries

- `no_visible_response` is a visual-response label, not a diagnosis of unconsciousness or medical incapacity.
- The MRM implementation is a CARLA simulation safety supervisor, not a production road planner.
- A shoulder/parking candidate is accepted only after lane, road-surface, obstacle, and corridor checks. If none is acceptable, the controller does not force a shoulder transition.
- Never commit credentials. Use the provided [`.env.example`](.env.example) as a variable-name template only.

## Attribution and licensing

This repository intentionally does not redistribute CARLA, Qwen, MediaPipe task bundles, MRL Eye Dataset data, trained checkpoints, or input videos. Obtain and use those dependencies under their own licences and terms. Add a project licence before making third-party reuse claims.
