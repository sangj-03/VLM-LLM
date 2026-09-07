# Reproduction guide

## Prerequisites

- Linux, Python 3.8 or later for the prototype scripts
- CARLA 0.9.15 with a `Town04_Opt` map and a matching Python API
- A CUDA-capable DGX/Linux host for Qwen3-VL and vLLM
- A Qwen3-VL model installation compatible with the scripts in `mrm_carla/bridge`
- A TCN checkpoint ensemble and its JSON configuration, created locally or obtained separately
- An in-cabin video for research use

Install the non-CARLA Python dependencies first:

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

CARLA is intentionally not listed as a generic pip dependency: use the Python API paired with the installed CARLA server release.

## 1. Feature extraction and TCN training

The feature extractor expects a MediaPipe task bundle, a YOLO detector weight, and a JSON configuration. These are large or version-specific assets and are not committed. Place them outside version control, then run from the feature-extraction directory so its local imports resolve:

```bash
cd driver_monitoring/feature_extraction
python 03_extract_features.py \
  --video /path/to/cabin.mp4 \
  --model /path/to/holistic_landmarker.task \
  --config /path/to/config.json \
  --yolo-model /path/to/yolo26n.pt \
  --out ../../results/features/cabin.csv
```

Create the labelled train/validation arrays using the same feature ordering in `common.py`; then train the causal binary TCN:

```bash
cd ../training
python 17_train_binary_tcn.py --help
```

The exact split, labels, thresholds, and checkpoint ensemble are experimental artefacts. Keep them in an access-controlled storage location and record their provenance before publishing a result.

## 2. Start the DGX VLM service

On the DGX host, configure secrets only in the shell or a local, ignored `.env` file:

```bash
cd mrm_carla/bridge
export VLLM_API_KEY='replace-locally'
export LLM_BRIDGE_TOKEN='replace-with-a-long-shared-secret'
./start_dgx_qwen3_vl_vllm.sh
```

In another terminal on the DGX host:

```bash
cd mrm_carla/bridge
export VLLM_API_KEY='replace-locally'
export LLM_BRIDGE_TOKEN='replace-with-the-same-secret'
./start_dgx_qwen3_vl_bridge.sh
```

## 3. Start CARLA and create the ego vehicle

On the CARLA host, point `PYTHONPATH` to the Python API that matches the CARLA server. Start the CARLA server, spawn the ego vehicle, and start the integrated stack in separate terminals:

```bash
cd mrm_carla/town04_autopilot
./run_carla_town04_opt.sh
./spawn_ego_vehicle.py --role-name town06_ego
export LLM_BRIDGE_TOKEN='replace-with-the-same-secret'
./run_autopilot_stack.sh --with-driver-monitor
```

The exact CARLA host/DGX port mapping is deployment-specific. Protect the bridge with a non-committed `LLM_BRIDGE_TOKEN` and expose it through a controlled SSH tunnel or protected network only.

## 4. Run the monitor worker

The monitor worker sends new cabin images to the bridge. Use an atomically-written image path in real-time mode:

```bash
cd mrm_carla/bridge
export LLM_BRIDGE_TOKEN='replace-with-the-same-secret'
python3 run_driver_safety_pipeline.py \
  --image-path /path/to/latest-cabin.jpg \
  --watch \
  --monitor-period-sec 4
```

## 5. Safe validation sequence

1. Run bridge unit tests.
2. Start CARLA with no traffic and use `--force-mrm` to test only the safety planner.
3. Confirm that the candidate surface, shoulder corridor, stop condition, and handback logs match the intended scenario.
4. Add traffic only after the no-traffic scenario is repeatable.

Do not run legacy direct-control scripts (`carla_llm_overlay.py` or `run_pcla_simlingo.py`) alongside `run_autopilot_stack.sh`; doing so violates the single-actuator design.
