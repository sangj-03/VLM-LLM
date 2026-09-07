# Asset policy

The repository contains source code, small example labels, documentation, and test code. The following research assets are intentionally excluded from Git history:

| Asset | Why excluded | How to supply it locally |
| --- | --- | --- |
| In-cabin MP4 recordings | Large and may contain identifiable people. | Use consented research footage stored outside Git. Pass its path with `--video`. |
| Frame crops and event clips | Derived from source video and can be large/sensitive. | Regenerate from authorised recordings. |
| TCN `.pt` checkpoints and ensemble JSON | Version-specific binary artefacts; checkpoints may exceed GitHub limits. | Train locally or retrieve from controlled project storage. |
| YOLO/MediaPipe/eye-classifier weights | Third-party or large model assets. | Download from their official sources; document versions locally. |
| Qwen3-VL/vLLM model cache | Large third-party model distribution. | Obtain through the model provider and run on the DGX host. |
| CARLA binaries/maps | Large third-party simulator distribution. | Install a CARLA release matching the Python API. |
| Tokens, passwords, SSH keys, and tunnels | Secret material. | Set environment variables locally; never put them in Git. |

GitHub rejects individual files larger than 100 MB and is not an appropriate host for the raw video and model assets above. If a vetted team needs a versioned large asset, use a private release store or Git LFS only after confirming its licence, retention policy, and access controls.
