---
title: Synapse Music V12
emoji: 🎵
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.2.0
python_version: 3.11
pinned: false
models:
  - SYNAPSEai1/SynapseMusicV12
  - SYNAPSEai1/SynapseMusicV12-XL-Turbo
  - SYNAPSEai1/SynapseMusicV12-XL-Base
  - SYNAPSEai1/SynapseMusicV12-XL-SFT
  - SYNAPSEai1/SynapseMusicV12-Composer-4B
  - SYNAPSEai1/SynapseMusicV12-Composer-0.6B
  - SYNAPSEai1/SynapseMusicV12-Captioner
  - SYNAPSEai1/SynapseMusicV12-Transcriber
  - SYNAPSEai1/SynapseMusicV12-VAE
license: mit
app_file: app.py
short_description: Full-song AI music generation, remixing.
---

<h1 align="center">Synapse Music V12</h1>

<h2 align="center">Full-Song AI Music Generation & Production</h2>

<p align="center">
Generate complete songs, vocals, lyrics, instrumentals, covers, remixes and editable music using the Synapse Music V12 model ecosystem.
</p>

---

## Synapse Music V12

**Synapse Music V12** is an AI music generation and production system built on top of the **Synapse Music V11** architecture.

The goal of Synapse Music is to provide a complete music-generation workflow rather than only basic text-to-audio generation.

Synapse Music V12 is designed around:

- full-song generation
- vocal generation
- instrumental generation
- custom lyrics
- automatic composition planning
- prompt expansion
- reference audio
- cover generation
- remixing
- repainting
- song completion
- multi-track generation
- stem extraction
- audio understanding
- BPM/key control
- LoRA training
- custom Synapse music models

The underlying Synapse Music V12 architecture provides efficient diffusion-based music generation combined with a language-model composition planner.

---

# Features

## Music Generation

- Complete song generation
- Instrumental generation
- Vocal music generation
- Custom lyrics
- Simple prompt generation
- Advanced/custom generation
- Up to 10-minute compositions
- 48 kHz stereo generation
- Batch generation
- 50+ lyric languages
- Detailed style prompting
- Instrument and production control

---

## Composition Intelligence

Synapse Music can use a dedicated Composer model to transform a simple request into a much more detailed musical blueprint.

Example:

```text
195 BPM brutal industrial hardtekk,
male distorted vocals,
huge kick switches
```

can be expanded into structured generation information including:

```text
Genre
Subgenres
BPM
Key
Time signature
Mood
Instrumentation
Vocal characteristics
Production characteristics
Song structure
Lyrics
Detailed generation caption
```

The resulting composition plan is then supplied to the music generation model.

---

# Synapse Music V12 Model Architecture

```text
                    SYNAPSE MUSIC V12
                           │
                           ▼
                  Synapse Composer
                           │
                 ┌─────────┴─────────┐
                 │                   │
              Simple              Custom
                 │                   │
                 └─────────┬─────────┘
                           ▼
                    Generation API
                           │
            ┌──────────────┼──────────────┐
            │              │              │
            ▼              ▼              ▼
        XL Turbo        XL Base         XL SFT
          Fast           Studio         Training
            │              │              │
            └──────────────┼──────────────┘
                           ▼
                     Audio Output
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
       Library           Studio          Download
```

---

# Model Family

## Synapse Music V12

Primary Synapse Music repository.

```text
SYNAPSEai1/SynapseMusicV12
```

---

## Synapse Music V12 XL Turbo

Primary fast/high-quality generation model.

```text
SYNAPSEai1/SynapseMusicV12-XL-Turbo
```

Intended for:

- standard song generation
- fast generations
- production inference
- interactive generation
- batch generation

---

## Synapse Music V12 XL Base

Foundation model for advanced generation and editing.

```text
SYNAPSEai1/SynapseMusicV12-XL-Base
```

Intended for:

- Studio
- advanced editing
- Cover
- Repaint
- Extract
- Lego / multi-track generation
- Complete
- experimentation
- fine-tuning

---

## Synapse Music V12 XL SFT

Supervised fine-tuned foundation checkpoint.

```text
SYNAPSEai1/SynapseMusicV12-XL-SFT
```

Intended for:

- high-quality generation
- experimentation
- Synapse fine-tuning
- custom checkpoints

---

# Composer Models

## Synapse Music V12 Composer 4B

```text
SYNAPSEai1/SynapseMusicV12-Composer-4B
```

Primary high-quality composition model.

Designed for:

- prompt rewriting
- song planning
- metadata generation
- lyric planning
- audio understanding
- composition reasoning
- melody-related conditioning

---

## Synapse Music V12 Composer 0.6B

```text
SYNAPSEai1/SynapseMusicV12-Composer-0.6B
```

Lightweight Composer option for environments where memory usage and generation speed are more important.

---

## Composer 1.7B

The SynapseMusicV12 1.7B LM is included through the main Synapse Music V12 model package rather than maintained as a separate Synapse repository.

---

# Dataset Intelligence

## Synapse Music V12 Captioner

```text
SYNAPSEai1/SynapseMusicV12-Captioner
```

Used for automatic music description and dataset preparation.

Potential metadata includes:

- genre
- subgenre
- mood
- instrumentation
- arrangement
- production characteristics
- vocal characteristics
- musical structure

---

## Synapse Music V12 Transcriber

```text
SYNAPSEai1/SynapseMusicV12-Transcriber
```

Used for audio transcription and training-data preparation where the source material and intended use are appropriately licensed or authorized.

---

# Synapse Music Studio

The Studio system is intended to provide editing operations in addition to standard generation.

Supported capabilities include:

| Feature | Purpose |
|---|---|
| Text2Music | Generate music from descriptions |
| Cover | Generate a new interpretation from reference audio |
| Repaint | Regenerate selected audio regions |
| Extract | Extract/separate musical components |
| Lego | Generate individual musical layers |
| Complete | Continue or complete existing music |
| Vocal2BGM | Generate accompaniment for vocals |
| Reference Audio | Condition generation using audio |
| Audio Understanding | Analyze musical characteristics |

---

# Generation Controls

Synapse Music V12 supports or is designed to expose controls for:

- prompt
- lyrics
- instrumental mode
- duration
- BPM
- key
- scale
- time signature
- reference audio
- generation seed
- inference steps
- guidance
- model selection
- Composer selection
- batch count
- thinking/composition mode

---

# Training

Synapse Music V12 is intended to support custom LoRA and fine-tuning workflows built on the Synapse Music V12 training system.

```text
Licensed / authorized audio
            │
            ▼
      Dataset ingestion
            │
     ┌──────┴───────┐
     ▼              ▼
 Captioner      Transcriber
     │              │
     └──────┬───────┘
            ▼
      Metadata cleanup
            │
            ▼
      Dataset validation
            │
            ▼
        LoRA / SFT
            │
            ▼
       Evaluation
            │
            ▼
     Synapse checkpoint
```

Training material should only be used where the required rights or permissions allow its use.

---

# Local Installation

## Requirements

- Python 3.11
- CUDA-capable GPU recommended
- CPU/MPS operation may be possible but considerably slower

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the Synapse Music repository:

```bash
git clone https://huggingface.co/spaces/SYNAPSEai1/Synapse-Music-V12
cd Synapse-Music-V12
```

Install dependencies:

```bash
uv sync
```

---

# Launch Synapse Music

Start the Gradio application:

```bash
uv run synapse-music
```

Default local interface:

```text
http://localhost:7860
```

The upstream-compatible command remains available:

```bash
uv run acestep
```

---

# REST API

Start the API:

```bash
uv run synapse-music-api
```

Default API address:

```text
http://localhost:8001
```

The upstream-compatible command is also retained:

```bash
uv run acestep-api
```

See:

```text
docs/en/API.md
```

for the existing Synapse Music V12 API documentation.

---

# Advanced Launch Options

Example:

```bash
uv run synapse-music \
  --server-name 0.0.0.0 \
  --port 7860 \
  --init_service true
```

Common options inherited from Synapse Music V11 include:

| Option | Description |
|---|---|
| `--port` | Gradio server port |
| `--server-name` | Server bind address |
| `--share` | Enable Gradio public sharing |
| `--language` | Interface language |
| `--init_service` | Initialize models during startup |
| `--config_path` | Select DiT model |
| `--lm_model_path` | Select Composer/LM |
| `--offload_to_cpu` | Enable CPU model offloading |

---

# Development

Add a dependency:

```bash
uv add package-name
```

Add a development dependency:

```bash
uv add --dev package-name
```

Update dependencies:

```bash
uv sync --upgrade
```

---

# Upstream Project

Synapse Music V12 is built using **Synapse Music V11** as its music-generation foundation.


---

# License

The upstream Synapse Music V11 project is distributed under the MIT License.

Synapse Music V12 retains required upstream notices and attribution.

Before distributing custom trained checkpoints, datasets or generated material, verify the applicable licenses and rights for all underlying components and training material.

---

# Status

**Synapse Music V12**

Current stage:

```text
✓ Synapse Music V12 foundation
✓ XL Turbo
✓ XL Base
✓ XL SFT
✓ Composer models
✓ Captioner
✓ Transcriber
✓ VAE
✓ Generation Space
→ Synapse model wiring
→ Synapse API
→ Custom Create interface
→ Synapse Studio
→ Training pipeline
→ Synapse LoRAs
→ Synapse fine-tuned checkpoints
```

The current V12 generation system is being developed from the validated Synapse Music V12 baseline.