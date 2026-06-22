# Player Highlight Tool

Generate per-player highlight reels from football match videos.

## Setup

### 1. Install Python dependencies

Install base packages first, then Re-ID (needs numpy in the environment before build):

```bash
pip uninstall torchreid -y
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-reid.txt
```

Verify Re-ID:

```bash
python -c "from torchreid.models import build_model; print('OK')"
```

**Note:** Requires Python 3.10+ and a CUDA-capable GPU (tested on RTX 4070 Ti SUPER).

### 2. Place model files

Copy these three model files into the `models/` directory:

| File | What it is |
|------|-----------|
| `yolo11m.pt` | YOLO v11 medium — player detection |
| `yolo-football-ball-detection.pt` | YOLO — ball detection |
| `reid_model_soccernet.pth` | OSNet Re-ID pretrained on SoccerNet |

### 3. Configure OpenAI API key (optional)

Copy `.env.example` to `.env` and add your OpenAI API key:

```bash
cp .env.example .env
# Edit .env and add your key
```

The GPT verification is optional — the tracker works without it. With it enabled, every 30 seconds GPT-4.1 checks if the tracker is still on the correct player using zoomed comparison crops.

**Budget:** ~$4 per match (~265 GPT calls at $0.015/call).

---

## Usage

### Interactive (local desktop)

```bash
python tracker.py --video path/to/match.mp4 --start 1850 --duration 30
```

### Headless (RunPod / SSH, no display)

No GUI required. Provide the player location at the `--start` frame using either a bounding box or a click point.

```bash
# Option A: exact bounding box (x,y,width,height)
python tracker.py --headless --video path/to/match.mp4 --start 1850 --duration 30 \
  --bbox 820,340,55,120 --jersey 10 --team red

# Option B: pick nearest detected player to a pixel (e.g. frame center)
python tracker.py --headless --video path/to/match.mp4 --start 1850 --duration 30 \
  --pos 960,540 --jersey 10 --team red
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | required | Path to match video file |
| `--start` | `0` | Start time in seconds |
| `--duration` | `30` | How many minutes to process |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--headless` | off | Run without OpenCV window (for cloud/SSH) |
| `--bbox` | — | Initial player box as `x,y,w,h` at `--start` frame (headless) |
| `--pos` | — | Pick nearest YOLO player to `x,y` at `--start` frame (headless) |
| `--jersey` | — | Jersey number (skips interactive prompt) |
| `--team` | — | Team color (skips interactive prompt) |
| `--progress-every` | `30` | Log progress every N seconds in headless mode |

In headless mode the tracker starts immediately and runs until the duration limit. Reference crops for GPT are saved automatically when an API key is set.

### Controls (interactive mode only)

| Key | Action |
|-----|--------|
| `S` | Select player (draw bounding box around them) |
| `K` | Save reference crop for GPT verification (max 3) |
| `SPACE` | Pause / Play |
| `A` / `D` | Skip back / forward 5 seconds |
| `Q` | Quit and merge highlights |

### Workflow

1. Video opens paused. Press `S`, draw a box around your player.
2. Press `K` 2-3 times while tracking to save reference crops (helps GPT verification).
3. Press `SPACE` to play. The tracker follows the player automatically.
4. Highlights are auto-recorded when the ball is near the player (3s pre-roll, 4s post-roll).
5. Press `Q` when done. Clips are merged into `output/highlight_reel.mp4` with fade transitions.

### Visual Indicators

- **Green box** = actively tracking
- **Yellow box + "CROSSING"** = another player overlapping (identity frozen, tracker holds through it)
- **Orange "OFFSCREEN"** = player left the frame, waiting for camera to pan back
- **Red "REC" dot** = recording a highlight clip
- **"BALL" label** = ball detected near player

---

## Directory Structure

```
player_highlight_tool/
  .env.example          # Template for OpenAI API key
  .env                  # Your API key (create from .env.example)
  requirements.txt      # Python dependencies
  HOW_TO_USE.md         # This file
  tracker.py            # Auto-tracking highlight generator
  models/               # Model files (you provide these)
    yolo11m.pt
    yolo-football-ball-detection.pt
    reid_model_soccernet.pth
  output/               # Generated clips and final highlight reel
    clip_001.mp4
    clip_002.mp4
    ...
    highlight_reel.mp4
```

---

## Tuning

Key parameters at the top of `tracker.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BALL_NEAR` | `100` | Pixel margin around player to count "ball touching" — increase if ball touches are missed |
| `BALL_CONF` | `0.20` | Ball detection confidence — lower = more detections but more false positives |
| `PRE_ROLL_SEC` | `3.0` | Seconds recorded before each ball touch |
| `POST_ROLL_SEC` | `4.0` | Seconds recorded after ball leaves |
| `COLOR_GATE` | `0.35` | Team color matching threshold — lower = stricter same-team filtering |
| `REID_THRESHOLD` | `0.50` | Re-ID score needed to re-acquire player after loss |

## Troubleshooting

**`ModuleNotFoundError: No module named 'numpy'` when installing deep-person-reid**

Install base deps first (includes numpy, Cython, tensorboard), then Re-ID:

```bash
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-reid.txt
```

**`ModuleNotFoundError: No module named 'tensorboard'` when installing deep-person-reid**

Same fix — `tensorboard` must be installed before the Re-ID package (setup imports the full library):

```bash
pip install tensorboard h5py six yacs
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-reid.txt
```

**`ModuleNotFoundError: gdown` / `tensorboard` / other torchreid import errors**

You likely installed the wrong PyPI `torchreid` package. Replace it with the official build:

```bash
pip uninstall torchreid -y
pip install -r requirements.txt
pip install --no-build-isolation -r requirements-reid.txt
```

Verify:

```bash
python -c "from torchreid.models import build_model; print('OK')"
```

## Known Limitations

- **Same-team player swaps** can happen during close crossings — the tracker uses Re-ID + team color but identical kits make same-team players hard to distinguish at 50-130px.
- **Ball detection is ~40% reliable** on wide-angle amateur footage — some touches will be missed and a few false triggers occur.
- **CSRT tracker is CPU-only** — this is the main speed bottleneck. YOLO and Re-ID run on GPU.
- **GPU strongly recommended** — runs ~1 FPS on CPU vs near real-time on a modern NVIDIA GPU.
