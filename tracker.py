"""
Player Highlight Tracker
========================
Tracks a single player through a full match and auto-records highlight clips
when the ball is near the tracked player.

Uses CSRT tracker + YOLO crossing detection + OSNet Re-ID verification +
optional GPT-4.1 periodic identity checks.

Usage:
    python tracker.py --video path/to/match.mp4 [--start 1850] [--duration 30]

Headless (SSH / RunPod, no display):
    python tracker.py --headless --video match.mp4 --start 1850 --duration 30 \\
        --bbox 820,340,55,120 --jersey 10 --team red
    python tracker.py --headless --video match.mp4 --start 1850 --pos 960,540

Controls (interactive mode only):
    S       = Select player (draw bounding box)
    K       = Save reference crop for GPT verification (max 3)
    SPACE   = Pause / Play
    A / D   = Skip back / forward 5 seconds
    Q       = Quit and merge highlights
"""

import os, sys, argparse
os.environ["PYTHONIOENCODING"] = "utf-8"

from dotenv import load_dotenv
load_dotenv()

import cv2, torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as T
from ultralytics import YOLO
from collections import deque
from pathlib import Path

try:
    from torchreid.models import build_model as build_reid_model
except ImportError:
    print("ERROR: torchreid is not installed correctly.")
    print("Do NOT use the PyPI 'torchreid' package — it is incompatible.")
    print("Install the official source build instead:")
    print("  pip uninstall torchreid -y")
    print("  pip install git+https://github.com/KaiyangZhou/deep-person-reid.git")
    sys.exit(1)

# ── CLI args ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Player Highlight Tracker")
parser.add_argument("--video", required=True, help="Path to match video")
parser.add_argument("--start", type=int, default=0, help="Start time in seconds")
parser.add_argument("--duration", type=int, default=30, help="Duration in minutes")
parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
parser.add_argument("--headless", action="store_true",
                    help="Run without GUI (requires --bbox or --pos)")
parser.add_argument("--bbox", type=str,
                    help="Initial player box as x,y,w,h pixels at --start frame")
parser.add_argument("--pos", type=str,
                    help="Headless: pick nearest detected player to x,y at --start frame")
parser.add_argument("--jersey", default="", help="Jersey number (skips prompt)")
parser.add_argument("--team", default="", help="Team color (skips prompt)")
parser.add_argument("--progress-every", type=int, default=30,
                    help="Headless: log progress every N seconds")
args = parser.parse_args()

HEADLESS = args.headless

def parse_xy(s, name):
    try:
        parts = [int(p.strip()) for p in s.split(",")]
    except ValueError:
        parser.error(f"{name} must be two integers: x,y")
    if len(parts) != 2:
        parser.error(f"{name} must be two integers: x,y")
    return parts[0], parts[1]

def parse_bbox(s):
    try:
        parts = [int(p.strip()) for p in s.split(",")]
    except ValueError:
        parser.error("--bbox must be four integers: x,y,w,h")
    if len(parts) != 4 or parts[2] <= 0 or parts[3] <= 0:
        parser.error("--bbox must be four integers: x,y,w,h (w and h > 0)")
    return tuple(parts)

if HEADLESS and not args.bbox and not args.pos:
    parser.error("--headless requires --bbox x,y,w,h or --pos x,y")
if not HEADLESS and (args.bbox or args.pos):
    parser.error("--bbox and --pos are only valid with --headless")

_DIR = Path(__file__).resolve().parent
VIDEO_PATH      = args.video
MODELS_DIR      = _DIR / "models"
REID_MODEL_PATH = str(MODELS_DIR / "reid_model_soccernet.pth")
YOLO_PATH       = str(MODELS_DIR / "yolo11m.pt")
BALL_PATH       = str(MODELS_DIR / "yolo-football-ball-detection.pt")
DEVICE          = args.device
START_SEC       = args.start
RUN_MINUTES     = args.duration

# ── Tracking config ───────────────────────────────────────────────────────────
YOLO_CONF       = 0.12
YOLO_IMGSZ      = 1920
COLLISION_CHECK = 4
OVERLAP_IOU     = 0.15
REID_THRESHOLD  = 0.50
VERIFY_THRESH   = 0.40
LOST_FRAMES     = 12
TEMPLATE_UPDATE = 3
TEMPLATE_ALPHA  = 0.1
COLOR_GATE      = 0.35
EDGE_MARGIN     = 45
OFFSCREEN_MAX   = int(30 * 30)

# ── Ball detection ──
BALL_CONF       = 0.20
BALL_IMGSZ      = 960
BALL_NEAR       = 100

# ── Highlight recording ──
PRE_ROLL_SEC    = 3.0
POST_ROLL_SEC   = 4.0

# ── Player identity ──
if args.jersey or args.team or HEADLESS:
    PLAYER_JERSEY = args.jersey.strip()
    PLAYER_TEAM = args.team.strip()
else:
    PLAYER_JERSEY = input("Jersey number (e.g. 10): ").strip()
    PLAYER_TEAM = input("Team color (e.g. red, white, blue): ").strip()

# ── GPT identity confirmation (optional) ──
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "")
GPT_COST_CALL   = 0.015
GPT_BUDGET      = 4.00
HAS_GPT = False
gpt_client = None
if OPENAI_KEY:
    try:
        from openai import OpenAI
        import base64
        gpt_client = OpenAI(api_key=OPENAI_KEY)
        HAS_GPT = True
        print("GPT verification enabled.")
    except ImportError:
        print("openai package not installed — GPT verification disabled.")
else:
    print("No OPENAI_API_KEY in .env — GPT verification disabled.")
gpt_calls = 0
gpt_max_calls = int(GPT_BUDGET / GPT_COST_CALL)

# ── Models ────────────────────────────────────────────────────────────────────
for f, name in [(YOLO_PATH, "yolo11m.pt"), (BALL_PATH, "yolo-football-ball-detection.pt"),
                (REID_MODEL_PATH, "reid_model_soccernet.pth")]:
    if not os.path.exists(f):
        print(f"ERROR: Missing model file: {f}")
        print(f"Place {name} in the models/ directory.")
        sys.exit(1)

print("Loading YOLO (players)...")
yolo = YOLO(YOLO_PATH)
yolo.to(DEVICE)

print("Loading YOLO (ball)...")
ball_yolo = YOLO(BALL_PATH)
ball_yolo.to(DEVICE)

print("Loading OSNet Re-ID (SoccerNet-pretrained)...")
reid = build_reid_model('osnet_x1_0', num_classes=1000, pretrained=False)
reid.classifier = nn.Identity()
ckpt = torch.load(REID_MODEL_PATH, map_location=DEVICE, weights_only=False)
sd = {k.replace('module.',''):v for k,v in ckpt['state_dict'].items() if 'classifier' not in k}
reid.load_state_dict(sd, strict=False)
reid.to(DEVICE)
reid.eval()
print(f"  Re-ID ready (SoccerNet, rank1={ckpt['rank1']:.2%})")

transform = T.Compose([T.ToPILImage(), T.Resize((256,128)), T.ToTensor(),
                       T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])

# ── Helper functions ──────────────────────────────────────────────────────────

def embed(crop):
    if crop is None or crop.size == 0: return None
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        e = reid(transform(rgb).unsqueeze(0).to(DEVICE)).squeeze()
        return nn.functional.normalize(e, p=2, dim=0)

def csim(a,b): return float(torch.dot(a,b))

def crop_of(frame,x,y,w,h,pad=0.15):
    H,W = frame.shape[:2]; px,py=int(w*pad),int(h*pad)
    return frame[max(0,y-py):min(H,y+h+py), max(0,x-px):min(W,x+w+px)]

def jersey_hist(frame,x,y,w,h):
    H,W = frame.shape[:2]
    tx,ty = x+int(w*0.2), y+int(h*0.15)
    tw,th = int(w*0.6), int(h*0.35)
    roi = frame[max(0,ty):min(H,ty+th), max(0,tx):min(W,tx+tw)]
    if roi.size == 0: return None
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv],[0,1],None,[30,32],[0,180,0,256])
    cv2.normalize(hist,hist,0,1,cv2.NORM_MINMAX)
    return hist

def color_sim(h1,h2):
    if h1 is None or h2 is None: return 0.0
    return max(0.0, cv2.compareHist(h1,h2,cv2.HISTCMP_CORREL))

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    x1=max(ax,bx); y1=max(ay,by); x2=min(ax+aw,bx+bw); y2=min(ay+ah,by+bh)
    inter=max(0,x2-x1)*max(0,y2-y1); union=aw*ah+bw*bh-inter
    return inter/union if union>0 else 0

def detect(frame):
    res = yolo.predict(frame, classes=0, conf=YOLO_CONF, imgsz=YOLO_IMGSZ,
                       half=True, verbose=False)
    out=[]
    for r in res:
        for b in r.boxes:
            x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
            out.append((x1,y1,x2-x1,y2-y1))
    return out

def detect_ball(frame):
    res = ball_yolo.predict(frame, conf=BALL_CONF, imgsz=BALL_IMGSZ,
                            half=True, verbose=False)
    best_ball = None
    best_conf = 0
    for r in res:
        for b in r.boxes:
            conf = float(b.conf[0])
            if conf > best_conf:
                best_conf = conf
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy().astype(int)
                best_ball = ((x1+x2)//2, (y1+y2)//2, x2-x1, y2-y1)
    return [best_ball] if best_ball else []

def ball_near_player(balls, player_box, margin=BALL_NEAR):
    if not balls or player_box is None: return False, None
    px, py, pw, ph = player_box
    ex, ey = px - margin, py - margin
    ew, eh = pw + 2*margin, ph + 2*margin
    for b in balls:
        bcx, bcy = b[0], b[1]
        if ex <= bcx <= ex+ew and ey <= bcy <= ey+eh:
            return True, b
    if balls:
        return False, balls[0]
    return False, None

# ── Camera motion compensation ────────────────────────────────────────────────

def estimate_camera_motion(prev_gray, gray, boxes):
    I = np.array([[1,0,0],[0,1,0]], np.float32)
    if prev_gray is None: return I
    mask = np.full(prev_gray.shape, 255, np.uint8)
    for (bx,by,bw,bh) in boxes:
        mask[max(0,by):max(0,by+bh), max(0,bx):max(0,bx+bw)] = 0
    p0 = cv2.goodFeaturesToTrack(prev_gray, 400, 0.01, 8, mask=mask)
    if p0 is None or len(p0) < 6: return I
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
    if p1 is None: return I
    st = st.flatten() == 1
    g0, g1 = p0[st], p1[st]
    if len(g0) < 6: return I
    M, _ = cv2.estimateAffinePartial2D(g0, g1)
    return M.astype(np.float32) if M is not None else I

def apply_motion(M, pt):
    x, y = pt[0], pt[1]
    return np.array([M[0,0]*x + M[0,1]*y + M[0,2],
                     M[1,0]*x + M[1,1]*y + M[1,2]])

# ── GPT zoomed-crop confirmation ─────────────────────────────────────────────
CROP_ZOOM_H = 256

def _upscale_crop(frame, x, y, w, h, target_h=CROP_ZOOM_H):
    fH, fW = frame.shape[:2]
    pad = 0.3
    px, py = int(w * pad), int(h * pad)
    x1, y1 = max(0, x - px), max(0, y - py)
    x2, y2 = min(fW, x + w + px), min(fH, y + h + py)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return None
    scale = target_h / max(crop.shape[0], 1)
    new_w = max(1, int(crop.shape[1] * scale))
    return cv2.resize(crop, (new_w, target_h), interpolation=cv2.INTER_LANCZOS4)

def gpt_confirm(frame, target_box, all_dets, ref_crops_b64):
    global gpt_calls
    if not HAS_GPT or gpt_calls >= gpt_max_calls or not ref_crops_b64:
        return None

    H, W = frame.shape[:2]
    tx, ty, tw, th = target_box
    tcx, tcy = tx + tw/2, ty + th/2

    candidates = []
    for d in all_dets:
        dx, dy, dw, dh = d
        dcx, dcy = dx + dw/2, dy + dh/2
        dist = np.sqrt((tcx - dcx)**2 + (tcy - dcy)**2)
        if target_hist is not None:
            dhist = jersey_hist(frame, dx, dy, dw, dh)
            if color_sim(target_hist, dhist) < COLOR_GATE:
                continue
        candidates.append((dist, d))
    candidates.sort(key=lambda x: x[0])

    if candidates:
        if candidates[0][0] > 80:
            candidates.insert(0, (0, target_box))
    candidates = candidates[:6]
    if not candidates:
        candidates = [(0, target_box)]

    label_map = []
    crop_images = []
    current_num = None
    for idx, (dist, d) in enumerate(candidates):
        num = idx + 1
        is_tracked = (dist < 30) and idx == 0
        crop = _upscale_crop(frame, *d)
        if crop is None: continue
        border_color = (0, 0, 255) if is_tracked else (0, 255, 0)
        cv2.rectangle(crop, (0, 0), (crop.shape[1]-1, crop.shape[0]-1), border_color, 4)
        cv2.putText(crop, str(num), (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, border_color, 3)
        _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
        crop_images.append(base64.b64encode(buf).decode('utf-8'))
        label_map.append((num, d))
        if is_tracked:
            current_num = num

    if current_num is None and label_map:
        current_num = label_map[0][0]

    messages = [
        {"role": "system", "content": (
            "You are a sports video player-identification assistant. "
            "You will see REFERENCE crops of a target player, then ZOOMED "
            "crops of nearby same-team candidates (numbered). Compare body "
            "shape, jersey color/pattern, shorts color, socks, sleeve length, "
            "and jersey number IF readable. The tracked player has a RED "
            "border. Say YES if the red-bordered player matches the reference. "
            "ONLY say NO if a different candidate is CLEARLY a better match. "
            "When in doubt, say YES. NEVER analyze faces."
        )},
        {"role": "user", "content": [
            {"type": "text", "text": (
                f"Target: {PLAYER_TEAM} team"
                + (f", jersey #{PLAYER_JERSEY}" if PLAYER_JERSEY else "")
                + f"\n\nREFERENCE crops of the target player ({len(ref_crops_b64)}):"
            )},
            *[{"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{c}", "detail": "high"
            }} for c in ref_crops_b64],
            {"type": "text", "text": (
                f"CANDIDATE crops (all same team, zoomed). "
                f"Player {current_num} (red border) is currently tracked:"
            )},
            *[{"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{c}", "detail": "high"
            }} for c in crop_images],
            {"type": "text", "text": (
                "Is the RED-bordered player the same as the reference?\n"
                "Reply ONLY one line:\n"
                "CORRECT: YES | CONFIDENCE: high/medium/low | REASON: ...\n"
                "or:\n"
                "CORRECT: NO | ACTUAL: <number> | CONFIDENCE: high/medium/low | REASON: ..."
            )},
        ]}
    ]

    try:
        resp = gpt_client.chat.completions.create(
            model="gpt-4.1", messages=messages, max_tokens=150, temperature=0.1)
        answer = resp.choices[0].message.content
        gpt_calls += 1
        spent = gpt_calls * GPT_COST_CALL
        remaining = gpt_max_calls - gpt_calls
        print(f"    GPT raw: {answer}")

        if "CORRECT:" not in answer:
            print(f"    GPT: parse error (${spent:.2f}, {remaining} left)")
            return None

        is_correct = "YES" in answer.split("CORRECT:")[1].split("|")[0].upper()
        conf_str = "medium"
        if "CONFIDENCE:" in answer:
            conf_str = answer.split("CONFIDENCE:")[1].split("|")[0].strip().lower()

        if is_correct:
            print(f"    GPT: CONFIRMED ({conf_str}) (${spent:.2f}, {remaining} left)")
            return True

        if "ACTUAL:" in answer and conf_str == "high":
            actual_str = answer.split("ACTUAL:")[1].split("|")[0].strip()
            if actual_str.isdigit():
                idx = int(actual_str) - 1
                if 0 <= idx < len(label_map):
                    _, correct_box = label_map[idx]
                    print(f"    GPT: WRONG! Target is #{actual_str} ({conf_str}) "
                          f"(${spent:.2f}, {remaining} left)")
                    return correct_box

        print(f"    GPT: unsure ({conf_str}) (${spent:.2f}, {remaining} left)")
        return None
    except Exception as e:
        print(f"    GPT error: {e}")
        return None

def save_ref_crop(frame, player_box):
    if not HAS_GPT or player_box is None:
        return
    x, y, w, h = player_box
    crop = crop_of(frame, x, y, w, h)
    if crop.size == 0:
        return
    _, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
    ref_crops_b64.append(base64.b64encode(buf).decode('utf-8'))
    if len(ref_crops_b64) > 3:
        ref_crops_b64[:] = ref_crops_b64[-3:]
    print(f"  Reference crop saved ({len(ref_crops_b64)}/3)")

def init_tracking(frame, bb):
    global tracker, box, center, state, tracked, lost, in_crossing
    global prev_gray, offscreen_frames, recovered_from_offscreen
    global template, target_hist
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bb)
    box = bb
    center = set_center(bb)
    state = "TRACKING"
    tracked = 0
    lost = 0
    in_crossing = False
    prev_gray = None
    offscreen_frames = 0
    recovered_from_offscreen = False
    ref_crops_b64.clear()
    template = embed(crop_of(frame, *bb))
    target_hist = jersey_hist(frame, *bb)

def pick_player_at_pos(frame, px, py):
    dets = detect(frame)
    if not dets:
        return None
    best = None
    best_dist = float("inf")
    for d in dets:
        cx, cy = d[0] + d[2] / 2, d[1] + d[3] / 2
        dist = (cx - px) ** 2 + (cy - py) ** 2
        if dist < best_dist:
            best_dist = dist
            best = d
    return best

# ── Video setup ───────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"ERROR: Cannot open video: {VIDEO_PATH}")
    sys.exit(1)

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30
cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_SEC * fps))

WIN = "Player Highlight Tracker"
if not HEADLESS:
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, 1280, 720)
    cv2.createTrackbar("Seek", WIN, 0, max(1, total - 1),
                       lambda v: cap.set(cv2.CAP_PROP_POS_FRAMES, v))

# ── State ─────────────────────────────────────────────────────────────────────
tracker=None; template=None; state="IDLE"; paused=not HEADLESS
frame_no=0; lost=0; tracked=0; box=None; last=None
center=None; target_hist=None
in_crossing = False; prev_gray = None
ref_crops_b64 = []
offscreen_frames = 0; recovered_from_offscreen = False
last_gpt_verify_frame = 0
last_ref_frame = 0
last_progress_frame = 0
GPT_VERIFY_INTERVAL = int(30 * fps)
AUTO_REF_INTERVAL = int(15 * fps)
PROGRESS_INTERVAL = max(1, int(args.progress_every * fps))

# ── Highlight recording state ──
output_dir = _DIR / "output"
output_dir.mkdir(exist_ok=True)
pre_roll_frames = int(PRE_ROLL_SEC * fps)
post_roll_frames = int(POST_ROLL_SEC * fps)
pre_roll_buf = deque(maxlen=pre_roll_frames)
recording = False; post_roll_countdown = 0
clip_writer = None; clip_count = 0; clip_paths = []
ball_involved = False
end_frame = int((START_SEC + RUN_MINUTES * 60) * fps)

def set_center(b):
    return np.array([b[0]+b[2]/2.0, b[1]+b[3]/2.0])

if HEADLESS:
    ret, seed_frame = cap.read()
    if not ret:
        print(f"ERROR: Cannot read frame at {START_SEC}s")
        sys.exit(1)
    if args.bbox:
        init_bb = parse_bbox(args.bbox)
    else:
        px, py = parse_xy(args.pos, "--pos")
        init_bb = pick_player_at_pos(seed_frame, px, py)
        if init_bb is None:
            print(f"ERROR: No players detected at --start frame near ({px}, {py})")
            sys.exit(1)
        print(f"  Auto-selected player box: {init_bb}")
    init_tracking(seed_frame, init_bb)
    save_ref_crop(seed_frame, init_bb)
    last = seed_frame
    frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    print(f"\nHeadless mode. Processing {RUN_MINUTES} min starting at {START_SEC}s.")
    print(f"  Player box: {init_bb}  jersey={PLAYER_JERSEY or '-'}  team={PLAYER_TEAM or '-'}")
else:
    print(f"\nReady. Processing {RUN_MINUTES} min starting at {START_SEC}s.")
    print("S=select player, K=save ref crop, SPACE=play, Q=quit.")

# ── Main loop ─────────────────────────────────────────────────────────────────
while True:
    if not paused:
        ret, frame = cap.read()
        if not ret: break
        frame_no = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        if not HEADLESS:
            cv2.setTrackbarPos("Seek", WIN, min(frame_no, total - 1))
    else:
        frame = last.copy() if last is not None else cap.read()[1]
        if frame is None: break

    last = frame; H,W = frame.shape[:2]; disp = frame.copy()

    # ── Camera motion compensation (only when not actively tracking) ──
    if not paused and state in ("SEARCHING", "OFFSCREEN"):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_boxes = [(box[0],box[1],box[2],box[3])] if box else []
        M = estimate_camera_motion(prev_gray, gray, mask_boxes)
        prev_gray = gray
        if center is not None:
            center = apply_motion(M, center)
    elif not paused:
        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ── TRACKING ──
    if state=="TRACKING" and not paused:
        ok, nb = tracker.update(frame)
        if ok:
            box = tuple(int(v) for v in nb); x,y,w,h = box; tracked+=1; lost=0
            center = set_center(box)
            offscreen_frames = 0

            was_crossing = in_crossing
            last_dets = None
            if tracked % COLLISION_CHECK == 0:
                last_dets = detect(frame)
                overlaps = [d for d in last_dets if iou(box,d) > OVERLAP_IOU]
                in_crossing = len(overlaps) >= 2

            if in_crossing:
                cv2.rectangle(disp,(x,y),(x+w,y+h),(0,220,220),3)
                cv2.putText(disp,"CROSSING",(x,y-8),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,220,220),2)
            else:
                if tracked % TEMPLATE_UPDATE == 0:
                    e = embed(crop_of(frame,x,y,w,h))
                    if e is not None and template is not None:
                        template = nn.functional.normalize(
                            template*(1-TEMPLATE_ALPHA)+e*TEMPLATE_ALPHA,p=2,dim=0)

                if was_crossing and not in_crossing:
                    e = embed(crop_of(frame,x,y,w,h))
                    if e is not None and template is not None:
                        sim = csim(template, e)
                        chist = jersey_hist(frame,x,y,w,h)
                        csimv = color_sim(target_hist, chist)
                        if sim < VERIFY_THRESH and csimv < COLOR_GATE:
                            print(f"  POST-CROSSING: identity mismatch "
                                  f"(reid={sim:.2f}, color={csimv:.2f}) -> searching")
                            state = "SEARCHING"; lost = 0
                        elif sim < VERIFY_THRESH:
                            print(f"  POST-CROSSING: low reid={sim:.2f} "
                                  f"but color ok={csimv:.2f} -> searching")
                            state = "SEARCHING"; lost = 0
                        else:
                            print(f"  POST-CROSSING: identity verified "
                                  f"(reid={sim:.2f}, color={csimv:.2f})")

                cv2.rectangle(disp,(x,y),(x+w,y+h),(0,230,0),3)
                cv2.putText(disp,"TRACKING",(x,y-8),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,230,0),2)

            # ── Periodic GPT verification ──
            gpt_interval = int(5 * fps) if recovered_from_offscreen else GPT_VERIFY_INTERVAL
            if (HAS_GPT and ref_crops_b64 and not in_crossing and
                    frame_no - last_gpt_verify_frame >= gpt_interval):
                last_gpt_verify_frame = frame_no
                dets_now = last_dets if last_dets is not None else detect(frame)
                print(f"  Periodic GPT check at frame {frame_no}...")
                result = gpt_confirm(frame, box, dets_now, ref_crops_b64)
                recovered_from_offscreen = False
                if result is True:
                    pass
                elif isinstance(result, tuple):
                    new_box = result
                    tracker = cv2.TrackerCSRT_create()
                    tracker.init(frame, new_box)
                    box = new_box; center = set_center(new_box); tracked = 0
                    target_hist = jersey_hist(frame, *new_box)
                    e = embed(crop_of(frame, *new_box))
                    if e is not None:
                        template = nn.functional.normalize(
                            template * 0.5 + e * 0.5, p=2, dim=0)
                    print(f"  GPT corrected — switched to different player")

            if (HEADLESS and HAS_GPT and len(ref_crops_b64) < 3 and
                    frame_no - last_ref_frame >= AUTO_REF_INTERVAL):
                save_ref_crop(frame, box)
                last_ref_frame = frame_no

        else:
            lost+=1
            if lost>=LOST_FRAMES:
                if center is not None and (center[0] < EDGE_MARGIN or
                        center[0] > W - EDGE_MARGIN or
                        center[1] < EDGE_MARGIN or
                        center[1] > H - EDGE_MARGIN):
                    state = "OFFSCREEN"; offscreen_frames = 0
                    print(f"  Player off-screen — waiting for camera to return")
                else:
                    state = "SEARCHING"
                    print("  CSRT lost - searching")

    # ── OFFSCREEN ──
    elif state=="OFFSCREEN" and not paused:
        offscreen_frames += 1
        if offscreen_frames > int(fps) and offscreen_frames % 5 == 0:
            dets = detect(frame)
            best = None; best_score = -1
            edge_margin = 250
            for d in dets:
                dx, dy, dw, dh = d
                dcx, dcy = dx + dw/2, dy + dh/2
                near_edge = False
                if center is not None:
                    if center[0] < EDGE_MARGIN + 50: near_edge = dcx < edge_margin
                    elif center[0] > W - EDGE_MARGIN - 50: near_edge = dcx > W - edge_margin
                    elif center[1] < EDGE_MARGIN + 50: near_edge = dcy < edge_margin
                    elif center[1] > H - EDGE_MARGIN - 50: near_edge = dcy > H - edge_margin
                if offscreen_frames > int(5 * fps): near_edge = True
                if not near_edge: continue
                chist = jersey_hist(frame, *d)
                csimv = color_sim(target_hist, chist)
                if target_hist is not None and csimv < COLOR_GATE: continue
                e = embed(crop_of(frame, *d))
                if e is None: continue
                s = csim(template, e)
                score = 0.6 * max(0, s) + 0.4 * csimv
                if score > best_score: best_score = score; best = d

            if best and best_score >= REID_THRESHOLD:
                tracker = cv2.TrackerCSRT_create(); tracker.init(frame, best)
                box = best; center = set_center(best)
                state = "TRACKING"; tracked = 0; lost = 0
                offscreen_frames = 0; recovered_from_offscreen = True
                e = embed(crop_of(frame, *best))
                if e is not None:
                    template = nn.functional.normalize(template * 0.8 + e * 0.2, p=2, dim=0)
                print(f"  Player re-entered! Re-ID score={best_score:.2f}")

        if offscreen_frames > OFFSCREEN_MAX:
            state = "IDLE"; print(f"  Off-screen too long — giving up")

        cv2.putText(disp, "OFFSCREEN - scanning", (W//2-160, H//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,120,0), 2)
        if center is not None:
            cp = tuple(np.clip(center, 0, [W-1, H-1]).astype(int))
            cv2.circle(disp, cp, 20, (255,120,0), 2)

    # ── SEARCHING ──
    elif state=="SEARCHING" and not paused:
        lost+=1; in_crossing = False
        radius = 150 + lost*20
        dets = detect(frame)
        best=None; best_score=-1; best_sim=0
        for d in dets:
            if center is not None and np.linalg.norm(set_center(d)-center) > radius:
                continue
            chist = jersey_hist(frame,*d)
            csimv = color_sim(target_hist, chist)
            if target_hist is not None and csimv < COLOR_GATE:
                cv2.rectangle(disp,(d[0],d[1]),(d[0]+d[2],d[1]+d[3]),(80,80,80),1)
                continue
            e = embed(crop_of(frame,*d))
            if e is None: continue
            s = csim(template,e)
            score = 0.6*max(0,s) + 0.4*csimv
            col = (0,200,255) if score>=REID_THRESHOLD else (0,0,180)
            cv2.rectangle(disp,(d[0],d[1]),(d[0]+d[2],d[1]+d[3]),col,1)
            cv2.putText(disp,f"{score:.2f}",(d[0],d[1]-4),
                        cv2.FONT_HERSHEY_SIMPLEX,0.4,col,1)
            if score>best_score: best_score=score; best=d; best_sim=s
        if center is not None:
            cv2.circle(disp, tuple(center.astype(int)), int(radius), (0,120,255), 1)
        if best and best_score>=REID_THRESHOLD:
            tracker=cv2.TrackerCSRT_create(); tracker.init(frame,best)
            box=best; center=set_center(best); state="TRACKING"; tracked=0; lost=0
            e=embed(crop_of(frame,*best))
            if e is not None:
                template=nn.functional.normalize(template*0.8+e*0.2,p=2,dim=0)
            print(f"  Re-identified! score={best_score:.2f} reid={best_sim:.2f}")
        if lost > int(30*fps): state="IDLE"; print("  Lost too long -> IDLE")

    if state=="TRACKING" and box and paused:
        x,y,w,h=box; cv2.rectangle(disp,(x,y),(x+w,y+h),(0,200,255),2)

    # ── Ball detection + highlight recording ──
    if state=="TRACKING" and box and not paused:
        balls = detect_ball(frame)
        ball_involved, closest_ball = ball_near_player(balls, box)
        for (bcx, bcy, bw, bh) in balls:
            ball_col = (0, 255, 255) if ball_involved else (200, 200, 200)
            cv2.circle(disp, (bcx, bcy), max(bw, bh)//2 + 4, ball_col, 2)
        if ball_involved and closest_ball:
            cv2.putText(disp, "BALL", (box[0], box[1]+box[3]+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)

        if ball_involved:
            if not recording:
                clip_count += 1
                clip_path = str(output_dir / f"clip_{clip_count:03d}.mp4")
                clip_paths.append(clip_path)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                clip_writer = cv2.VideoWriter(clip_path, fourcc, fps, (W, H))
                for buf_frame in pre_roll_buf:
                    clip_writer.write(buf_frame)
                recording = True
                print(f"  HIGHLIGHT #{clip_count}: ball touch! Recording...")
            post_roll_countdown = post_roll_frames
            clip_writer.write(frame)
        elif recording:
            clip_writer.write(frame)
            post_roll_countdown -= 1
            if post_roll_countdown <= 0:
                clip_writer.release(); clip_writer = None; recording = False
                dur = os.path.getsize(clip_paths[-1]) / 1024
                print(f"  HIGHLIGHT #{clip_count}: saved ({dur:.0f} KB)")

        if not recording:
            pre_roll_buf.append(frame.copy())
    elif not paused and not recording:
        pre_roll_buf.append(frame.copy())
    elif not paused and recording:
        clip_writer.write(frame)
        post_roll_countdown -= 1
        if post_roll_countdown <= 0:
            clip_writer.release(); clip_writer = None; recording = False
            print(f"  HIGHLIGHT #{clip_count}: saved (player lost during clip)")

    if recording:
        cv2.circle(disp, (W-30, 55), 10, (0,0,255), -1)
        cv2.putText(disp, "REC", (W-70, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

    # ── Duration limit ──
    if not paused and frame_no >= end_frame:
        print(f"\n  Reached {RUN_MINUTES}-minute limit. Stopping.")
        break

    if HEADLESS and not paused and frame_no - last_progress_frame >= PROGRESS_INTERVAL:
        elapsed_s = (frame_no / fps) - START_SEC if fps > 0 else 0
        remain_s = max(0, RUN_MINUTES * 60 - elapsed_s)
        print(f"  [{int(elapsed_s // 60)}:{int(elapsed_s % 60):02d}] "
              f"state={state} clips={clip_count} "
              f"{int(remain_s // 60)}:{int(remain_s % 60):02d} left")
        last_progress_frame = frame_no

    if HEADLESS:
        continue

    # ── HUD ──
    cv2.rectangle(disp,(0,0),(W,38),(20,20,20),-1)
    if in_crossing and state=="TRACKING":
        col=(0,220,220)
        cv2.putText(disp,"TRACKING (crossing)",(10,28),
                    cv2.FONT_HERSHEY_SIMPLEX,0.75,col,2)
    else:
        col = {"IDLE":(120,120,120), "TRACKING":(0,220,0),
               "SEARCHING":(0,120,255), "OFFSCREEN":(255,120,0)
               }.get(state,(255,255,255))
        cv2.putText(disp,state,(10,28),cv2.FONT_HERSHEY_SIMPLEX,0.75,col,2)

    refs_tag = f"refs:{len(ref_crops_b64)}/3" if ref_crops_b64 else "K=save ref"
    clips_tag = f"clips:{clip_count}" if clip_count else ""
    cv2.putText(disp,f"{refs_tag}  {clips_tag}",(W//2-90,28),
                cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,200,200),1)
    elapsed_s = (frame_no / fps) - START_SEC if fps > 0 else 0
    remain_s = max(0, RUN_MINUTES * 60 - elapsed_s)
    cv2.putText(disp,f"{int(remain_s//60)}:{int(remain_s%60):02d} left",
                (W-160,28),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),1)
    if state=="IDLE":
        cv2.putText(disp,"Press S to select a player",(W//2-180,H//2),
                    cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,100,255),2)

    cv2.imshow(WIN,disp)
    key = cv2.waitKey(1 if not paused else 20) & 0xFF
    if key==ord('q'): break
    elif key==32: paused=not paused
    elif key==ord('s'):
        paused=True
        bb = cv2.selectROI(WIN, last, fromCenter=False, showCrosshair=True)
        if bb != (0, 0, 0, 0):
            init_tracking(last, bb)
            print("  Player selected. Press K to save ref crops, SPACE to play.")
        paused = False
    elif key == ord('k') or key == ord('K'):
        if state == "TRACKING" and box and HAS_GPT:
            save_ref_crop(frame if not paused else last, box)
        else:
            print("  K: need to be tracking with GPT enabled")
    elif key==ord('d'):
        nf=min(total-1,frame_no+int(5*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES,nf); frame_no=nf; prev_gray=None
    elif key==ord('a'):
        nf=max(0,frame_no-int(5*fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES,nf); frame_no=nf; prev_gray=None

# ── Cleanup + merge ──────────────────────────────────────────────────────────
if clip_writer is not None:
    clip_writer.release()
    print(f"  HIGHLIGHT #{clip_count}: saved (final clip)")

cap.release()
if not HEADLESS:
    cv2.destroyAllWindows()

if clip_paths:
    output_path = str(output_dir / "highlight_reel.mp4")
    print(f"\nMerging {len(clip_paths)} clips with fade transitions...")
    FADE_FRAMES = 20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))
    total_frames = 0
    for ci, cp in enumerate(clip_paths):
        r = cv2.VideoCapture(cp)
        frames = []
        while True:
            ok, f = r.read()
            if not ok: break
            frames.append(f)
        r.release()
        n = len(frames)
        for i, f in enumerate(frames):
            if i < FADE_FRAMES:
                alpha = i / FADE_FRAMES
                f = (f.astype(np.float32) * alpha).astype(np.uint8)
            elif i >= n - FADE_FRAMES:
                alpha = (n - 1 - i) / FADE_FRAMES
                f = (f.astype(np.float32) * max(0, alpha)).astype(np.uint8)
            out.write(f)
            total_frames += 1
        if ci < len(clip_paths) - 1:
            black = np.zeros((H, W, 3), np.uint8)
            for _ in range(15):
                out.write(black)
                total_frames += 1
    out.release()
    dur_s = total_frames / fps
    print(f"Done! {len(clip_paths)} clips -> {output_path} ({dur_s:.1f}s)")
else:
    print("\nNo highlights recorded.")
