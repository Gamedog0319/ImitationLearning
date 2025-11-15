import pickle
import cv2
import numpy as np
import math

# === Load dataset ===
data_path = "datasets/navigation_dataset_hybrid.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} frames from {data_path}")

# === Settings ===
delay = 70            # ms between frames (lower = faster playback)
scale_factor = 1.0    # < 1.0 to shrink window if it's too big

# === Goal (diamond tower column at x=50,z=5) ===
GOAL_X, GOAL_Z = 50.0, 5.0
GOAL_RADIUS = 5.0     # <= 5 blocks horizontally counts as 'goal_reached'

KEYS = ["w", "a", "s", "d", "space", "j", "k", "n", "m"]

def horiz_goal_reached(entry):
    """Goal if horizontal (XZ) distance to (50,5) <= 5 blocks."""
    pos = entry.get("pos")
    if not pos:
        # Fallback to precomputed state if present
        st = entry.get("state") or {}
        return bool(st.get("goal_reached", False))
    x, y, z = pos
    if x is None or z is None:
        return False
    return math.hypot(x - GOAL_X, z - GOAL_Z) <= GOAL_RADIUS

def any_input(entry):
    """True if any relevant key is pressed or action indicates motion."""
    keys = entry.get("keys") or {}
    if keys:
        if any(keys.get(k, 0) for k in KEYS):
            return True

    act = entry.get("action") or {}
    # treat any nonzero control as an input/motion
    if act:
        if (act.get("move", 0) != 0 or
            abs(act.get("turn", 0.0)) > 1e-3 or
            act.get("jump", 0) == 1 or
            abs(act.get("look_updown", 0.0)) > 1e-3 or
            act.get("look_left", 0) == 1 or
            act.get("look_right", 0) == 1 or
            act.get("look_up", 0) == 1 or
            act.get("look_down", 0) == 1):
            return True

    return False

def is_idle(entry):
    """Idle = no movement and no button press."""
    keys = entry.get("keys") or {}
    any_key = any(keys.get(k, 0) for k in KEYS) if keys else False

    act = entry.get("action") or {}
    moving = False
    if act:
        moving = (
            act.get("move", 0) != 0 or
            abs(act.get("turn", 0.0)) > 1e-3 or
            act.get("jump", 0) == 1 or
            abs(act.get("look_updown", 0.0)) > 1e-3 or
            act.get("look_left", 0) == 1 or
            act.get("look_right", 0) == 1 or
            act.get("look_up", 0) == 1 or
            act.get("look_down", 0) == 1
        )

    return (not any_key) and (not moving)

def compute_state(entry):
    """Priority: goal_reached > idle > searching."""
    if horiz_goal_reached(entry):
        return "goal_reached"
    if is_idle(entry):
        return "idle"
    return "searching"

# === Playback ===
for i, entry in enumerate(data):
    frame = entry.get("frame_full")
    if frame is None:
        frame = entry.get("frame_small")

    if frame is None:
        continue  # skip if somehow no frame

    # Compute state per frame
    state = compute_state(entry)

    # Convert for display (stored as RGB, OpenCV expects BGR)
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Optional downscale for viewing
    if scale_factor != 1.0:
        height, width = frame_bgr.shape[:2]
        new_w = int(width * scale_factor)
        new_h = int(height * scale_factor)
        frame_bgr = cv2.resize(frame_bgr, (new_w, new_h))

    # Overlay state (and basic HUD)
    hud_lines = [
        f"Frame {i+1}/{len(data)}",
        f"State: {state}",
    ]
    act = entry.get("action")
    if act is not None:
        hud_lines.append(f"Action: {act}")

    y = 24
    for line in hud_lines:
        cv2.putText(
            frame_bgr,
            str(line),
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 26

    cv2.imshow("Recorded Gameplay (Hybrid)", frame_bgr)
    print(f"Frame {i+1}/{len(data)} | State: {state} | Action: {entry.get('action')}")

    key = cv2.waitKey(delay) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(" "):  # Spacebar pauses
        cv2.waitKey(-1)  # wait until any key is pressed

cv2.destroyAllWindows()
