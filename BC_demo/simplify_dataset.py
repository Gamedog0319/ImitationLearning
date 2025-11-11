# Minimal: Turn the .pkl to images + CSV with {LEFT, RIGHT, GO}
import os, pickle
import cv2
import numpy as np

PKL_PATH      = "../ProjectFiles/datasets/navigation_dataset_hybrid.pkl"
OUT_DIR       = "dataset_malmo"
IMG_SIZE      = 84
TURN_ZERO     = 0.05   # turn val below this is treated as 0 
TURN_THRESH   = 0.15   # >= this RIGHT, <= this LEFT
GO_STRIDE     = 3      # keep every Nth GO in a GO run (set 1 to keep all)

# Map raw action to LEFT RIGHT GO None (Priority: TURN beat MOVE)
def decide_label(action):
    move = int(action.get("move", 0))
    turn = float(action.get("turn", 0.0))
    if abs(turn) < TURN_ZERO:
        turn = 0.0
    if turn <= -TURN_THRESH:
        return "LEFT"
    if turn >= +TURN_THRESH:
        return "RIGHT"
    if move == 1:
        return "GO"
    return None

def main():
    # Load
    with open(PKL_PATH, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {len(data)} frames from {PKL_PATH}")

    # Prepare output
    frames_dir = os.path.join(OUT_DIR, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "actions.csv")

    rows = [("filename", "action")]
    fname_counter = 1
    go_run = 0
    seen_goal = False

    for entry in data:
        # stop at first goal to avoid too much idle
        state = entry.get("state") or {}
        if state.get("goal_reached", False):
            seen_goal = True
        if seen_goal:
            break

        # choose frame (prefer small)
        frame = entry.get("frame_small")
        if frame is None:
            full = entry.get("frame_full")
            if full is None:
                continue
            frame = cv2.resize(full, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        else:
            if frame.shape[:2] != (IMG_SIZE, IMG_SIZE):
                frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

        # label
        action = entry.get("action") or {}
        lab = decide_label(action)
        if lab is None:
            continue

        # downsample long GO runs, 
        if lab == "GO":
            go_run += 1
            if GO_STRIDE > 1 and (go_run % GO_STRIDE) != 1:
                continue
        else:
            go_run = 0

        # save PNG (convert RGB->BGR for cv2)
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        fname = f"{fname_counter:06d}.png"
        cv2.imwrite(os.path.join(frames_dir, fname), bgr)
        rows.append((fname, lab))
        fname_counter += 1

    # write CSV
    with open(csv_path, "w", newline="") as f:
        f.write("filename,action\n")
        for fn, lab in rows[1:]:
            f.write(f"{fn},{lab}\n")

    print(f"[DONE] Kept {len(rows)-1} frames → {csv_path}")
    print(f"       Images in {frames_dir}")

if __name__ == "__main__":
    main()