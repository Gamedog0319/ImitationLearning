import pickle, cv2, numpy as np

# === Load dataset ===
data_path = "datasets/farming_dataset_hybrid.pkl"
with open(data_path, "rb") as f:
    data = pickle.load(f)

print(f"✅ Loaded {len(data)} frames from {data_path}")

# === Settings ===
delay = 70
scale_factor = 1.0  # adjust <1.0 to shrink window if too big

# === Playback ===
for i, entry in enumerate(data):
    frame = entry.get("frame_full")
    if frame is None:
        frame = entry.get("frame_small")

    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if scale_factor != 1.0:
        w, h = int(frame_bgr.shape[1] * scale_factor), int(frame_bgr.shape[0] * scale_factor)
        frame_bgr = cv2.resize(frame_bgr, (w, h))

    cv2.imshow("Recorded Gameplay (Hybrid)", frame_bgr)
    print(f"Frame {i+1}/{len(data)} | Action: {entry['action']}")

    key = cv2.waitKey(delay) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(" "):  # Spacebar pauses
        cv2.waitKey(-1)  # wait until any key is pressed

cv2.destroyAllWindows()
