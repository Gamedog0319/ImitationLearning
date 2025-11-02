import os
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, pickle, numpy as np, cv2, keyboard, json, math
from threading import Lock, Thread
from mss import mss

# === Optional OS-level mouse control for human mode ===
# pip install pyautogui ; grant Accessibility perms on macOS
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    HAS_PYAUTO = True
except Exception:
    HAS_PYAUTO = False

# ================= User-tunable config =================
USE_MALMO_COMMANDS = True                  # Works in agent mode; Malmo may ignore in pure human play
USE_OS_MOUSE_EMULATION = True and HAS_PYAUTO  # Works in human mode (needs Minecraft window focus)

# Keyboard → target rates (before smoothing), in Malmo's [-1..1] space
TURN_RATE_A_D = 0.6     # A/D contributes this much yaw rate
TURN_RATE_JK  = 1.0     # j/k contributes this much yaw rate
PITCH_RATE_UP = 1.0     # n   (flip signs if vertical feels inverted)
PITCH_RATE_DN = -1.0    # m

TURN_MAX, PITCH_MAX = 1.0, 1.0

# Smoothing (one-pole EMA) for camera rates
SMOOTH_TC = 0.12        # seconds (0.08–0.20 is a good range)

# Human-mode mouse emulation scaling (pixels/sec at full rate=1.0)
MOUSE_YAW_PPS   = 900.0
MOUSE_PITCH_PPS = 700.0

# Loop pacing
TICK_SEC_TARGET = 0.010  # ~50 FPS (use 0.016 for ~60 FPS)

# === Mission XML ===
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>Hybrid Recording (HD Screen + Diamond Tower)</Summary>
  </About>

  <ServerSection>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;"/>
      <DrawingDecorator>
        <!-- Diamond tower base at (50,2,5) -->
        <DrawCuboid x1="50" y1="2" z1="5" x2="50" y2="102" z2="5" type="diamond_block"/>
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="60000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Builder</Name>
    <AgentStart>
      <Placement x="0" y="2" z="5" yaw="90"/>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <!-- Smooth movement with yaw/pitch; correct attribute name -->
      <ContinuousMovementCommands turnSpeedDegs="180"/>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

# === Setup Malmo ===
agent_host = MalmoPython.AgentHost()
agent_host.parse([])
mission = MalmoPython.MissionSpec(mission_xml, True)
record_spec = MalmoPython.MissionRecordSpec()
client_pool = MalmoPython.ClientPool()
client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

# === Dataset / State ===
dataset = []
lock = Lock()

current_action = {
    "move": 0,           # -1 back, 0 none, +1 fwd
    "turn": 0.0,         # smoothed yaw rate [-1..1]
    "jump": 0,           # 0/1
    "look_updown": 0.0,  # smoothed pitch rate [-1..1]
    "look_left": 0, "look_right": 0, "look_up": 0, "look_down": 0
}
KEYS = ["w","a","s","d","space","j","k","n","m"]

# === Screen capture ===
sct = mss()
# Adjust to your Minecraft window position
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

# === Goal geometry ===
GOAL_X, GOAL_Y, GOAL_Z = 50.0, 2.0, 5.0
GOAL_RADIUS = 1.0      # ≤1 block (XZ)
GROUND_TOL = 0.6       # treat Y≈2.0 as ground

def get_latest_position(world_state):
    if world_state.number_of_observations_since_last_state == 0:
        return None, None, None, None, None
    try:
        ob = json.loads(world_state.observations[-1].text)
        return ob.get("XPos"), ob.get("YPos"), ob.get("ZPos"), ob.get("Yaw"), ob.get("Pitch")
    except Exception:
        return None, None, None, None, None

def is_on_ground(y):
    return (y is not None) and (abs(y - GOAL_Y) <= GROUND_TOL)

def goal_reached(x, y, z):
    if x is None or y is None or z is None: return False
    if not is_on_ground(y): return False
    return math.hypot(x - GOAL_X, z - GOAL_Z) <= GOAL_RADIUS

def is_idle(act, keys_pressed):
    if any(keys_pressed[k] for k in KEYS): return False
    return (act["move"]==0 and act["turn"]==0.0 and act["jump"]==0 and
            act["look_updown"]==0.0 and act["look_left"]==0 and act["look_right"]==0 and
            act["look_up"]==0 and act["look_down"]==0)

def get_high_level_label(act, reached, keys_pressed):
    if reached: return "goal_reached"
    if is_idle(act, keys_pressed): return "idle"
    if act["turn"] < 0 or act["look_left"]==1 or keys_pressed["j"]: return "turn_left"
    if act["turn"] > 0 or act["look_right"]==1 or keys_pressed["k"]: return "turn_right"
    if act["move"] != 0 or keys_pressed["w"] or keys_pressed["s"]: return "go_around"
    return "go_around"

def get_mode_state(reached: bool, idle_flag: bool) -> str:
    """
    Collapse to exactly one of: 'idle', 'searching', 'goal_reached'.
    Priority: goal_reached > idle > searching.
    """
    if reached: return "goal_reached"
    if idle_flag: return "idle"
    return "searching"

# --- Smoothing state ---
turn_rate_s   = 0.0  # smoothed yaw rate
pitch_rate_s  = 0.0  # smoothed pitch rate
mouse_frac_x  = 0.0  # fractional pixel carryover (x)
mouse_frac_y  = 0.0  # fractional pixel carryover (y)

def record_loop():
    global dataset, turn_rate_s, pitch_rate_s, mouse_frac_x, mouse_frac_y
    world_state = agent_host.getWorldState()
    last_t = time.perf_counter()

    while world_state.is_mission_running:
        now = time.perf_counter()
        dt = max(1e-3, now - last_t)                # clamp dt to avoid spikes
        last_t = now
        alpha = dt / (SMOOTH_TC + dt)               # EMA coefficient

        # ---- Raw keys ----
        keys_pressed = {k: int(keyboard.is_pressed(k if k != "space" else "space")) for k in KEYS}

        # ---- Targets from keys (unsmoothed) ----
        move_val = keys_pressed["w"] - keys_pressed["s"]
        turn_target = (
            (-TURN_RATE_A_D if keys_pressed["a"] else 0.0) + (TURN_RATE_A_D if keys_pressed["d"] else 0.0) +
            (-TURN_RATE_JK  if keys_pressed["j"] else 0.0) + (TURN_RATE_JK  if keys_pressed["k"] else 0.0)
        )
        pitch_target = (PITCH_RATE_UP if keys_pressed["n"] else 0.0) + (PITCH_RATE_DN if keys_pressed["m"] else 0.0)
        turn_target  = max(-TURN_MAX,  min(TURN_MAX,  turn_target))
        pitch_target = max(-PITCH_MAX, min(PITCH_MAX, pitch_target))
        jump_val = keys_pressed["space"]

        # ---- Smooth the camera rates ----
        turn_rate_s  = (1.0 - alpha) * turn_rate_s  + alpha * turn_target
        pitch_rate_s = (1.0 - alpha) * pitch_rate_s + alpha * pitch_target

        with lock:
            current_action["move"] = int(move_val)
            current_action["turn"] = float(turn_rate_s)
            current_action["jump"] = int(jump_val)
            current_action["look_updown"] = float(pitch_rate_s)
            current_action["look_left"]  = keys_pressed["j"]
            current_action["look_right"] = keys_pressed["k"]
            current_action["look_up"]    = keys_pressed["n"]
            current_action["look_down"]  = keys_pressed["m"]

        # ---- Drive the game ----
        # A) Malmo (agent mode)
        if USE_MALMO_COMMANDS:
            try:
                agent_host.sendCommand(f"move {current_action['move']}")
                agent_host.sendCommand(f"turn {current_action['turn']}")
                agent_host.sendCommand(f"pitch {current_action['look_updown']}")
                agent_host.sendCommand(f"jump {current_action['jump']}")
            except RuntimeError:
                pass

        # B) OS mouse emulation (human mode) using smoothed rates
        mouse_dx = mouse_dy = 0.0
        if USE_OS_MOUSE_EMULATION:
            dx = turn_rate_s   * MOUSE_YAW_PPS    * dt
            dy = -pitch_rate_s * MOUSE_PITCH_PPS  * dt  # negative: typical game pitch
            dx_total = dx + mouse_frac_x
            dy_total = dy + mouse_frac_y
            step_x = int(round(dx_total))
            step_y = int(round(dy_total))
            mouse_frac_x = dx_total - step_x
            mouse_frac_y = dy_total - step_y
            if step_x != 0 or step_y != 0:
                try:
                    pyautogui.moveRel(step_x, step_y, duration=0)
                except Exception:
                    step_x = step_y = 0
            mouse_dx, mouse_dy = float(step_x), float(step_y)

        # ---- Position & view ----
        x, y, z, yaw_obs, pitch_obs = get_latest_position(world_state)
        reached = goal_reached(x, y, z)

        # ---- Screen grab ----
        img = np.array(sct.grab(monitor))[:, :, :3]
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (84, 84), interpolation=cv2.INTER_AREA)

        # ---- Label & state flags/mode ----
        idle_flag = is_idle(current_action, keys_pressed)
        label = get_high_level_label(current_action, reached, keys_pressed)
        mode3 = get_mode_state(reached, idle_flag)

        # ---- Log one frame ----
        dataset.append({
            "frame_full": frame_rgb,
            "frame_small": small_frame,
            "action": current_action.copy(),        # smoothed/continuous actions
            "keys": keys_pressed.copy(),            # raw key presses (0/1)
            "camera": {                             # camera rates
                "turn_rate_target": turn_target,
                "pitch_rate_target": pitch_target,
                "turn_rate": turn_rate_s,           # smoothed/commanded
                "pitch_rate": pitch_rate_s
            },
            "mouse_applied": {"dx": mouse_dx, "dy": mouse_dy},  # OS mouse deltas
            "view_obs": {"yaw": yaw_obs, "pitch": pitch_obs},
            "state": { "goal_reached": bool(reached), "idle": bool(idle_flag) },
            "mode": mode3,                          # exactly 'idle' | 'searching' | 'goal_reached'
            "label": label,                          # fine-grained category
            "pos": (x, y, z),
            "dt": dt,
            "timestamp": time.time()
        })

        # pace loop
        to_sleep = max(0.0, TICK_SEC_TARGET - (time.perf_counter() - now))
        if to_sleep > 0: time.sleep(to_sleep)
        world_state = agent_host.getWorldState()

    print("🟢 Recording thread finished.")

# ==== Mission run ====
print("Starting mission...")
agent_host.startMission(mission, client_pool, record_spec, 0, "Building")

print("Waiting for mission to start", end="")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.2)
    world_state = agent_host.getWorldState()
print("\n✅ Mission started! Smooth look: j/k (yaw), n/m (pitch).")

recorder = Thread(target=record_loop, daemon=True)
recorder.start()

while world_state.is_mission_running:
    time.sleep(0.5)
    world_state = agent_host.getWorldState()

recorder.join()

os.makedirs("datasets", exist_ok=True)
save_path = "datasets/navigation_dataset_hybrid.pkl"
with open(save_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"🏁 Mission finished — {len(dataset)} frames recorded.")
print(f"✅ Saved as {save_path}")
