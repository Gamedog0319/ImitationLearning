import os, sys
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, pickle, numpy as np, cv2, keyboard, json, math, random
from threading import Lock, Thread
from mss import mss
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "BC_demo"))
from bc import load_policy_and_predict

# === Optional OS-level mouse control for human mode ===
try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    HAS_PYAUTO = True
except Exception:
    HAS_PYAUTO = False

# ================= User-tunable config =================
USE_MALMO_COMMANDS = True                      # agent-mode commands
USE_OS_MOUSE_EMULATION = True and HAS_PYAUTO   # human-mode mouse nudge

# Keyboard → target rates (before smoothing), in Malmo's [-1..1] space
TURN_RATE_A_D = 0.6     # A/D yaw rate
TURN_RATE_JK  = 1.0     # j/k yaw rate
PITCH_RATE_UP = 1.0     # n
PITCH_RATE_DN = -1.0    # m
TURN_MAX, PITCH_MAX = 1.0, 1.0

# --- Smoothing & control pacing (SMOOTHER CAMERA) ---
SMOOTH_TC = 0.20          # ↑ more smoothing than before (0.12 → 0.20)
CONTROL_SUBSTEPS = 5       # do 5 micro-updates per tick for buttery motion
SLEW_YAW_PER_S   = 4.0     # max change in yaw rate units/sec
SLEW_PITCH_PER_S = 4.0     # max change in pitch rate units/sec

# Human-mode mouse emulation scaling (pixels/sec at full rate=1.0)
MOUSE_YAW_PPS   = 900.0
MOUSE_PITCH_PPS = 700.0

# Loop pacing (dataset capture cadence ~10 Hz)
TICK_SEC_TARGET = 0.10

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
        <DrawCuboid x1="50" y1="2" z1="5" x2="50" y2="102" z2="5" type="diamond_block"/>
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="60000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>

  <AgentSection mode="Survival">
    <Name>Navigator</Name>
    <AgentStart>
      <Placement x="0" y="2" z="5" yaw="90"/>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ContinuousMovementCommands turnSpeedDegs="180"/>
      <AgentQuitFromReachingPosition>
        <Marker x="50" y="2" z="5" tolerance="1.0" description="goal"/>
      </AgentQuitFromReachingPosition>
    </AgentHandlers>
  </AgentSection>
</Mission>'''

# === Setup Agent ===
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse([])
except RuntimeError as e:
    print("Error:", e)
    raise SystemExit(1)

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
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

# === Goal geometry (for labels) ===
GOAL_X, GOAL_Y, GOAL_Z = 50.0, 2.0, 5.0
GOAL_RADIUS = 1.0
GROUND_TOL = 0.6

def get_latest_position(world_state):
    if world_state.number_of_observations_since_last_state == 0:
        return None, None, None, None, None
    try:
        ob = json.loads(world_state.observations[-1].text)
        return ob.get("XPos"), ob.get("YPos"), ob.get("ZPos"), ob.get("Yaw"), ob.get("Pitch")
    except Exception:
        return None, None, None, None, None

def is_on_ground(y): return (y is not None) and (abs(y - GOAL_Y) <= GROUND_TOL)

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
    if reached: return "goal_reached"
    if idle_flag: return "idle"
    return "searching"

# --- Smoothing state ---
turn_rate_s   = 0.0  # smoothed yaw rate
pitch_rate_s  = 0.0  # smoothed pitch rate
mouse_frac_x  = 0.0  # fractional pixel carryover (x)
mouse_frac_y  = 0.0  # fractional pixel carryover (y)

def _apply_smoothing_and_slew(rate, target, sub_dt, slew_per_s):
    """EMA toward target, then clamp the per-step change to a slew limit."""
    alpha = sub_dt / (SMOOTH_TC + sub_dt)           # low-pass
    candidate = (1.0 - alpha) * rate + alpha * target
    delta = candidate - rate
    max_step = slew_per_s * sub_dt
    if   delta >  max_step: delta =  max_step
    elif delta < -max_step: delta = -max_step
    return rate + delta

def record_loop():
    global dataset, turn_rate_s, pitch_rate_s, mouse_frac_x, mouse_frac_y
    world_state = agent_host.getWorldState()
    last_t = time.perf_counter()

    while world_state.is_mission_running:
        now = time.perf_counter()
        dt_total = max(1e-3, now - last_t)  # clamp dt to avoid spikes
        last_t = now

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

        # ---- Multi-substep control update (smoother) ----
        sub_dt = dt_total / CONTROL_SUBSTEPS
        mouse_dx = mouse_dy = 0.0

        for _ in range(CONTROL_SUBSTEPS):
            # update smoothed rates with EMA + slew limit
            turn_rate_s  = _apply_smoothing_and_slew(turn_rate_s,  turn_target,  sub_dt, SLEW_YAW_PER_S)
            pitch_rate_s = _apply_smoothing_and_slew(pitch_rate_s, pitch_target, sub_dt, SLEW_PITCH_PER_S)

            # send agent commands each substep for finer motion
            if USE_MALMO_COMMANDS:
                try:
                    agent_host.sendCommand(f"move {int(move_val)}")
                    agent_host.sendCommand(f"turn {turn_rate_s}")
                    agent_host.sendCommand(f"pitch {pitch_rate_s}")
                    agent_host.sendCommand(f"jump {int(jump_val)}")
                except RuntimeError:
                    pass

            # OS mouse emulation per substep
            if USE_OS_MOUSE_EMULATION:
                dx = turn_rate_s   * MOUSE_YAW_PPS   * sub_dt
                dy = -pitch_rate_s * MOUSE_PITCH_PPS * sub_dt  # negative: typical game pitch
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
                mouse_dx += float(step_x)
                mouse_dy += float(step_y)

        # ---- Publish current action snapshot ----
        with lock:
            current_action["move"] = int(move_val)
            current_action["turn"] = float(turn_rate_s)
            current_action["jump"] = int(jump_val)
            current_action["look_updown"] = float(pitch_rate_s)
            current_action["look_left"]  = keys_pressed["j"]
            current_action["look_right"] = keys_pressed["k"]
            current_action["look_up"]    = keys_pressed["n"]
            current_action["look_down"]  = keys_pressed["m"]

        # ---- Position & view ----
        x, y, z, yaw_obs, pitch_obs = get_latest_position(world_state)
        reached = goal_reached(x, y, z)

        # ---- Screen grab ----
        img = np.array(sct.grab(monitor))[:, :, :3]
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (84, 84), interpolation=cv2.INTER_AREA)

        # ---- Labels / modes ----
        idle_flag = is_idle(current_action, keys_pressed)
        label = get_high_level_label(current_action, reached, keys_pressed)
        mode3 = get_mode_state(reached, idle_flag)

        # ---- Log one frame ----
        dataset.append({
            "frame_full": frame_rgb,
            "frame_small": small_frame,
            "action": current_action.copy(),        # smoothed actions
            "keys": keys_pressed.copy(),            # raw key presses
            "camera": {
                "turn_rate_target": turn_target,
                "pitch_rate_target": pitch_target,
                "turn_rate": turn_rate_s,
                "pitch_rate": pitch_rate_s
            },
            "mouse_applied": {"dx": mouse_dx, "dy": mouse_dy},
            "view_obs": {"yaw": yaw_obs, "pitch": pitch_obs},
            "state": { "goal_reached": bool(reached), "idle": bool(idle_flag) },
            "mode": mode3,
            "label": label,
            "pos": (x, y, z),
            "dt": dt_total,
            "timestamp": time.time()
        })

        # pace loop (dataset cadence)
        to_sleep = max(0.0, TICK_SEC_TARGET - (time.perf_counter() - now))
        if to_sleep > 0:
            time.sleep(to_sleep)
        world_state = agent_host.getWorldState()

    print("🟢 Recording thread finished.")


def grab_rgb(monitor):
    with mss() as sct:
        frame = np.array(sct.grab(monitor))       # BGRA
        rgb = frame[:, :, :3][:, :, ::-1].copy()  # convert BGRA → RGB
    return rgb

def move_agent(agent_host, move_type, amount=0.0, duration=1.0):
    """
    Sends continuous movement commands to a Malmo agent for a given duration.

    """
    # Apply movement
    if move_type == 'GO':
        agent_host.sendCommand(f"move {amount}")
    if move_type == 'RIGHT' or move_type == 'LEFT':
        agent_host.sendCommand(f"turn {amount}")

    # Keep command active
    time.sleep(duration)

    # Stop movement
    if move_type == 'GO':
        agent_host.sendCommand("move 0.0")
    if move_type == 'RIGHT' or move_type == 'LEFT':
        agent_host.sendCommand("turn 0.0")

# ==== Mission run ====
print("Starting mission...")
agent_host.startMission(mission, client_pool, record_spec, 0, "Navigation")

# save_dir = "test_shots"
# os.makedirs(save_dir, exist_ok=True)
# count = 1

print("Waiting for mission to start", end="")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.2)
    world_state = agent_host.getWorldState()
print("\n✅ Mission started! Smooth look: j/k (yaw), n/m (pitch).")

# recorder = Thread(target=record_loop, daemon=True)
# recorder.start()

while world_state.is_mission_running:
    
    time.sleep(0.5)
    world_state = agent_host.getWorldState()

    # # Capture screen
    # img = np.array(sct.grab(monitor))
    # img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)  # Convert from BGRA to BGR

    # # Build file path (like test_shots/1.png, test_shots/2.png, etc.)
    # filename = os.path.join(save_dir, f"{count}.png")

    # # Save image
    # cv2.imwrite(filename, img)
    # print(f"Saved: {filename}") 

    frame_rgb = grab_rgb(monitor)

    actions = {
        'LEFT': -0.2,
        'RIGHT': 0.2,
        'GO': 1.0}
    next_action = load_policy_and_predict(frame_rgb)

    move_agent(agent_host, next_action, actions[next_action])

# recorder.join()

# === Save (same path/style) ===
# os.makedirs("datasets", exist_ok=True)
# save_path = "datasets/navigation_dataset_hybrid.pkl"
# with open(save_path, "wb") as f:
#     pickle.dump(dataset, f)

print("🏁 Mission finished")
# print(f"✅ Saved as {save_path}")
