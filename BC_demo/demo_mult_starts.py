import os
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, numpy as np, random
from mss import mss
from bc import load_policy_and_predict

# === Set-up Mission ===
possible_starts = ['./start1.xml', './start2.xml']
mission_file = random.choice(possible_starts)
with open(mission_file, 'r') as f:
    print("Loading mission from %s" % mission_file)
    mission_xml = f.read()
    mission = MalmoPython.MissionSpec(mission_xml, True)

# === Setup Agent ===
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse([])
except RuntimeError as e:
    print("Error:", e)
    raise SystemExit(1)

# mission = MalmoPython.MissionSpec(mission_xml, True)
record_spec = MalmoPython.MissionRecordSpec()

client_pool = MalmoPython.ClientPool()
client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

# === Screen capture ===
sct = mss()
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

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
print("Waiting for mission to start", end="")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.2)
    world_state = agent_host.getWorldState()
print("\n✅ Mission started!")


while world_state.is_mission_running:
    time.sleep(0.5)
    world_state = agent_host.getWorldState()
    frame_rgb = grab_rgb(monitor)
    actions = {
        'LEFT': -0.2,
        'RIGHT': 0.2,
        'GO': 1.0}
    next_action = load_policy_and_predict(frame_rgb)
    move_agent(agent_host, next_action, actions[next_action])

print("🏁 Mission finished")
