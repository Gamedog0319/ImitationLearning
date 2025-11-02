import os
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, pickle, numpy as np, cv2, keyboard
from threading import Lock, Thread
from mss import mss  # screen capture

# === Mission XML ===
# Native controls + inventory + placed stone block 2 blocks ahead.
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About>
        <Summary>Hybrid Recording (Full Control + Preloaded Inventory + Stone Block Ahead)</Summary>
    </About>

    <ServerSection>
        <ServerHandlers>
            <FlatWorldGenerator generatorString="3;7,2;1;"/>
            <DrawingDecorator>
                <!-- ✅ Place a single stone block 2 blocks ahead (x+2 from agent) -->
                <DrawBlock x="2" y="3" z="0" type="stone"/>
            </DrawingDecorator>
            <ServerQuitFromTimeUp timeLimitMs="60000"/>
            <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
    </ServerSection>

    <AgentSection mode="Survival">
        <Name>Builder</Name>

        <!-- ✅ Preload hotbar inventory -->
        <AgentStart>
            <Placement x="0.5" y="3" z="0.5" yaw="0"/>
            <Inventory>
                <InventoryItem slot="0" type="dirt" quantity="20"/>
                <InventoryItem slot="1" type="stone" quantity="20"/>
                <InventoryItem slot="2" type="wooden_door" quantity="1"/>
            </Inventory>
        </AgentStart>

        <!-- ✅ Observation only — full player control -->
        <AgentHandlers>
            <ObservationFromFullStats/>
        </AgentHandlers>
    </AgentSection>
</Mission>'''

# === Setup Agent ===
agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse([])
except RuntimeError as e:
    print("Error:", e)
    exit(1)

mission = MalmoPython.MissionSpec(mission_xml, True)
record_spec = MalmoPython.MissionRecordSpec()

client_pool = MalmoPython.ClientPool()
client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))

# === Shared dataset ===
dataset = []
lock = Lock()
current_action = {"move": 0, "turn": 0, "jump": 0}

# === Screen capture setup ===
sct = mss()
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

# === Background recording thread ===
def record_loop():
    world_state = agent_host.getWorldState()
    while world_state.is_mission_running:
        with lock:
            current_action["move"] = int(keyboard.is_pressed("w")) - int(keyboard.is_pressed("s"))
            if keyboard.is_pressed("a"):
                current_action["turn"] = -0.3
            elif keyboard.is_pressed("d"):
                current_action["turn"] = 0.3
            else:
                current_action["turn"] = 0
            current_action["jump"] = int(keyboard.is_pressed("space"))

        img = np.array(sct.grab(monitor))[:, :, :3]
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (84, 84), interpolation=cv2.INTER_AREA)

        dataset.append({
            "frame_full": frame_rgb,
            "frame_small": small_frame,
            "action": current_action.copy()
        })

        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    print("🟢 Recording thread finished.")

# === Start Mission ===
print("Starting mission...")
agent_host.startMission(mission, client_pool, record_spec, 0, "Building")

print("Waiting for mission to start", end="")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.2)
    world_state = agent_host.getWorldState()

print("\n✅ Mission started! Full control — build and record freely.")
print("🪨 There’s a stone block 2 blocks in front of you, on the ground.")
print("🎁 Check your hotbar for dirt, stone, and a wooden door.")

# === Start recording ===
recorder = Thread(target=record_loop, daemon=True)
recorder.start()

while world_state.is_mission_running:
    time.sleep(0.5)
    world_state = agent_host.getWorldState()

recorder.join()

# === Save dataset ===
os.makedirs("datasets", exist_ok=True)
save_path = "datasets/building_dataset_hybrid.pkl"
with open(save_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"🏁 Mission finished — {len(dataset)} frames recorded.")
print(f"✅ Saved as {save_path}")
