import os
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, pickle, numpy as np, cv2, keyboard
from threading import Lock, Thread
from mss import mss

# === Mission XML ===
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About>
        <Summary>Hybrid Recording (HD Screen + Malmo metadata)</Summary>
    </About>

    <ServerSection>
        <!-- Initial conditions -->
        <ServerInitialConditions>
            <Time>
                <StartTime>6000</StartTime>
                <AllowPassageOfTime>false</AllowPassageOfTime>
            </Time>
        </ServerInitialConditions>

        <ServerHandlers>
            <!-- Flat world base -->
            <FlatWorldGenerator generatorString="3;7,2;1;"/>

            <!-- Build 30x30 farmland -->
            <DrawingDecorator>
                <!-- Ground base -->
                <DrawCuboid x1="-20" y1="4" z1="-20"
                            x2="20" y2="4" z2="20" type="grass"/>

                <!-- Farmland soil -->
                <DrawCuboid x1="-15" y1="5" z1="-15"
                            x2="15" y2="5" z2="15" type="farmland"/>

                <!-- Water channels for hydration -->
                <DrawCuboid x1="-15" y1="5" z1="-10"
                            x2="15" y2="5" z2="-10" type="water"/>
                <DrawCuboid x1="-15" y1="5" z1="-4"
                            x2="15" y2="5" z2="-4" type="water"/>
                <DrawCuboid x1="-15" y1="5" z1="2"
                            x2="15" y2="5" z2="2" type="water"/>
                <DrawCuboid x1="-15" y1="5" z1="8"
                            x2="15" y2="5" z2="8" type="water"/>

                <!-- Fully grown wheat (approximation) -->
                <DrawCuboid x1="-15" y1="6" z1="-15"
                            x2="15" y2="6" z2="15" type="wheat"/>
            </DrawingDecorator>

            <ServerQuitFromTimeUp timeLimitMs="60000"/>
            <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
    </ServerSection>

    <AgentSection mode="Survival">
        <Name>Builder</Name>
        <AgentStart>
            <Placement x="0.5" y="6" z="18.5" yaw="180"/>
        </AgentStart>

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

dataset = []
lock = Lock()
current_action = {"move": 0, "turn": 0, "jump": 0}

sct = mss()
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

def record_loop():
    global dataset
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

        with lock:
            dataset.append({
                "frame_full": frame_rgb,
                "frame_small": small_frame,
                "action": current_action.copy()
            })

        time.sleep(0.1)
        world_state = agent_host.getWorldState()

    print("🟢 Recording thread finished.")

print("Starting mission...")
agent_host.startMission(mission, client_pool, record_spec, 0, "Farming")

print("Waiting for mission to start", end="")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.2)
    world_state = agent_host.getWorldState()
print("\n✅ Mission started! You can now play normally — farm with WASD + mouse.")

recorder = Thread(target=record_loop, daemon=True)
recorder.start()

while world_state.is_mission_running:
    time.sleep(0.5)
    world_state = agent_host.getWorldState()

recorder.join()

os.makedirs("datasets", exist_ok=True)
save_path = "datasets/farming_dataset_hybrid.pkl"
with open(save_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"🏁 Mission finished — {len(dataset)} frames recorded.")
print(f"✅ Saved as {save_path}")
