import os
# 🪟 Ensure Minecraft opens at a good window size
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, pickle, numpy as np, cv2, keyboard
from threading import Lock, Thread
from mss import mss  # 🆕 screen capture

# === Mission XML ===
# No <HumanLevelCommands/> so WASD/mouse control is native.
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <About>
        <Summary>Combat Recording (2 Zombies + Diamond Sword + Native Control)</Summary>
    </About>

    <ServerSection>
        <ServerHandlers>
            <!-- Flat world (grass on dirt) -->
            <FlatWorldGenerator generatorString="3;7,2;1;"/>

            <!-- Spawn 2 zombies near the player -->
            <DrawingDecorator>
                <DrawEntity x="2"  y="3" z="1" type="Zombie"/>
                <DrawEntity x="-2" y="3" z="1" type="Zombie"/>
            </DrawingDecorator>

            <!-- Quit mission after 60 seconds -->
            <ServerQuitFromTimeUp timeLimitMs="60000"/>
            <ServerQuitWhenAnyAgentFinishes/>
        </ServerHandlers>
    </ServerSection>

    <!-- ✅ Player-controlled Survival mode -->
    <AgentSection mode="Survival">
        <Name>Warrior</Name>

        <AgentStart>
            <!-- Start slightly above ground to avoid clipping -->
            <Placement x="0.5" y="3" z="0.5" yaw="0"/>
            <Inventory>
                <!-- Give player a diamond sword -->
                <InventoryItem slot="0" type="diamond_sword" quantity="1"/>
            </Inventory>
        </AgentStart>

        <AgentHandlers>
            <ObservationFromFullStats/>
            <!-- Native WASD/mouse control enabled automatically -->
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
# Adjust these coordinates to fit your Minecraft window position
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

# === Background recording thread ===
def record_loop():
    global dataset
    world_state = agent_host.getWorldState()

    while world_state.is_mission_running:
        # --- Read keyboard input ---
        with lock:
            current_action["move"] = int(keyboard.is_pressed("w")) - int(keyboard.is_pressed("s"))
            if keyboard.is_pressed("a"):
                current_action["turn"] = -0.3
            elif keyboard.is_pressed("d"):
                current_action["turn"] = 0.3
            else:
                current_action["turn"] = 0
            current_action["jump"] = int(keyboard.is_pressed("space"))

        # --- Screen capture (HD + small) ---
        img = np.array(sct.grab(monitor))[:, :, :3]  # BGR image from screen
        frame_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        small_frame = cv2.resize(frame_rgb, (84, 84), interpolation=cv2.INTER_AREA)

        with lock:
            dataset.append({
                "frame_full": frame_rgb,       # HD frame (with inventory, GUI)
                "frame_small": small_frame,    # 84×84 version for ML
                "action": current_action.copy()
            })

        time.sleep(0.1)  # ~10 FPS
        world_state = agent_host.getWorldState()

    print("🟢 Recording thread finished.")

# === Start Mission ===
print("Starting mission...")
agent_host.startMission(mission, client_pool, record_spec, 0, "Combat")

print("Waiting for mission to start", end="")
world_state = agent_host.getWorldState()
while not world_state.has_mission_begun:
    print(".", end="", flush=True)
    time.sleep(0.2)
    world_state = agent_host.getWorldState()
print("\n✅ Mission started! You can now play normally — WASD, mouse, attack, and jump work.")
print("🧟 Two zombies are nearby — use your diamond sword (slot 1) to fight!")

# === Start recording in the background ===
recorder = Thread(target=record_loop, daemon=True)
recorder.start()

# === Keep main thread alive while mission runs ===
while world_state.is_mission_running:
    time.sleep(0.5)
    world_state = agent_host.getWorldState()

recorder.join()

# === Save dataset ===
os.makedirs("datasets", exist_ok=True)
save_path = "datasets/combat_dataset_hybrid.pkl"
with open(save_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"🏁 Mission finished — {len(dataset)} frames recorded.")
print(f"✅ Saved as {save_path}")
