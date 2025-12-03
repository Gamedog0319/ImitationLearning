import os
import cv2
os.environ["MALMO_MINECRAFT_RESOLUTION"] = "1280x720"

import malmo.MalmoPython as MalmoPython
import time, numpy as np
from mss import mss
from bc import load_policy_and_predict

# === Mission XML ===
mission_xml = '''<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<Mission xmlns="http://ProjectMalmo.microsoft.com"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <About>
    <Summary>Activley Sampling BC Policy</Summary>
  </About>
  <ServerSection>
    <ServerInitialConditions>
        <Time>
            <StartTime>10000</StartTime>
        </Time>
        <Weather>clear</Weather>
    </ServerInitialConditions>
    <ServerHandlers>
      <FlatWorldGenerator generatorString="3;7,2;1;" forceReset="true"/>
      <DrawingDecorator>
        <DrawCuboid x1="50" y1="2" z1="0" x2="50" y2="102" z2="10" type="gold_block"/>
      </DrawingDecorator>
      <ServerQuitFromTimeUp timeLimitMs="60000"/>
      <ServerQuitWhenAnyAgentFinishes/>
    </ServerHandlers>
  </ServerSection>
  
  

  <AgentSection mode="Survival">
    <Name>Navigator</Name>
    <AgentStart>
      <Placement x="30" y="2" z="5" yaw="90"/>
    </AgentStart>
    <AgentHandlers>
      <ObservationFromFullStats/>
      <ContinuousMovementCommands turnSpeedDegs="180"/>

      <!-- IMPORTANT: This is the player (agent) camera feed -->
      <VideoProducer want_depth="false">
        <Width>1280</Width>
        <Height>720</Height>
      </VideoProducer>

      <AgentQuitFromTouchingBlockType>
        <Block type="gold_block"/>
      </AgentQuitFromTouchingBlockType>
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

# === Screen capture ===
sct = mss()
monitor = {"top": 100, "left": 100, "width": 1280, "height": 720}

# def grab_rgb(monitor):
#     with mss() as sct:
#         frame = np.array(sct.grab(monitor))       # BGRA
#         rgb = frame[:, :, :3][:, :, ::-1].copy()  # convert BGRA → RGB
#     return rgb
def grab_rgb():
    # if len(world_state.video_frames == 0):
    #     # No frame this tick, skip logging this iteration
    #         # (you can also choose to reuse the previous frame)
    #     time.sleep(0.01)
    #     continue
    vf = world_state.video_frames[-1]  # latest frame
    img = np.frombuffer(vf.pixels, dtype=np.uint8)
    img = img.reshape((vf.height, vf.width, 3))   # (H, W, 3), RGB

    frame_rgb = img
    # Still resize to 84x84 in case you change VideoProducer size later
    small_frame = cv2.resize(frame_rgb, (1280, 720), interpolation=cv2.INTER_AREA)
    
    return small_frame

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
    frame_rgb = grab_rgb()
    actions = {
        'LEFT': -0.2,
        'RIGHT': 0.2,
        'GO': 1.0
        }
    next_action = load_policy_and_predict(frame_rgb)
    move_agent(agent_host, next_action, actions[next_action])

print("🏁 Mission finished")
