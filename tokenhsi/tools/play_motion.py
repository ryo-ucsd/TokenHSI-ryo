# tools/play_motion.py  (minimal)
import time, argparse, numpy as np
from isaacgym import gymapi

parser = argparse.ArgumentParser()
parser.add_argument("--motion-file",  required=True)
#parser.add_argument("--object-file",  required=True)
parser.add_argument("--asset-cfg",    default="tokenhsi/data/assets/mjcf/amp_humanoid.xml")
parser.add_argument("--speed", type=float, default=1.0)
args = parser.parse_args()

hum = np.load(args.motion_file, allow_pickle=True)      # [T, ...]  root + joints
print(hum)
T   = len(hum)

gym  = gymapi.acquire_gym()
sim  = gym.create_sim(0, 0, gymapi.SimParams())
env  = gym.create_env(sim, gymapi.Vec3(-1,-1,0), gymapi.Vec3(1,1,1), 1)

hum_asset = gym.load_asset(sim, "", args.asset_cfg, gymapi.AssetOptions())
box_asset = gym.create_box(sim, 1,1,1, gymapi.AssetOptions())

hum_handle = gym.create_actor(env, hum_asset, gymapi.Transform(), "hum", 0, 0)
box_handle = gym.create_actor(env, box_asset, gymapi.Transform(), "box", 0, 0)

viewer = gym.create_viewer(sim, gymapi.CameraProperties())
for t in range(T):
    # humanoid root
    hum_rs = gym.get_actor_root_state(env, hum_handle)
    hum_rs['pose']['p'][:] = hum[t, 0:3]
    hum_rs['pose']['r'][:] = [0,0,0,1]
    gym.set_actor_root_state(env, hum_handle, hum_rs)

    # box
    # box_rs = gym.get_actor_root_state(env, box_handle)
    # box_rs['pose']['p'][:] = box[t, 0:3]
    # box_rs['pose']['r'][:] = box[t, 3:7]
    #gym.set_actor_root_state(env, box_handle, box_rs)

    gym.simulate(sim);  gym.fetch_results(sim, True);  gym.step_graphics(sim);  gym.draw_viewer(viewer, sim)
    time.sleep(args.speed * gym.get_sim_time_step(sim))
