import os
import glob
import numpy as np
from PIL import Image

import habitat_sim


def find_scene():
    scene_files = sorted(glob.glob("data/**/*.glb", recursive=True))
    if len(scene_files) == 0:
        raise FileNotFoundError("No .glb scene found under data/.")

    print("Found scenes:")
    for s in scene_files:
        print("  ", s)

    for s in scene_files:
        if "apartment_1.glb" in s:
            return s

    return scene_files[0]


def make_sim(scene_path, width=640, height=480):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "color_sensor"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.resolution = [height, width]
    rgb_spec.position = [0.0, 1.5, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_spec.resolution = [height, width]
    depth_spec.position = [0.0, 1.5, 0.0]

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def save_rgb_depth(observations):
    os.makedirs("results/images", exist_ok=True)

    rgb = observations["color_sensor"]
    if rgb.shape[-1] == 4:
        rgb = rgb[:, :, :3]

    rgb = rgb.astype(np.uint8)
    Image.fromarray(rgb).save("results/images/habitat_rgb.png")

    depth = observations["depth_sensor"].astype(np.float32)
    np.save("results/images/habitat_depth.npy", depth)

    depth_clip = np.clip(depth, 0.0, 10.0)
    depth_vis = (depth_clip / 10.0 * 255.0).astype(np.uint8)
    Image.fromarray(depth_vis).save("results/images/habitat_depth.png")

    print("Saved RGB image:        results/images/habitat_rgb.png")
    print("Saved depth image:      results/images/habitat_depth.png")
    print("Saved raw depth numpy:  results/images/habitat_depth.npy")
    print("RGB shape:", rgb.shape)
    print("Depth shape:", depth.shape)
    print("Depth min:", float(depth.min()))
    print("Depth max:", float(depth.max()))


def main():
    scene_path = find_scene()
    print("Using scene:", scene_path)

    sim = make_sim(scene_path)
    agent = sim.initialize_agent(0)

    state = agent.get_state()

    if sim.pathfinder.is_loaded:
        state.position = sim.pathfinder.get_random_navigable_point()
        agent.set_state(state)
        print("Agent position:", state.position)
    else:
        print("Warning: navmesh not loaded. Using default agent position.")

    observations = sim.get_sensor_observations()
    save_rgb_depth(observations)

    sim.close()


if __name__ == "__main__":
    main()
