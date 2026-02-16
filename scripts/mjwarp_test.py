import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import numpy as np
import imageio
import time

import mujoco
import mujoco_warp as mjw
import warp as wp


def render_gif(frames, out_path, fps=30):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, frames, fps=fps)
    print(f"Wrote {out_path} ({len(frames)} frames)")


def sample_ctrl(mjm: mujoco.MjModel, nworld: int, rng: np.random.Generator):
    nu = mjm.nu
    if nu == 0:
        return np.zeros((nworld, 0), dtype=np.float32)

    lo = mjm.actuator_ctrlrange[:, 0]
    hi = mjm.actuator_ctrlrange[:, 1]

    lo = np.where(np.isfinite(lo), lo, -1.0)
    hi = np.where(np.isfinite(hi), hi, 1.0)

    u = rng.uniform(size=(nworld, nu)).astype(np.float32)
    ctrl = lo[None, :] + u * (hi - lo)[None, :]
    return ctrl.astype(np.float32)


def copy_world_to_cpu(mjm, cpu_data, qpos_i, qvel_i):
    cpu_data.qpos[:] = qpos_i
    cpu_data.qvel[:] = qvel_i
    mujoco.mj_forward(mjm, cpu_data)


def tile_images(images, grid_rows, grid_cols):
    """
    images: list of (H, W, 3) arrays
    returns: single tiled image
    """
    h, w, c = images[0].shape
    grid = np.zeros((h * grid_rows, w * grid_cols, c), dtype=np.uint8)

    idx = 0
    for r in range(grid_rows):
        for c_ in range(grid_cols):
            grid[r*h:(r+1)*h, c_*w:(c_+1)*w] = images[idx]
            idx += 1
    return grid

def randomize_initial_states(mjm, d, rng, position_scale=0.05):
    """
    Randomize initial qpos per world.
    For testing, we add small noise to all qpos.
    """

    qpos = d.qpos.numpy()  # shape (nworld, nq)

    # Add small noise to each world
    noise = rng.normal(scale=position_scale, size=qpos.shape)
    qpos += noise

    # Optional: zero velocities
    qvel = np.zeros_like(d.qvel.numpy())

    wp.copy(d.qpos, wp.array(qpos, dtype=wp.float32))
    wp.copy(d.qvel, wp.array(qvel, dtype=wp.float32))

def main(
    xml_path: str,
    out_gif: str = "gifs/mjwarp_grid.gif",
    nworld: int = 100,
    grid_rows: int = 10,
    grid_cols: int = 10,
    steps: int = 10,
    world_res: int = 64,
    nconmax: int = 64,
    njmax: int = 256,
    seed: int = 0,
    use_graph: bool = True,
):

    assert nworld == grid_rows * grid_cols, "Grid size must match nworld"

    rng = np.random.default_rng(seed)

    print("Loading MuJoCo model...")
    mjm = mujoco.MjModel.from_xml_path(xml_path)

    device = wp.get_preferred_device()
    print("Warp device:", device)

    with wp.ScopedDevice(device):

        print("Uploading model to MJWarp...")
        m = mjw.put_model(mjm)
        d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)

        # Randomize initial states per world
        randomize_initial_states(mjm, d, rng)

        wp.synchronize_device()

        # --------------------
        # Warmup
        # --------------------
        print("Warming up...")
        for _ in range(5):
            mjw.step(m, d)
        wp.synchronize_device()
        print("Warmup complete.")

        # --------------------
        # Graph capture
        # --------------------
        graph = None
        if use_graph:
            print("Capturing graph...")
            with wp.ScopedCapture() as cap:
                mjw.step(m, d)
            graph = cap.graph
            print("Graph captured.")

        # Renderer
        cpu_data = mujoco.MjData(mjm)
        renderer = mujoco.Renderer(mjm, height=world_res, width=world_res)

        frames = []

        print("Starting simulation...")
        start_time = time.time()

        for t in range(steps):

            ctrl = sample_ctrl(mjm, nworld, rng)
            wp.copy(d.ctrl, wp.array(ctrl, dtype=wp.float32))

            if graph is not None:
                wp.capture_launch(graph)
            else:
                mjw.step(m, d)

            # Pull all worlds
            qpos = d.qpos.numpy()
            qvel = d.qvel.numpy()

            world_images = []

            for i in range(nworld):
                copy_world_to_cpu(mjm, cpu_data, qpos[i], qvel[i])
                renderer.update_scene(cpu_data)
                img = renderer.render()
                world_images.append(img)

            grid_img = tile_images(world_images, grid_rows, grid_cols)
            frames.append(grid_img)

            if (t + 1) % 1 == 0:
                print(f"Step {t+1}/{steps}")

        wp.synchronize_device()
        total_time = time.time() - start_time
        print(f"Simulation finished in {total_time:.3f} seconds")
        print(f"Throughput: {(steps * nworld) / total_time:.1f} steps/sec")

        renderer.close()

    render_gif(frames, out_gif, fps=30)


if __name__ == "__main__":

    xml_path = "/Metaworld/metaworld/assets/sawyer_xyz/sawyer_pick_place_v3.xml"

    main(
        xml_path=xml_path,
        out_gif="gifs/mjwarp_pickplace_10x10.gif",
        nworld=100,
        grid_rows=10,
        grid_cols=10,
        steps=10,
        world_res=64,
        nconmax=64,
        njmax=256,
        use_graph=True,
    )
