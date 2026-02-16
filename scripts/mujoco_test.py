import os
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import numpy as np
import imageio
import time

import mujoco


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


def tile_images(images, grid_rows, grid_cols):
    h, w, c = images[0].shape
    grid = np.zeros((h * grid_rows, w * grid_cols, c), dtype=np.uint8)

    idx = 0
    for r in range(grid_rows):
        for c_ in range(grid_cols):
            grid[r*h:(r+1)*h, c_*w:(c_+1)*w] = images[idx]
            idx += 1
    return grid


def randomize_initial_states_cpu(mjm, datas, rng, position_scale=0.05):
    for d in datas:
        d.qpos[:] += rng.normal(scale=position_scale, size=d.qpos.shape)
        d.qvel[:] = 0
        mujoco.mj_forward(mjm, d)


def main(
    xml_path: str,
    out_gif: str = "gifs/mujoco_cpu_grid.gif",
    nworld: int = 100,
    grid_rows: int = 10,
    grid_cols: int = 10,
    steps: int = 10,
    world_res: int = 64,
    seed: int = 0,
):

    assert nworld == grid_rows * grid_cols

    rng = np.random.default_rng(seed)

    print("Loading MuJoCo model...")
    mjm = mujoco.MjModel.from_xml_path(xml_path)

    # Create independent worlds (MjData instances)
    datas = [mujoco.MjData(mjm) for _ in range(nworld)]

    print("Randomizing initial states...")
    randomize_initial_states_cpu(mjm, datas, rng)

    # Warmup
    print("Warming up...")
    for _ in range(5):
        for d in datas:
            mujoco.mj_step(mjm, d)
    print("Warmup complete.")

    # Renderer
    renderer = mujoco.Renderer(mjm, height=world_res, width=world_res)

    frames = []

    print("Starting simulation...")
    start_time = time.time()

    for t in range(steps):

        ctrl = sample_ctrl(mjm, nworld, rng)

        world_images = []

        for i, d in enumerate(datas):
            d.ctrl[:] = ctrl[i]
            mujoco.mj_step(mjm, d)

            renderer.update_scene(d)
            img = renderer.render()
            world_images.append(img)

        grid_img = tile_images(world_images, grid_rows, grid_cols)
        frames.append(grid_img)

        print(f"Step {t+1}/{steps}")

    total_time = time.time() - start_time

    print(f"Simulation finished in {total_time:.3f} seconds")
    print(f"Throughput: {(steps * nworld) / total_time:.1f} steps/sec")

    renderer.close()

    render_gif(frames, out_gif, fps=30)


if __name__ == "__main__":

    xml_path = "/Metaworld/metaworld/assets/sawyer_xyz/sawyer_pick_place_v3.xml"

    main(
        xml_path=xml_path,
        out_gif="gifs/mujoco_cpu_pickplace_10x10.gif",
        nworld=100,
        grid_rows=10,
        grid_cols=10,
        steps=10,
        world_res=64,
    )
