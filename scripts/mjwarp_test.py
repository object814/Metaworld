import os
os.environ["MUJOCO_GL"] = "egl"  # headless rendering

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
    ctrl = lo[None, :].astype(np.float32) + u * (hi - lo)[None, :].astype(np.float32)
    return ctrl


def copy_world_to_cpu(mjm: mujoco.MjModel, cpu_data: mujoco.MjData, qpos_i, qvel_i):
    cpu_data.qpos[:] = qpos_i
    cpu_data.qvel[:] = qvel_i
    mujoco.mj_forward(mjm, cpu_data)


def main(
    xml_path: str,
    out_gif: str = "gifs/mjwarp_batch_debug.gif",
    nworld: int = 256,
    steps: int = 300,
    render_world: int = 0,
    width: int = 480,
    height: int = 480,
    nconmax: int = 64,
    njmax: int = 256,
    seed: int = 0,
    use_graph: bool = True,
):

    rng = np.random.default_rng(seed)

    print("Loading MuJoCo model...")
    mjm = mujoco.MjModel.from_xml_path(xml_path)

    device = wp.get_preferred_device()
    print("Warp device:", device)

    with wp.ScopedDevice(device):

        print("Uploading model to MJWarp...")
        m = mjw.put_model(mjm)
        d = mjw.make_data(mjm, nworld=nworld, nconmax=nconmax, njmax=njmax)

        # ----------------------------
        # WARMUP (CRITICAL FIX)
        # ----------------------------
        print("Warming up MJWarp kernels...")
        for i in range(5):
            mjw.step(m, d)
        wp.synchronize_device()
        print("Warmup complete.")

        # ----------------------------
        # CUDA Graph Capture (optional)
        # ----------------------------
        graph = None
        if use_graph:
            print("Capturing CUDA graph...")
            with wp.ScopedCapture() as cap:
                mjw.step(m, d)
            graph = cap.graph
            print("Graph captured.")

        # CPU renderer
        cpu_data = mujoco.MjData(mjm)
        renderer = mujoco.Renderer(mjm, height=height, width=width)
        mujoco.mj_forward(mjm, cpu_data)

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

            # Pull state for visualization
            qpos = d.qpos.numpy()
            qvel = d.qvel.numpy()

            copy_world_to_cpu(mjm, cpu_data, qpos[render_world], qvel[render_world])

            renderer.update_scene(cpu_data)
            img = renderer.render()
            frames.append(img)

            if (t + 1) % 50 == 0:
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
        out_gif="gifs/mjwarp_pickplace_batch.gif",
        nworld=256,
        steps=200,
        render_world=0,
        width=480,
        height=480,
        nconmax=64,
        njmax=256,
        use_graph=True,
    )
