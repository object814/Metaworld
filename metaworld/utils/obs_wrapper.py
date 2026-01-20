import gymnasium as gym
import numpy as np
from gymnasium import spaces

import cv2
import mujoco


class ProprioImageObsWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper, returns a dict observation with:
      {
        "proprio": <robot ee pos (3) + robot ee vel(3) + gripper state (1), float32>,
        "image": <rgb image (H,W,3), uint8>
      }

    Requirements:
      - env must be created with render_mode="rgb_array"
      - env.render() must return an RGB array
    """

    def __init__(self,
                 env: gym.Env,
                 image_height: int = 128,
                 image_width: int = 128,
                 resize_interpolation: int | None = None,
                ):
        super().__init__(env)

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.resize_interpolation = resize_interpolation

        # --- Proprio space ---
        proprio_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )

        # --- Image space ---
        image_space = spaces.Box(
            low=0,
            high=255,
            shape=(self.image_height, self.image_width, 3),
            dtype=np.uint8,
        )

        self.observation_space = spaces.Dict(
            {
                "proprio": proprio_space,
                "image": image_space,
            }
        )

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        """Resize RGB frame to (H,W,3) uint8."""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        if frame.shape[0] == self.image_height and frame.shape[1] == self.image_width:
            return frame

        # Interpolation
        if self.resize_interpolation is None:
            # Use AREA for downsampling, LINEAR for upsampling by default
            interp = cv2.INTER_AREA if (frame.shape[0] > self.image_height or frame.shape[1] > self.image_width) else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (self.image_width, self.image_height), interpolation=interp)
        return resized.astype(np.uint8)

    def observation(self, obs):
        """Return dict observation with proprio and image."""
        # Original observation is assumed to be a 1D array, with first 7 values as proprio
        proprio = np.asarray(obs[:7], dtype=np.float32)

        frame = self.env.render()
        if frame is None:
            raise RuntimeError("env.render() returned None. Use render_mode='rgb_array'.")
        frame = self._resize(np.asarray(frame))
        return {"proprio": proprio, "image": frame}

class ProprioMultiImageObsWrapper(gym.ObservationWrapper):
    """
    Gym environment observation wrapper, returns a dict observation with:
      {
        "proprio": <robot ee pos (3) + robot ee vel(3) + gripper state (1), float32>,
        "images": <rgb image (N,H,W,3), uint8> # staced images from N cameras
      }

    Requirements:
      - env must be created with render_mode="rgb_array"
      - env.render() must return an RGB array
    """

    def __init__(self, 
                 env: gym.Env,
                 image_height: int = 128,
                 image_width: int = 128,
                 camera_names: list[str] = ["topview", "corner", "front"],
                 resize_interpolation: int | None = None
                ):
        super().__init__(env)

        self.image_height = int(image_height)
        self.image_width = int(image_width)
        self.resize_interpolation = resize_interpolation

        # Handle multiple cameras
        self.camera_names = list(camera_names)
        assert len(self.camera_names) > 0, "At least one camera name must be provided."
        # Check camera ids
        self._camera_ids = [self._get_camera_id(name) for name in self.camera_names]
        # Renderer
        self._renderer = mujoco.Renderer(
            self.env.unwrapped.model, 
            height=self.image_height, 
            width=self.image_width,
            )

        # --- Proprio space ---
        proprio_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(7,),
            dtype=np.float32,
        )
        # --- Image space ---
        images_space = spaces.Box(
            low=0,
            high=255,
            shape=(len(camera_names), self.image_height, self.image_width, 3),
            dtype=np.uint8,
        )

        self.observation_space = spaces.Dict(
            {
                "proprio": proprio_space,
                "images": images_space,
            }
        )

    def _get_camera_id(self, camera_name: str) -> int:
        """Get camera id from camera name."""
        m = self.env.unwrapped.model
        # Find camera id by iterating through cameras
        for cam_id in range(m.ncam):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_id)
            if name == camera_name:
                return cam_id
        raise ValueError(f"Camera '{camera_name}' not found. Available cameras: {self._list_camera_names()}")
    
    def _get_multi_camera_frames(self) -> np.ndarray:
        """
        Render all cameras from the current state.
        Returns uint8 array of shape (N,H,W,3).
        """
        data = self.env.unwrapped.data

        imgs = []
        for cam_id in self._camera_ids:
            self._renderer.update_scene(data, camera=cam_id)
            img = self._renderer.render()
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)  # (N,H,W,3)
        return imgs
    
    def observation(self, obs):
        """Return dict observation with proprio and image."""
        # Original observation is assumed to be a 1D array, with first 7 values as proprio
        proprio = np.asarray(obs[:7], dtype=np.float32)
        images = self._get_multi_camera_frames().astype(np.uint8)

        return {"proprio": proprio, "images": images}
    
    def close(self) -> None:
        self._renderer.close()
        return super().close()