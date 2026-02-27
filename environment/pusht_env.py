import gymnasium 
import numpy as np
import pusht

class PushTEnv():
    def __init__(self, render_mode, obs_type):

        self.env = pusht.make("PushT-v0", render_mode=render_mode, obs_type=obs_type)
        
    def reset(self) -> np.ndarray:
        observation, _ = self.env.reset()

        return observation
    
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated

        return observation, reward, done, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()
    
    def get_success(self, info: dict) -> bool:
        is_success = info.get("is_success", False)
        return is_success