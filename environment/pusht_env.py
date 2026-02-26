import gymnasium 
import numpy as np

class PushTEnv():
    def __init__(self, render_mode, obs_type):

        self.gym = 
        
    def reset(self) -> np.ndarray:
        observation, _ = self.env.reset()

        return observation
    
    def get_success(self, info: dict) -> bool:
        _, _, truncated, done = info

        if truncated or done:
            return True
        else: 
            return False