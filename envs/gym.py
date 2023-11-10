import gym
import cv2
import numpy as np

class Gym:
    def __init__(self, name, action_repeat=1, size=(64, 64), seed=0):
        self._env = gym.make(name, render_mode='rgb_array')
        self.domain = 'gym'
        self.task = name
        self._action_repeat = action_repeat
        self._size = size
        self.reward_range = [-np.inf, np.inf]


    @property
    def observation_space(self):
        spaces = {}
        spaces['states'] = self._env.observation_space
        spaces["image"] = gym.spaces.Box(0, 255, self._size + (3,), dtype=np.uint8)
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        return self._env.action_space
    
    def sim_state_to_array(self, sim_state):
        return np.concatenate([sim_state.qpos, sim_state.qvel])
    
    def step(self, action):
        sim_state = self._env.sim.get_state()
        observation, reward, termination, truncation, info = self._env.step(action)
        next_sim_state = self._env.sim.get_state()
        info.update({"next_sim_state": self.sim_state_to_array(next_sim_state),
                "sim_state": self.sim_state_to_array(sim_state)
                })
        obs = {'states': observation, 'is_terminal': termination, 'is_first': False}
        obs['image'] = cv2.resize(self._env.render(), self._size)
        termination = termination or truncation
        return obs, reward, termination, info
    
    def set_state(self, obs, sim_state):
        self._env.set_state(sim_state[:len(sim_state)//2], sim_state[len(sim_state)//2:])

    def reset(self):
        info = {}
        observation, _ = self._env.reset()
        obs = {}
        obs['states'] = observation
        obs['is_terminal'] = False
        obs['is_first'] = True
        obs['image'] = cv2.resize(self._env.render(), self._size)
        sim_state = self._env.sim.get_state()
        return obs

    def get_sim_state(self):
        return self.sim_state_to_array(self._env.sim.get_state())
    
    def get_terminals(self, states):
        if "Hopper" in self._env.spec.id:
            height = states[:, :, 0]
            angle = states[:, :, 1]
            not_done =  np.isfinite(states).all(axis=-1) \
                        * np.abs(states < 100).all(axis=-1) \
                        * (height > .7) \
                        * (np.abs(angle) < .2)
            done = ~not_done
            return done
        elif "Walker" in self._env.spec.id:
            height = states[:, :, 0]
            angle = states[:, :, 1]
            not_done =  (height > 0.8) \
                        * (height < 2.0) \
                        * (angle > -1.0) \
                        * (angle < 1.0)
            done = ~not_done
            return done
        elif "Cheetah" in self._env.spec.id:
            return np.full((states.shape[0], states.shape[1]), False)
        else:
            raise NotImplementedError


class DWMBufferToEnv:

    def __init__(self, buffer):
        self.data_buffer = buffer
        self.ep_num = None
        self.step_num = None
        self.total_steps = 0
        self.id = ""

    def reset(self):
        if self.ep_num is None:
            self.ep_num = 0
        else:
            self.ep_num += 1
        self.step_num = 0
        self.total_steps += 1
        self.id = str(self.ep_num)
        print("Episode Number: ", self.ep_num, "Total Steps: ", self.total_steps)

        self.current_path_length = self.data_buffer._dict["path_lengths"][self.ep_num]
        info = {}
        observation = self.data_buffer._dict["observations"][self.ep_num][self.step_num]
        obs = {}
        obs['states'] = observation
        obs['is_terminal'] = False
        obs['is_first'] = True
        self.current_sim_state = self.data_buffer._dict["sim_states"][self.ep_num][self.step_num]
        return obs
    
    def get_sim_state(self):
        return self.current_sim_state
    
    def step(self, action):
        sim_state = self.data_buffer._dict["sim_states"][self.ep_num][self.step_num]

        # "step the env"
        self.step_num += 1
        self.total_steps += 1
        info = {}
        next_sim_state = self.data_buffer._dict["sim_states"][self.ep_num][self.step_num]
        self.current_sim_state = next_sim_state
        info.update({"next_sim_state": next_sim_state,
                "sim_state": sim_state,
                "action_taken": self.data_buffer._dict["actions"][self.ep_num][self.step_num],
                })
        
        observation = self.data_buffer._dict["observations"][self.ep_num][self.step_num]
        reward = self.data_buffer._dict["rewards"][self.ep_num][self.step_num][0]
        termination = bool(self.data_buffer._dict["terminals"][self.ep_num][self.step_num])
        obs = {'states': observation, 'is_terminal': termination, 'is_first': False}

        if self.step_num == self.current_path_length - 1:
            termination = True
        return obs, reward, termination, info