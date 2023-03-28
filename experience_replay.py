# experience replay

# importing the libraries
import numpy as np
from collections import namedtuple, deque

# defining one step
step = namedtuple("step", ["state", "action", "done"])

#making thee AI progress on sevral (n_steps) steps

class NStepProgress:
    
    def __init__(self, env, ai, n_step):
        self.ai = ai
        self.rewards = []
        self.env = env
        self.n_step = n_step
        
    def __iter__(self):
        state = self.env.reset()
        history = deque()
        reward = 0.0
        while True:
            action = self.ai(np.array([state]))[0][0]
            next_state, r, is_done, _ = self.env(action)
            reward += r
            history.append(Step(state = state, action = action, reward = r, done = is_done))
            while len(history) > self.n_step + 1:
                history.popleft()
            if len(history) == self.n_step +1:
                yield tuple(history)
            state = next_state
            if is_done:
                if len(history) >= 1:
                    history.popleft()
                while len(history) >= 1:
                    yield tuple(history)
                    history.popleft()
                    self.rewards.append(reward)
                    reward = 0.0
                    state = self.env.reset()
                    history.clear()
                    
    def rewards_steps(self):
        rewards_steps = self.rewards
        self.rewards = []
        return rewards_steps
    
# Implementing Experience Replay

class ReplayMemory:
    
    def __init__(self, n_step, capacity = 1000):
        self.capacity = capacity
        self.n_steps = n_steps
        self.n_steps_iter = iter(n_steps)
        self.buffer = deque()
        
    def sample_batch(self, batch_size): #creates an itreator that returns a random batches
        ofs = 0
        vals = list(self.buffer)
        np.random.shuffle(vals)
        while (ofs+1)*batch_size <= len(self.buffer):
            yield vals[ofs*batch_size:(ofs+1)*batch_size]
            ofs += 1
            
        def run_steps(self, samples):
            while samples > 0:
                entry = next(self.n_steps_iter) # 10 consecutive steps
                self.buffer.append(entry) # we put 200 for the current episode
                samples -= 1 
            while len(self.buffer) > self.capacity: # we accumlate than the capacity (1000)
                self.buffer.popleft 
        
        
        
        
        
        
        
        
        
        
        
        
        
        