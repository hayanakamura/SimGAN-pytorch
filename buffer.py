import numpy as np

class Buffer():
    def __init__(self, rng):
        self.rng = rng
        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size

        image_dims = [config.channels, config.height, config.width]

        self.buff = np.zeros([self.buffer_size] + image_dims)
        self.idx = 0

    def add_to_buffer(self, batchs):
        bs = len(batchs)

        if self.idx + bs > self.buffer_size:
            random_idx1 = self.rng.choice(self.idx, bs//2)
            random_idx2 = self.rng.choice(bs, bs//2)
            self.data[random_idx1] = batchs[random_idx2]
        else:
            self.data[self.idx:self.idx+bs] = batchs
            self.idx += bs

    def get_from_buffer(self,):
        assert  self.idx > n, ''
        if n is None:
            n = self.batch_size//2
        random_idx = self.rng.choice(self.idx, n)
        
        return self.data[random_idx]
