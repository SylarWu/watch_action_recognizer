class UNetConfig(object):
    def __init__(self):
        self.n_axis = 3
        self.init_channels = 16
        self.pool_sizes = [14, 7, 2]
        self.bottleneck_factor = 4
