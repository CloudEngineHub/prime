import os

class WorldInfo:
    """This class parse env var about torch world into class variables."""
    world_size: int
    rank: int
    local_rank: int
    local_world_size: int

    def __init__(self):
        self.world_size = int(os.environ["WORLD_SIZE"])
        self.rank = int(os.environ["RANK"])
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.local_world_size = int(os.environ["LOCAL_WORLD_SIZE"])