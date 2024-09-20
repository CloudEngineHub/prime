import os

world_info = None

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

def get_world_info() -> WorldInfo:
    """
    Return a WorldInfo singleton.
    """
    global world_info
    if world_info is None:
        world_info = WorldInfo()
    return world_info

