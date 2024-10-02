import torch
from zeroband.comms import ElasticDeviceMesh

edm = ElasticDeviceMesh(backend="gloo")

if edm.world_info.global_rank == 0:
    tensor = torch.randn(1000)
    work = edm.global_pg.send([tensor], 1, 0)
else:
    tensor = torch.randn(1000)
    work = edm.global_pg.recv([tensor], 0, 0)

work.wait()


print(f"Rank {edm.world_info.global_rank}:", tensor[:10])

del edm
