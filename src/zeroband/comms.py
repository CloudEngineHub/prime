import os
from torch.distributed.device_mesh import init_device_mesh
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
import torch.distributed as dist
from datetime import timedelta
import time
from typing import List, Tuple, Optional
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup


TCPSTORE_TIMEOUT = timedelta(seconds=int(os.getenv("ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS", "300")))
TCPSTORE_POLLING_INTERVAL = float(os.getenv("ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS", "0.1"))
MAX_JOINERS = 100  # Maximum number of nodes that can join in a single reinit
MAX_LEAVERS = 100  # Maximum number of nodes that can leave in a single reinit


class ElasticDeviceMesh:
    """A class to manage the process groups for elastic training without restarts.

    The way it works is rank 0 coordinates the joining and leaving of nodes.
    Rank 0 manages the status to coordinate the creation and recreation of the process groups.
    When a node wants to join, rank 0 will setup the store so that all nodes know the new world size and their respective ranks.

    Store keys used:
    - status: "init", "running", "reinit"
    - world_size: The current world size
    - mesh_count: The version of the mesh
    - rank_{uuid}: The rank of the node with the given uuid
    - rank_map_{rank}: The new rank of the node with the given rank. Used to remap ranks when nodes leave.
    - joiner_{i}: The uuid of the ith joiner. Its a KV implmentation of a queue.
    - leaver_{i}: The uuid of the ith leaver. Its a KV implmentation of a queue.
    """

    local_pg: dist.ProcessGroup
    global_pg: dist.ProcessGroup

    def __init__(self):
        self._logger = get_logger()
        self.world_info = get_world_info()

        # Initialize global process group
        self.global_pg = FakeProcessGroup(self.world_info.rank, 1)
        if self.world_info.global_world_size > 1:
            self.global_pg = self._init_global_pg()

        # Initialize local process group
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        self.mesh = init_device_mesh(
            "cuda",
            (self.world_info.nnodes, self.world_info.local_world_size),
            mesh_dim_names=("internode", "intranode"),
        )
        self.local_pg = self.mesh.get_group("intranode")

        if self.world_info.rank == 0:
            self._logger.info(f"global pg world : {self.global_pg.size()}, local pg: {self.local_pg.size()}")
        else:
            self._logger.info(f"local pg world : {self.local_pg.size()}")

    def __del__(self):
        dist.destroy_process_group()

    def _init_global_store_and_status(self):
        """Initialize the global store with mesh_count, joiner_0, leaver_0, and status. Also sets the global status."""
        if self.world_info.global_rank == 0:
            self.global_store.set("mesh_count", "0")
            self.global_store.set("joiner_0", "null")
            self.global_store.set("leaver_0", "null")
            self.global_store.set("status", "init")
            self.global_status = "init"
        else:
            self.global_status = self._wait_for_status()

    def _queue_join(self, unique_id: str):
        """Queue a node to join the mesh."""
        for i in range(MAX_JOINERS):
            joiner_id = self.global_store.get(f"joiner_{i}").decode("utf-8")
            if joiner_id == "null":
                self.global_store.set(f"joiner_{i}", unique_id)
                self.global_store.set(f"joiner_{i + 1}", "null")
                break
        else:
            raise RuntimeError("Too many joiners")

    def _queue_leave(self, unique_id: str):
        """Queue a node to leave the mesh."""
        for i in range(MAX_LEAVERS):
            leaver_id = self.global_store.get(f"leaver_{i}").decode("utf-8")
            if leaver_id == "null":
                self.global_store.set(f"leaver_{i}", unique_id)
                self.global_store.set(f"leaver_{i + 1}", "null")
                break
        else:
            raise RuntimeError("Too many leavers")

    def _get_joiners_and_leavers(self) -> Tuple[List[str], List[str]]:
        joiners = []
        leavers = []
        for i in range(MAX_JOINERS):
            joiner_id = self.global_store.get(f"joiner_{i}").decode("utf-8")
            if joiner_id == "null":
                break
            joiners.append(joiner_id)
        for i in range(MAX_LEAVERS):
            leaver_id = self.global_store.get(f"leaver_{i}").decode("utf-8")
            if leaver_id == "null":
                break
            leavers.append(leaver_id)
        print(f"Joiners: {joiners}, Leavers: {leavers}")
        return joiners, leavers

    def _clear_joiners_and_leavers(self):
        self.global_store.set("joiner_0", "null")
        self.global_store.set("leaver_0", "null")

    def _wait_for_status(self, status: Optional[str] = None) -> str:
        """Wait for status to be set in the store.

        Args:
            store (dist.Store): The store to check.
            status (Optional[str], optional): The status to wait for. If None, wait for any status. Defaults to None.
        Returns:
            status (str): The status.
        """
        while True:
            try:
                ret = self.global_store.get("status").decode("utf-8")
                if status is None or ret == status:
                    return ret
                time.sleep(TCPSTORE_POLLING_INTERVAL)
            except dist.DistStoreError as e:
                if status is not None:
                    raise e
                time.sleep(0.1)

    def _get_assigned_global_rank_and_size(self) -> Tuple[int, int]:
        """Get the assigned global rank from the leader."""
        return

    def _init_global_pg(self) -> dist.ProcessGroup:
        # Each rank gets its own global store with global rank 0 as the master
        self._global_leader = self.world_info.global_rank == 0
        self.global_store = dist.TCPStore(
            host_name=self.world_info.global_addr,
            port=self.world_info.global_port + self.world_info.rank,
            timeout=TCPSTORE_TIMEOUT,
            is_master=self._global_leader,
        )

        # Initialize store
        self._init_global_store_and_status()

        if self.global_status == "init":  # First time init path
            self.mesh_count = 0
            prefix_store = dist.PrefixStore("mesh_0", self.global_store)
        elif self.global_status == "running":  # Join path
            # Ask to join and then wait for the status to be "reinit"
            self._queue_join(self.world_info.global_unique_id)
            self._wait_for_status("reinit")

            # Get the global rank and world size and create a new prefix store
            self.world_info.global_rank = int(
                self.global_store.get(f"rank_{self.world_info.global_unique_id}").decode("utf-8")
            )
            self.world_info.global_world_size = int(self.global_store.get("world_size").decode("utf-8"))
            self.mesh_count = int(self.global_store.get("mesh_count").decode("utf-8"))
            prefix_store = dist.PrefixStore(f"mesh_{self.mesh_count}", self.global_store)
        else:
            # TODO: Could be in "reinit" status. We probably just recurse until running in this case
            raise RuntimeError(f"Unknown status {self.global_status}")
        pg = dist.ProcessGroupGloo(
            prefix_store, self.world_info.global_rank, self.world_info.global_world_size, TCPSTORE_TIMEOUT
        )
        if self._global_leader:
            self.global_store.set("status", "running")
        self.global_store.set(f"rank_{self.world_info.global_unique_id}", str(self.world_info.global_rank))
        self.global_status = "running"

        # Setting instance variables
        self.leaving = False  # TODO: do we need this?
        return pg

    def _resolve_world(self):
        """Set the new world size and ranks for all nodes."""
        # Find joiners and leavers
        joiners, leavers = self._get_joiners_and_leavers()
        # If no joiners or leavers, no resolution needed
        if len(joiners) == 0 and len(leavers) == 0:
            return

        # Remap live ranks to smaller world_size caused by leavers
        leaving_ranks = {int(self.global_store.get(f"rank_{leaver_id}").decode("utf-8")) for leaver_id in leavers}
        live_ranks = [i for i in range(0, self.world_size, self.local_world_size) if i not in leaving_ranks]
        for i, rank in enumerate(live_ranks):
            self.global_store.set(f"rank_map_{rank}", str(i * self.local_world_size))
        new_world_size = len(live_ranks) * self.local_world_size

        # Give joiners new ranks
        for joiner_id in joiners:
            self.global_store.set(f"rank_{joiner_id}", str(new_world_size))
            new_world_size += self.local_world_size

        # Update world_size
        self.global_store.set("world_size", str(new_world_size))
        self.global_store.set("mesh_count", str(self.mesh_count + 1))
        # Set status to "reinit"
        self.global_store.set("status", "reinit")

    def maybe_reinit_device_mesh(self):
        """Reinitialize the device mesh if there are joiners or leavers."""
        if self.rank == 0:
            self._resolve_world()
        dist.barrier()
        status = self.global_store.get("status").decode("utf-8")
        if status == "running":
            return

        print("Reinitializing device mesh")
        dist.destroy_process_group()
        print("Destroyed process group")
        if self.leaving:
            print("Leaving")
            return

        # Check if we got remapped
        prev_uuid_rank = int(self.global_store.get(f"rank_{self.world_info.global_unique_id}").decode("utf-8"))
        new_uuid_rank = int(self.global_store.get(f"rank_map_{prev_uuid_rank}").decode("utf-8"))
        self.rank = new_uuid_rank + self.local_rank

        self.world_size = int(self.global_store.get("world_size").decode("utf-8"))
        self.mesh_count = int(self.global_store.get("mesh_count").decode("utf-8"))
        self.prefix_store = dist.PrefixStore(f"mesh_{self.mesh_count}", self.global_store)
        dist.init_process_group(
            backend="cpu:gloo,cuda:nccl", store=self.prefix_store, rank=self.rank, world_size=self.world_size
        )

        if self.rank == 0:
            self._clear_joiners_and_leavers()
            self.global_store.set("status", "running")
        # Update rank if needed (otherwise, the next remap will do the lookup incorrectly)
        if self.local_rank == 0 and new_uuid_rank != prev_uuid_rank:
            self.global_store.set(f"rank_{self.world_info.global_unique_id}", str(new_uuid_rank))
        # Reinitialize sub process groups
        self.world_rank = self.rank // self.local_world_size
