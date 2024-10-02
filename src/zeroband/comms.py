import sys
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

    def __init__(self, backend: str = "cpu:gloo,cuda:nccl"):
        self._logger = get_logger()
        self.world_info = get_world_info()

        # Initialize global process group
        self.global_pg = FakeProcessGroup(self.world_info.rank, 1)
        if self.world_info.global_world_size > 1:
            self._init_global_pg()

        # Initialize local process group
        dist.init_process_group(backend=backend)
        self.mesh = init_device_mesh(
            "cuda",
            (self.world_info.nnodes, self.world_info.local_world_size),
            mesh_dim_names=("internode", "intranode"),
        )
        self.local_pg = self.mesh.get_group("intranode")

        # Logging
        self._logger.info(f"global_pg size : {self.global_pg.size()}, local_pg size: {self.local_pg.size()}")

    def __del__(self):
        dist.destroy_process_group()

    def _init_global_store_and_status(self):
        """Initialize the global store with mesh_count, joiner_0, leaver_0, and status. Also sets the global status."""
        if self._global_leader:
            self.global_store.set("mesh_count", "0")
            self.global_store.set("joiner_0", "null")
            self.global_store.set("leaver_0", "null")
            self.global_store.set("status", "init")
            self.global_status = "init"
        else:
            self.global_status = self._wait_for_status()

    def _queue_join(self):
        """Queue a node to join the mesh."""
        for i in range(MAX_JOINERS):
            joiner_id = self.global_store.get(f"joiner_{i}").decode("utf-8")
            if joiner_id == "null":
                self.global_store.set(f"joiner_{i}", self.world_info.global_unique_id)
                self.global_store.set(f"joiner_{i + 1}", "null")
                break
        else:
            raise RuntimeError("Too many joiners")

    def _queue_leave(self):
        """Queue a node to leave the mesh."""
        self.leaving = True
        for i in range(MAX_LEAVERS):
            leaver_id = self.global_store.get(f"leaver_{i}").decode("utf-8")
            if leaver_id == "null":
                self._logger.debug(f"Queueing leaver {self.world_info.global_unique_id} at index {i}")
                self.global_store.set(f"leaver_{i}", self.world_info.global_unique_id)
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
        self._logger.debug(f"Joiners: {joiners}, Leavers: {leavers}")
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

    def _init_global_pg(self) -> None:
        # Each rank gets its own global store with global rank 0 as the master
        time_start = time.perf_counter()
        self._logger.info(
            f"Elastic Device mesh init: Looking for peers via {self.world_info.global_addr}:{self.world_info.global_port}"
        )
        self._global_leader = self.world_info.global_rank == 0
        self.global_store = dist.TCPStore(
            host_name=self.world_info.global_addr,
            port=self.world_info.global_port + self.world_info.rank,
            timeout=TCPSTORE_TIMEOUT,
            is_master=self._global_leader,
        )
        self._logger.debug(
            f"Global store created at {self.world_info.global_addr}:{self.world_info.global_port + self.world_info.rank}"
        )

        # Initialize store values
        self._init_global_store_and_status()

        # Initialize prefix store
        if self.global_status == "init":  # First time init path
            self.mesh_count = 0  # TODO: privatize?
            prefix_store = dist.PrefixStore("mesh_0", self.global_store)
        elif self.global_status == "running":  # Join path
            # Ask to join and then wait for the status to be "reinit"
            self._logger.info("Waiting to join")
            self._queue_join()
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

        # Create process group
        self.global_pg = dist.ProcessGroupGloo(
            prefix_store, self.world_info.global_rank, self.world_info.global_world_size, TCPSTORE_TIMEOUT
        )

        # Update global store values
        if self._global_leader:
            self.global_store.set("status", "running")
        self.global_status = "running"
        self.global_store.set(f"rank_{self.world_info.global_unique_id}", str(self.world_info.global_rank))

        # Setting instance variables
        self.leaving = False  # TODO: do we need this?
        # This is to match the barrier in maybe_reinit_global_pg.
        # We might be able to get away with only doing in joining path.
        # Let's not risk it for now though.
        dist.barrier(self.global_pg)
        self._logger.info(
            f"Elastic Device mesh init done with {self.global_pg.size()} peers in {time.perf_counter() - time_start} seconds"
        )

    def _resolve_world(self):
        """Set the new world size and ranks for all nodes if there are joiners or leavers. Else, do nothing."""
        # Find joiners and leavers
        joiners, leavers = self._get_joiners_and_leavers()
        # If no joiners or leavers, no resolution needed
        if len(joiners) == 0 and len(leavers) == 0:
            return

        # Remap live ranks to smaller world_size caused by leavers
        leaving_ranks = {int(self.global_store.get(f"rank_{leaver_id}").decode("utf-8")) for leaver_id in leavers}
        live_ranks = [i for i in range(self.world_info.global_world_size) if i not in leaving_ranks]
        for i, rank in enumerate(live_ranks):
            self.global_store.set(f"rank_map_{rank}", str(i))
        new_world_size = len(live_ranks)

        # Give joiners new ranks
        for joiner_id in joiners:
            self.global_store.set(f"rank_{joiner_id}", str(new_world_size))
            new_world_size += 1

        # Update world_size
        self.global_store.set("world_size", str(new_world_size))
        self.global_store.set("mesh_count", str(self.mesh_count + 1))
        # Set status to "reinit"
        self.global_store.set("status", "reinit")

    def maybe_reinit_global_pg(self):
        """Reinitialize the global_pg if there are joiners or leavers."""
        if self._global_leader:
            self._resolve_world()
        dist.barrier(self.global_pg)
        status = self.global_store.get("status").decode("utf-8")
        if status == "running":  # No joiners or leavers
            return

        # Reinit Path
        self._logger.info("Reinitializing global_pg")
        if sys.getrefcount(self.global_pg) > 2:
            self._logger.warning(
                f"Global PG refcount was {sys.getrefcount(self.global_pg)} when 2 is expected during deletion. This may cause a memory leak."
            )
        del self.global_pg
        self._logger.info("Destroyed process group")
        if self.leaving:
            self._logger.info("Leaving")
            return

        # Check if we got remapped
        old_global_rank = self.world_info.global_rank
        self.world_info.global_rank = int(
            self.global_store.get(f"rank_map_{self.world_info.global_rank}").decode("utf-8")
        )

        self.world_info.global_world_size = int(self.global_store.get("world_size").decode("utf-8"))
        self.mesh_count = int(self.global_store.get("mesh_count").decode("utf-8"))
        self._logger.debug(
            f"New global rank: {self.world_info.global_rank}, New global world size: {self.world_info.global_world_size} New mesh count: {self.mesh_count}"
        )
        prefix_store = dist.PrefixStore(f"mesh_{self.mesh_count}", self.global_store)

        # Create process group
        self.global_pg = dist.ProcessGroupGloo(
            prefix_store, self.world_info.global_rank, self.world_info.global_world_size, TCPSTORE_TIMEOUT
        )

        if self._global_leader:
            self._clear_joiners_and_leavers()
            self.global_store.set("status", "running")

        # Update rank if needed (otherwise, the next remap will do the lookup incorrectly)
        if old_global_rank != self.world_info.global_rank:
            self.global_store.set(f"rank_{self.world_info.global_unique_id}", str(self.world_info.global_rank))
        # Without this barrier, a node might queue leave before the leaving queue is cleared
        dist.barrier(self.global_pg)
