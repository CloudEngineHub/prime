import sys
import os
import time
from pydantic import BaseModel, ValidationError
from torch.distributed.device_mesh import init_device_mesh
from zeroband.utils.world_info import get_world_info
from zeroband.utils.logging import get_logger
import torch.distributed as dist
from datetime import timedelta
from typing import Callable, List, Tuple, Optional
from torch.testing._internal.distributed.fake_pg import FakeProcessGroup
import multiprocessing as mp

TCPSTORE_TIMEOUT = timedelta(seconds=int(os.getenv("ZERO_BAND_GLOBAL_STORE_TIMEOUT_SECONDS", "300")))
TCPSTORE_POLLING_INTERVAL = float(os.getenv("ZERO_BAND_GLOBAL_STORE_POLLING_INTERVAL_SECONDS", "0.1"))
MAX_JOINERS = 100  # Maximum number of nodes that can join in a single reinit
HEARTBEAT_INTERVAL = int(
    os.getenv("ZERO_BAND_EDM_HEARTBEAT_INTERVAL_SECONDS", "2")
)  # Interval in seconds between heartbeats
HEARTBEAT_TIMEOUT = int(
    os.getenv("ZERO_BAND_EDM_HEARTBEAT_TIMEOUT_SECONDS", "10")
)  # Time in seconds after which a node is considered dead if no heartbeat is received


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

        # Start heartbeat
        self._start_heartbeat()

        # Logging
        self._logger.info(f"global_pg size : {self.global_pg.size()}, local_pg size: {self.local_pg.size()}")

    def __del__(self):
        self._stop_heartbeat()
        dist.destroy_process_group()

    def _init_global_store_and_status(self):
        """Initialize the global store with mesh_count, joiner_0, and status. Also sets the global status."""
        if self._global_leader:
            self.global_store.set("mesh_count", "0")
            self.global_store.set("joiner_0", "null")
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

    def _get_joiners(self) -> Tuple[List[str], List[str]]:
        joiners = []
        for i in range(MAX_JOINERS):
            joiner_id = self.global_store.get(f"joiner_{i}").decode("utf-8")
            if joiner_id == "null":
                break
            joiners.append(joiner_id)
        return joiners

    def _clear_joiners(self):
        self.global_store.set("joiner_0", "null")

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
            self.global_store.set("resolved_time", str(time.time()))
        self.global_status = "running"
        self.global_store.set(f"rank_{self.world_info.global_unique_id}", str(self.world_info.global_rank))

        # Setting instance variables
        self.leaving = False  # TODO: do we need this?
        # This is to match the barrier in maybe_reinit_global_pg.
        # We might be able to get away with only doing in joining path.
        # Let's not risk it for now though.
        dist.barrier(self.global_pg)
        self._last_resolved_time = self.global_store.get("resolved_time").decode("utf-8")
        self.live_recovery = LiveRecovery(self.global_store)

        self._logger.info(
            f"Elastic Device mesh init done with {self.global_pg.size()} peers in {time.perf_counter() - time_start} seconds"
        )

    def _start_heartbeat(self):
        """Start sending heartbeats to the global store in a separate process."""
        self._heartbeat_stop_event = mp.Event()
        self._heartbeat_process = mp.Process(target=self._heartbeat_loop, args=(self._heartbeat_stop_event,))
        self._heartbeat_process.start()

    def _stop_heartbeat(self):
        """Stop the heartbeat process."""
        self._send_deathrattle()
        if hasattr(self, "_heartbeat_stop_event"):
            self._heartbeat_stop_event.set()
            self._heartbeat_process.join()

    def _heartbeat_loop(self, stop_event: mp.Event):
        """Continuously send heartbeats until stopped."""
        try:
            while not stop_event.is_set():
                self._send_heartbeat()
                time.sleep(HEARTBEAT_INTERVAL)
        finally:
            self._send_deathrattle()

    def _send_heartbeat(self):
        """Send a heartbeat to the global store."""
        current_time = time.time()
        try:
            self.global_store.set(f"heartbeat_{self.world_info.global_rank}", str(current_time))
        except Exception:
            pass

    def _send_deathrattle(self):
        """Send a deathrattle to the global store."""
        if hasattr(self, "global_store"):
            self.global_store.set(f"heartbeat_{self.world_info.global_rank}", "-100")
        else:
            import warnings

            warnings.warn("global_store garbage collected. Skipping deathrattle.")

    def _check_heartbeats(self) -> List[str]:
        """Check heartbeats and return a list of nodes that have missed their heartbeats."""
        dead_nodes = []
        current_time = time.time()
        for i in range(self.world_info.global_world_size):
            try:
                last_heartbeat = float(self.global_store.get(f"heartbeat_{i}").decode("utf-8"))
                if current_time - last_heartbeat > HEARTBEAT_TIMEOUT:
                    dead_nodes.append(i)
            except dist.DistStoreError:
                self._logger.warning(f"Node {i} has no heartbeat")
        return dead_nodes

    def _resolve_world(self):
        """Set the new world size and ranks for all nodes if there are joiners or dead nodes. Else, do nothing."""
        # Find joiners
        joiners = self._get_joiners()

        # Check for dead nodes
        dead_nodes = self._check_heartbeats()
        self._logger.debug(f"Joiners: {joiners}, Dead nodes: {dead_nodes}")

        # If no joiners or dead nodes, no resolution needed
        if len(joiners) == 0 and len(dead_nodes) == 0:
            return

        # Remap live ranks to smaller world_size caused by dead nodes
        leaving_ranks = set(dead_nodes)
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
        """Reinitialize the global_pg if there are joiners or dead nodes."""

        if self.world_info.global_world_size == 1:
            # no op if we only have one node
            return

        time_start = time.perf_counter()
        self._logger.debug("Resolving world")

        self.live_recovery.stop_background_loop()  # need to stop live recovery loop to avoid deadlocks

        if self._global_leader:
            self._resolve_world()
            self.global_store.set("resolved_time", str(time.time()))
        else:
            while (ans := self.global_store.get("resolved_time").decode("utf-8")) == self._last_resolved_time:
                time.sleep(TCPSTORE_POLLING_INTERVAL)
            self._last_resolved_time = ans

        self._logger.debug("World resolved in %s seconds", time.perf_counter() - time_start)

        status = self.global_store.get("status").decode("utf-8")
        if status == "running":  # No joiners or dead nodes
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
        self._logger.info("Reinitialized global_pg in %s seconds", time.perf_counter() - time_start)

        if self._global_leader:
            self._clear_joiners()
            self.global_store.set("status", "running")

        # Update rank if needed (otherwise, the next remap will do the lookup incorrectly)
        if old_global_rank != self.world_info.global_rank:
            self.global_store.set(f"rank_{self.world_info.global_unique_id}", str(self.world_info.global_rank))
        # Without this barrier, a node might queue leave before the leaving queue is cleared
        dist.barrier(self.global_pg)

        self.live_recovery.init_background_loop()
        self._logger.debug("Reinitialized global_pg in %s seconds", time.perf_counter() - time_start)

    def get_global_pg(self, maybe_reinit: bool = False) -> dist.ProcessGroup:
        """Get the global process group. If maybe_reinit is True, reinitialize the global process group if needed."""
        if maybe_reinit:
            self.maybe_reinit_global_pg()
        return self.global_pg


class LiveRecoveryModel(BaseModel):
    dest_rank: int
    src_rank: int | None = None


class LiveRecoveryStore:
    """
    Wrapping a Store to get and set a Pydantic Base Model
    """

    def __init__(self, store: dist.Store):
        self.store = store
        self._logger = get_logger()

    def set(self, key: str, data: LiveRecoveryModel | None) -> None:
        if data is None:
            self.store.set(key, "null")
        else:
            self.store.set(key, data.model_dump_json())

    def get(self, key: str, fail_on_error: bool = False) -> LiveRecoveryModel | None:
        data = self.store.get(key).decode("utf-8")
        if data == "null":
            return None
        try:
            return LiveRecoveryModel.model_validate_json(data)
        except ValidationError as e:
            self._logger.debug(f"Catching validation error {e} while getting live recovery data {data}")
            if fail_on_error:
                raise e
            return None


class LiveRecovery:
    """Handle the live recovery:

    under the hood:

    Each node check indefinitly the "live_recovery" key in the store.
    If the key is set by a node, it means that the node + 1 need a live recovery.
    The node + 1 set the key with its rank and start sending the ckpt to the node.
    The node + 1 wait for the ckpt to be ack by the node and then remove the key from the store.
    """

    _live_recovery_key: str = "live_recovery"

    def __init__(self, global_store: dist.Store):
        self.global_store = global_store

        self._logger = get_logger()
        self.world_info = get_world_info()

        self.live_ckpt_store = LiveRecoveryStore(dist.PrefixStore("live_ckpt", self.global_store))
        self.live_ckpt_store.set(self._live_recovery_key, None)

        self._dest_rank = mp.Value("i", -1)

        self.init_background_loop()

    def init_background_loop(self) -> mp.Process:
        self._stop_event = mp.Event()
        self._live_recovery_process = mp.Process(
            target=self._live_recovery_loop, args=(self._stop_event, self._dest_rank)
        )
        self._live_recovery_process.start()

    def stop_background_loop(self):
        self._stop_event.set()
        self._live_recovery_process.join()

    def __del__(self):
        self.stop_background_loop()

    def _live_recovery_loop(self, stop_event: mp.Event, dest_rank: mp.Value) -> None:
        while not stop_event.is_set():
            data = self.live_ckpt_store.get(self._live_recovery_key)
            # self._logger.debug(f"data: {data}")
            if data is None:
                continue
            else:
                # only the rank + 1 send the live ckpt, this is to avoid deadlock
                # todo: could be more optimized in term of bandwidth if we send to the closest rank
                if data.dest_rank == self.world_info.global_rank + 1:
                    data.src_rank = self.world_info.global_rank
                    self.live_ckpt_store.set(self._live_recovery_key, data)
                    dest_rank.value = data.src_rank

            time.sleep(HEARTBEAT_INTERVAL)

    def should_send_live_ckpt(self) -> int | None:
        """Return the rank to send the live ckpt to and None if not needed to send.
        Will return None most of the time
        """

        if self._dest_rank.value != -1:
            dest_rank = self._dest_rank.value
            self._dest_rank.value = -1
            return dest_rank
        return None

    def live_ckpt_done_callback(self) -> Callable:
        def callback():
            self.live_ckpt_store.set(self._live_recovery_key, None)
            self._dest_rank.value = -1

        return callback

    def get_src(self) -> int:
        """
        Send a signal to all other node that this node need a live recovery. Await for a response from all other nodes
        and return a src rank from which to pool which should have start sending the live ckpt.

        This function can take up to TCPSTORE_TIMEOUT seconds to complete.
        """
        # todo make this live reco in parralel

        time_start = time.perf_counter()
        self._logger.info("Waiting for live recovery source")

        data = self.live_ckpt_store.get(self._live_recovery_key)

        if data is not None:
            raise RuntimeError(
                "Live recovery already in progress, cannot handle multiple live recovery at the same time for now"
            )

        self.live_ckpt_store.set(self._live_recovery_key, LiveRecoveryModel(dest_rank=self.world_info.global_rank))

        while time.perf_counter() - time_start < TCPSTORE_TIMEOUT.total_seconds():
            value = self.live_ckpt_store.get("need_live_recovery").decode("utf-8")
            # self._logger.debug(f"value: {value}")
            if value != str(self.world_info.global_rank):
                data = self.live_ckpt_store.get(self._live_recovery_key)

                if data.dest_rank == self.world_info.global_rank:
                    if data.src_rank is not None:
                        self.live_ckpt_store.set(self._live_recovery_key, None)
                        return data.src_rank

            time.sleep(TCPSTORE_POLLING_INTERVAL)

        raise RuntimeError("Timed out waiting for live recovery")
