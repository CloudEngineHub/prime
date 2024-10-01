import pickle
from typing import Any, Protocol
import importlib
from zeroband.utils.logging import get_logger
import aiohttp
from aiohttp import ClientError
import asyncio


async def get_external_ip(max_retries=3, retry_delay=5):
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get('https://api.ipify.org', timeout=10) as response:
                    response.raise_for_status()
                    return await response.text()
            except ClientError as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
    return None


class Monitor(Protocol):
    def __init__(self, project, config): ...

    def log(self, metrics: dict[str, Any]): ...

    def set_stage(self, stage: str): ...

    def finish(self): ...


class HttpMonitor:
    """
    Logs the status of nodes, and training progress to an API
    """

    def __init__(self, config, *args, **kwargs):
        self.data = []
        self.log_flush_interval = config["monitor"]["log_flush_interval"]
        self.base_url = config["monitor"]["base_url"]
        self.auth_token = config["monitor"]["auth_token"]

        self._logger = get_logger()

        self.run_id = config.get("run_id", None)
        if self.run_id is None:
            raise ValueError("run_id must be set for HttpMonitor")

        self.node_ip_address = None
        self.node_ip_address_fetch_status = None

    def _remove_duplicates(self):
        seen = set()
        unique_logs = []
        for log in self.data:
            log_tuple = tuple(sorted(log.items()))
            if log_tuple not in seen:
                unique_logs.append(log)
                seen.add(log_tuple)
        self.data = unique_logs

    def set_stage(self, stage: str):
        import time

        # add a new log entry with the stage name
        self.data.append({"stage": stage, "time": time.time()})
        self._handle_send_batch(flush=True)  # it's useful to have the most up-to-date stage broadcasted

    def log(self, data: dict[str, Any]):
        # Lowercase the keys in the data dictionary
        lowercased_data = {k.lower(): v for k, v in data.items()}
        self.data.append(lowercased_data)

        self._handle_send_batch()

    def _handle_send_batch(self, flush: bool = False):
        if len(self.data) >= self.log_flush_interval or flush:
            import asyncio

            # do this in a separate thread to not affect training loop
            asyncio.create_task(self._send_batch())

    async def _set_node_ip_address(self):
        if self.node_ip_address is None and self.node_ip_address_fetch_status != "failed":
            ip_address = await get_external_ip()
            if ip_address is None:
                self._logger.error("Failed to get external IP address")
                # set this to "failed" so we keep trying again
                self.node_ip_address_fetch_status = "failed"
            else:
                self.node_ip_address = ip_address
                self.node_ip_address_fetch_status = "success"

    async def _send_batch(self):
        import aiohttp

        await self._set_node_ip_address()
        self._remove_duplicates()

        batch = self.data[:self.log_flush_interval]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }
        payload = {
            "node_ip_address": self.node_ip_address,
            "logs": batch
        }
        api = f"{self.base_url}/training_runs/{self.run_id}/logs"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api, json=payload, headers=headers) as response:
                    if response is not None:
                        response.raise_for_status()
        except Exception as e:
            self._logger.error(f"Error sending batch to server: {str(e)}")
            pass

        self.data = self.data[self.log_flush_interval :]
        return True

    def _finish(self):
        import requests

        headers = {"Content-Type": "application/json"}
        api = f"{self.base_url}/training_runs/{self.run_id}/finish"
        try:
            response = requests.post(api, headers=headers)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            self._logger.debug(f"Failed to send finish signal to http monitor: {e}")
            return False

    def finish(self):
        # Send any remaining logs
        while self.data:
            self._send_batch()

        self._finish()


class WandbMonitor:
    def __init__(self, project, config, resume: bool):
        if importlib.util.find_spec("wandb") is None:
            raise ImportError("wandb is not installed. Please install it to use WandbMonitor.")

        import wandb

        wandb.init(
            project=project, config=config, resume="auto" if resume else None
        )  # make wandb reuse the same run id if possible

    def log(self, metrics: dict[str, Any]):
        import wandb

        wandb.log(metrics)

    def set_stage(self, stage: str):
        # no-op
        pass

    def finish(self):
        import wandb

        wandb.finish()


class DummyMonitor:
    def __init__(self, project, config, *args, **kwargs):
        self.project = project
        self.config = config
        open(project, "a").close()  # Create an empty file at the project path

        self.data = []

    def log(self, metrics: dict[str, Any]):
        self.data.append(metrics)

    def set_stage(self, stage: str):
        # no-op
        pass

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
