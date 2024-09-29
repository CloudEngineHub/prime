import pickle
from typing import Any, Protocol
import importlib
from zeroband.utils.logging import get_logger

logger = get_logger()


class Monitor(Protocol):
    def __init__(self, project, config): ...

    def log(self, metrics: dict[str, Any]): ...

    def finish(self): ...


class HttpMonitor:
    """
    Logs the status of nodes, and training progress to an API
    """

    def __init__(self, config, *args, **kwargs):
        self.data = []
        self.batch_size = getattr(config.metric_logger, 'batch_size', 10)
        self.base_url = config['metric_logger']['base_url']
        self.auth_token = config['metric_logger']['auth_token']

        self.run_id = config.get('run_id', None)
        if self.run_id is None:
            raise ValueError("run_id must be set for HttpMonitor")

    def _remove_duplicates(self):
        seen = set()
        unique_logs = []
        for log in self.data:
            log_tuple = tuple(sorted(log.items()))
            if log_tuple not in seen:
                unique_logs.append(log)
                seen.add(log_tuple)
        self.data = unique_logs

    def log(self, data: dict[str, Any]):
        import asyncio

        # Lowercase the keys in the data dictionary
        lowercased_data = {k.lower(): v for k, v in data.items()}
        self.data.append(lowercased_data)
        if len(self.data) >= self.batch_size:
            # do this in a separate thread to not affect training loop
            asyncio.create_task(self._send_batch())

    async def _send_batch(self):
        import aiohttp

        self._remove_duplicates()
        batch = self.data[:self.batch_size]
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.auth_token}"
        }
        payload = {
            "logs": batch
        }
        api = f"{self.base_url}/training_runs/{self.run_id}/logs"

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(api, json=payload, headers=headers) as response:
                    await response.raise_for_status()
            except aiohttp.ClientError as e:
                logger.debug(f"Failed to send batch of logs to http monitor: {e}")
                return False

        self.data = self.data[self.batch_size:]
        return True

    def _finish(self):
        import requests
        headers = {
            "Content-Type": "application/json"
        }
        api = f"{self.base_url}/training_runs/{self.run_id}/finish"
        try:
            response = requests.post(api, headers=headers)
            response.raise_for_status()
            return True
        except requests.RequestException as e:
            logger.debug(f"Failed to send finish signal to http monitor: {e}")
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

    def finish(self):
        with open(self.project, "wb") as f:
            pickle.dump(self.data, f)
