import subprocess

from zeroband.utils.logging import get_logger


def wget(source: str, destination: str) -> None:
    logger = get_logger()
    cmd = f"wget -r -np -nH --cut-dirs=1 -P {destination} {source}"

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Error output: {e.stderr}")
        raise e
