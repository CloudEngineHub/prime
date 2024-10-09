import subprocess

import shutil

def _get_cut_dirs_from_url(url: str) -> int:
    return len(url.rstrip().partition("//")[-1].split("/"))

def wget(source: str, destination: str) -> None:
    # logger = get_logger()
    cmd = f"wget -r -np -nH --cut-dirs={_get_cut_dirs_from_url(source)} -P {destination} {source}"

    if shutil.which("wget") is None:
        raise RuntimeError("wget is required but not found. Please install wget and try again.")

    try:
        subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        # logger.error(f"Error output: {e.stderr}")
        print(f"Error output: {e.stderr}")
        raise e
