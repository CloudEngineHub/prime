from zeroband.utils.wget import wget

path = "my_outputs"
dest_rank = 1
wget(
    source=f"http://localhost:{8000+dest_rank}/latest/diloco_{dest_rank}",
    destination=path,
)
wget(
    source=f"http://localhost:{8000+dest_rank}/latest/diloco_{dest_rank}/.metadata",
    destination=path,
)
