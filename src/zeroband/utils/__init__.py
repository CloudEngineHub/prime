from torch.distributed.fsdp import ShardingStrategy


__all__ = ["get_sharding_strategy"]


def get_sharding_strategy(sharding_strategy: str) -> ShardingStrategy:
    if sharding_strategy == "FULL_SHARD":
        return ShardingStrategy.FULL_SHARD
    elif sharding_strategy == "SHARD_GRAD_OP":
        return ShardingStrategy.SHARD_GRAD_OP
    elif sharding_strategy == "NO_SHARD":
        return ShardingStrategy.NO_SHARD
    elif sharding_strategy == "HYBRID_SHARD":
        return ShardingStrategy.HYBRID_SHARD
    elif sharding_strategy == "_HYBRID_SHARD_ZERO2":
        return ShardingStrategy._HYBRID_SHARD_ZERO2
    else:
        raise ValueError(
            f"Invalid sharding_strategy: {sharding_strategy}. Please choose 'FULL_SHARD', 'SHARD_GRAD_OP', 'NO_SHARD', 'HYBRID_SHARD', or '_HYBRID_SHARD_ZERO2'."
        )
