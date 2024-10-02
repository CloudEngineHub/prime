from zeroband.models.llama.model import Transformer

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from zeroband.utils.logging import get_logger


def apply_ac_ckpt(model: Transformer):
    """Apply activation checkpointing to the model."""
    logger = get_logger()

    for layer_id, transformer_block in model.layers.named_children():
        transformer_block = checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.layers.register_module(layer_id, transformer_block)

    logger.info("Applied activation checkpointing to the model")
