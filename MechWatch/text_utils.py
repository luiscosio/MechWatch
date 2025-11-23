from __future__ import annotations

from typing import Optional

from transformer_lens import HookedTransformer


def truncate_prompt_to_tokens(
    model: HookedTransformer,
    text: str,
    max_tokens: Optional[int],
) -> str:
    """
    Trim the prompt to the last `max_tokens` tokens (including BOS) to
    bound runtime cost. Returns the original text when `max_tokens` is
    None or the prompt is already short enough.
    """

    if not max_tokens or max_tokens <= 0:
        return text

    tokens = model.to_tokens(text, prepend_bos=True)
    if tokens.shape[-1] <= max_tokens:
        return text

    truncated = tokens[:, -max_tokens:]
    return model.to_string(truncated[0])

