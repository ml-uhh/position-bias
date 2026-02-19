"""Dataset wrapper for sequence filtering."""

from collections.abc import Iterator

from einops import rearrange
import structlog
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer

logger = structlog.get_logger()


class DatasetWrapper(IterableDataset):  # type: ignore[misc]
    """Dataset wrapper that filters sequences by token length."""

    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        tokenizer: PreTrainedTokenizer,
        seq_length: int,
    ) -> None:
        """
        Initialize the dataset wrapper.

        Store the dataset, tokenizer and sequence length and prefilter the dataset so that only entries with sufficient text length are retained.

        Args:
            dataset: The source iterable dataset.
            tokenizer: Tokenizer to use for length checking.
            seq_length: Required sequence length (tokens).

        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.seq_length = seq_length

        self.dataset = self.dataset.filter(lambda x: len(x["text"]) >= self.seq_length)  # pyright: ignore[reportAttributeAccessIssue]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """
        Iterate over the dataset, yielding only valid-length sequences.

        Yields:
            Dictionary containing tokenized inputs (e.g., input_ids, attention_mask) as tensors.

        """
        if isinstance(self.dataset, IterableDataset):
            iterator = iter(self.dataset)
        else:
            iterator = (self.dataset[i] for i in range(len(self.dataset)))  # pyright: ignore[reportArgumentType]

        while True:
            try:
                sample = next(iterator)
                text = sample["text"]
                tokenized = self.tokenizer(  # pyright: ignore[reportCallIssue]
                    text,
                    padding=False,
                    truncation=True,
                    max_length=self.seq_length,
                    return_tensors="pt",
                )
                if tokenized["input_ids"].shape[-1] == self.seq_length:  # pyright: ignore[reportAttributeAccessIssue]
                    yield {
                        k: rearrange(v, "1 ... -> ...") for k, v in tokenized.items()
                    }

            except StopIteration:
                break
