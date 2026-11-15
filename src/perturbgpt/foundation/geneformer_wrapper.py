"""Optional Geneformer-based embedding extractor (experimental).

Provided for the documented scGPT-vs-Geneformer comparison.  Geneformer is
**not** a dependency of the core package: install the extra
(``pip install -e ".[geneformer]"``) and download a checkpoint + gene
dictionary from Hugging Face (``ctheodoris/Geneformer``).

How this differs from :class:`~perturbgpt.foundation.scgpt_wrapper.ScGPTWrapper`
----------------------------------------------------------------------------
* Geneformer consumes **ranked** gene tokens (VOM: highest-expressed genes
  first) in a fixed-length context (2048), not binned expression values.
* It has no ``<cls>`` token; a cell embedding is the mean of the final hidden
  states over the real (non-padded) tokens.
* A gene embedding is the row of the model's word-embedding matrix for that
  gene's token id.

Caveats
-------
This wrapper is intentionally minimal and marked experimental: it mirrors the
ScGPTWrapper API so the two can be swapped in a comparison notebook, but the
headline results in this repository are produced with scGPT.  Tokenisation
details (cell normalisation, ENSEMBL-vs-symbol mapping) follow Geneformer's
documentation only approximately.

All heavy imports are deferred, so importing this module never requires
``transformers`` or a checkpoint; instantiating the wrapper without them raises
a clear ImportError/FileNotFoundError.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_DIR = "ctheodoris/Geneformer"
DEFAULT_TOKEN_DICTIONARY = "gene_dictionary_gc30M.pkl"
DEFAULT_MAX_LEN = 2048
DEFAULT_BATCH_SIZE = 32

SPECIAL_TOKENS = {"<pad>", "<mask>", "<cls>"}


def _require_geneformer() -> None:
    """Import ``transformers`` (Geneformer's runtime), with a helpful error."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModel  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'transformers' package is required for GeneformerWrapper. "
            "Install with: pip install -e \".[geneformer]\" "
            "(or pip install transformers), then download a checkpoint from "
            "https://huggingface.co/ctheodoris/Geneformer."
        ) from exc


class GeneformerWrapper:
    """Frozen Geneformer model for embedding extraction.

    Parameters
    ----------
    model_dir : str or Path
        Local directory or Hugging Face model id (e.g.
        ``"ctheodoris/Geneformer"``).
    token_dictionary_path : str or Path, optional
        Pickle holding ``{gene_symbol: token_id}`` (Geneformer's
        ``gene_dictionary_gc30M.pkl``).  When omitted, the file is looked up
        inside *model_dir* and, failing that, downloaded from the Hub with
        ``huggingface_hub`` if available.
    device : str
        ``"cpu"`` or ``"cuda"``.
    max_len : int
        Number of ranked genes per cell (Geneformer's context length).
    batch_size : int
        Cells per forward pass.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = DEFAULT_MODEL_DIR,
        token_dictionary_path: Optional[Union[str, Path]] = None,
        device: str = "cpu",
        max_len: int = DEFAULT_MAX_LEN,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        _require_geneformer()

        import torch
        from transformers import AutoModel

        self.device = torch.device(device)
        self.max_len = int(max_len)
        self.batch_size = int(batch_size)
        self.model_dir = model_dir

        logger.info("Loading Geneformer from %s", model_dir)
        self.model = AutoModel.from_pretrained(str(model_dir))
        self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.token_dictionary = self._load_token_dictionary(token_dictionary_path)
        self.d_model = int(self.model.config.hidden_size)
        self.pad_token_id = self.token_dictionary.get("<pad>", 0)

    # ------------------------------------------------------------ tokenising

    def _load_token_dictionary(
        self, path: Optional[Union[str, Path]]
    ) -> dict[str, int]:
        candidate: Optional[Path] = Path(path) if path is not None else None
        if candidate is None:
            local = Path(self.model_dir) / DEFAULT_TOKEN_DICTIONARY
            if local.exists():
                candidate = local

        if candidate is None or not candidate.exists():
            try:  # pragma: no cover - requires network
                from huggingface_hub import hf_hub_download

                candidate = Path(
                    hf_hub_download(
                        repo_id=str(self.model_dir),
                        filename=DEFAULT_TOKEN_DICTIONARY,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - surfaced with context
                raise FileNotFoundError(
                    "Could not locate Geneformer's gene dictionary "
                    f"({DEFAULT_TOKEN_DICTIONARY}). Pass "
                    "token_dictionary_path=... explicitly."
                ) from exc

        with candidate.open("rb") as fh:
            dictionary = pickle.load(fh)
        if not isinstance(dictionary, dict):
            raise ValueError(f"{candidate} is not a {{gene: id}} dictionary.")
        return {str(k): int(v) for k, v in dictionary.items()}

    def _tokenize_cell(self, values: np.ndarray, var_names: Sequence[str]) -> list[int]:
        """Return token ids for the top-*max_len* genes by expression."""
        ranked = np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")
        tokens: list[int] = []
        for idx in ranked:
            if values[idx] <= 0:
                break
            symbol = str(var_names[idx])
            token = self.token_dictionary.get(symbol)
            if token is not None:
                tokens.append(token)
            if len(tokens) >= self.max_len:
                break
        return tokens

    # ------------------------------------------------------ cell embeddings

    def extract_cell_embeddings(self, adata: ad.AnnData) -> np.ndarray:
        """Mean-pooled Geneformer embedding for every cell in *adata*.

        Returns
        -------
        np.ndarray
            Shape ``(n_cells, d_model)``, dtype ``float32``.
        """
        import torch

        X = adata.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)
        var_names = list(adata.var_names.astype(str))

        n_cells = X.shape[0]
        embeddings = np.zeros((n_cells, self.d_model), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n_cells, self.batch_size):
                end = min(start + self.batch_size, n_cells)
                batch_tokens = [
                    self._tokenize_cell(X[i], var_names) for i in range(start, end)
                ]
                width = max((len(t) for t in batch_tokens), default=1)
                input_ids = torch.full(
                    (len(batch_tokens), width), self.pad_token_id, dtype=torch.long
                )
                attention_mask = torch.zeros_like(input_ids)
                for row, tokens in enumerate(batch_tokens):
                    if not tokens:
                        continue
                    input_ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
                    attention_mask[row, : len(tokens)] = 1

                output = self.model(
                    input_ids=input_ids.to(self.device),
                    attention_mask=attention_mask.to(self.device),
                )
                hidden = output.last_hidden_state  # [B, T, d_model]
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                embeddings[start:end] = pooled.cpu().numpy().astype(np.float32)

        return embeddings

    # ------------------------------------------------------ gene embeddings

    def get_gene_embedding(self, gene_symbol: str) -> np.ndarray:
        """Return Geneformer's input embedding for a single gene symbol."""
        if gene_symbol not in self.token_dictionary:
            raise KeyError(
                f"Gene symbol {gene_symbol!r} not found in the Geneformer "
                f"token dictionary ({len(self.token_dictionary)} tokens)."
            )
        token_id = self.token_dictionary[gene_symbol]
        import torch

        with torch.no_grad():
            emb = self.model.get_input_embeddings().weight[token_id]
        return emb.detach().cpu().numpy().astype(np.float32)

    def get_combination_embedding(
        self,
        gene_symbols: Sequence[str],
        method: str = "sum",
    ) -> np.ndarray:
        """Combine per-gene embeddings (``sum``, ``mean`` or ``max``)."""
        if method not in ("sum", "mean", "max"):
            raise ValueError(
                f"Unknown combination method {method!r}. "
                "Expected one of: 'sum', 'mean', 'max'."
            )
        embs = np.stack([self.get_gene_embedding(g) for g in gene_symbols])
        if method == "sum":
            return embs.sum(axis=0).astype(np.float32)
        if method == "mean":
            return embs.mean(axis=0).astype(np.float32)
        return embs.max(axis=0).astype(np.float32)
