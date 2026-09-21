"""Frozen scGPT wrapper for per-cell and per-gene embedding extraction.

Loads a pretrained scGPT checkpoint and exposes:

* extract_cell_embeddings -- per-cell embeddings for an AnnData object
* get_gene_embedding / get_combination_embedding -- per-gene embedding
  lookup with combinatorial combination (default: summation)

The wrapper runs scGPT's own preprocessing/tokenization pipeline
(gene-vocabulary matching, normalize_total -> log1p -> HVG selection ->
expression binning), which is separate from the baseline HVG/normalisation
pipeline in perturbgpt.data.preprocessing.  Do not reuse the baseline
preprocessing for this.

Checkpoint layout (see the official scGPT repo for download instructions):
    model_dir/
        args.json       -- model architecture hyper-parameters
        vocab.json      -- gene vocabulary (gene symbol -> token ID)
        best_model.pt   -- pretrained state dict

All scGPT imports are deferred so this module can be imported (and tested)
even when the scgpt package is not installed.  Instantiating
ScGPTWrapper without scgpt raises a clear ImportError.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional, Sequence, Union

import anndata as ad
import numpy as np

logger = logging.getLogger(__name__)

SPECIAL_TOKENS = ["<pad>", "<cls>", "<eoc>"]
PAD_TOKEN = "<pad>"
PAD_VALUE = -2
MASK_VALUE = -1
DEFAULT_N_BINS = 51
DEFAULT_N_HVG = 1200
DEFAULT_BATCH_SIZE = 64


def _require_scgpt():
    """Import scGPT sub-modules, raising a helpful error if unavailable."""
    try:
        from scgpt.model import TransformerModel  # noqa: F401
        from scgpt.preprocess import Preprocessor  # noqa: F401
        from scgpt.tokenizer.gene_tokenizer import (  # noqa: F401
            GeneVocab,
            tokenize_and_pad_batch,
        )
        from scgpt.utils import set_seed  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The 'scgpt' package is required for ScGPTWrapper. "
            "Install it with: pip install scgpt \"flash-attn<1.0.5\" "
            "(see https://github.com/bowang-lab/scGPT for full instructions)."
        ) from exc


class ScGPTWrapper:
    """Frozen scGPT model for embedding extraction.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the scGPT checkpoint files
        (``args.json``, ``vocab.json``, ``best_model.pt``).
    device : str
        PyTorch device string, e.g. ``"cpu"`` or ``"cuda"``.
    n_bins : int
        Number of expression bins for scGPT's discretization.
    n_hvg : int
        Number of highly-variable genes to keep (scGPT's own HVG step).
    batch_size : int
        Batch size for the cell-embedding forward pass.
    seed : int
        Random seed for reproducibility.

    Attributes
    ----------
    model : TransformerModel
        The frozen scGPT transformer model.
    vocab : GeneVocab
        Gene vocabulary mapping gene symbols to token IDs.
    d_model : int
        Embedding dimensionality of the model.
    """

    def __init__(
        self,
        model_dir: Union[str, Path],
        device: str = "cpu",
        n_bins: int = DEFAULT_N_BINS,
        n_hvg: int = DEFAULT_N_HVG,
        batch_size: int = DEFAULT_BATCH_SIZE,
        seed: int = 42,
    ) -> None:
        _require_scgpt()

        import torch
        from scgpt.model import TransformerModel
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from scgpt.utils import set_seed, load_pretrained


        set_seed(seed)

        model_dir = Path(model_dir)
        if not model_dir.is_dir():
            raise FileNotFoundError(
                f"scGPT checkpoint directory not found: {model_dir}"
            )

        # Load model architecture config
        args_path = model_dir / "args.json"
        with args_path.open() as fh:
            model_args = json.load(fh)
        logger.info("Loaded model args from %s", args_path)

        # Load gene vocabulary
        vocab_path = model_dir / "vocab.json"
        with vocab_path.open() as fh:
            vocab_dict = json.load(fh)
        gene_list = list(vocab_dict.keys())
        self.vocab = GeneVocab(
            gene_list_or_vocab=gene_list,
            specials=SPECIAL_TOKENS,
            special_first=True,
        )
        logger.info("Loaded vocabulary: %d tokens", len(self.vocab))

        # Build model
        self.model = TransformerModel(
            ntoken=len(self.vocab),
            d_model=model_args.get("embsize", 512),
            nhead=model_args.get("nheads", 8),
            d_hid=model_args.get("d_hid", 512),
            nlayers=model_args.get("nlayers", 12),
            nlayers_cls=model_args.get("n_layers_cls", 3),
            # n_cls=1,
            vocab=self.vocab,
            dropout=model_args.get("dropout", 0.0),
            pad_token=PAD_TOKEN,
            pad_value=PAD_VALUE,
            do_mvc=model_args.get("MVC", False),
            do_dab=False,
            use_batch_labels=False,
            domain_spec_batchnorm=False,
            input_emb_style=model_args.get("input_emb_style", 'continuous'),
            n_input_bins=model_args.get("n_bins", 51),
            cell_emb_style="cls",
            # mvc_decoder_style="inner product",
            ecs_threshold=0.0,
            explicit_zero_prob=False,
            use_fast_transformer=True,
            # use_fast_transformer=False,
            pre_norm=False,
        )

        # Load pretrained weights
        ckpt_path = model_dir / "best_model.pt"
        # state_dict = torch.load(ckpt_path, map_location="cpu")
        # self.model.load_state_dict(state_dict)
        load_pretrained(self.model, torch.load(ckpt_path, map_location='cpu'), verbose=False)
        logger.info("Loaded checkpoint from %s", ckpt_path)

        # Freeze
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        self.device = torch.device(device)
        self.model = self.model.to(self.device)

        self.d_model = self.model.d_model
        self.n_bins = n_bins
        self.n_hvg = n_hvg
        self.batch_size = batch_size

    # --------------------------------------------------------- preprocessing

    def _preprocess_for_scgpt(self, adata: ad.AnnData) -> ad.AnnData:
        """Run scGPT's own preprocessing pipeline on *adata*.

        Steps (all performed by scGPT's ``Preprocessor``):

        1. **Normalise total** to 1e4 counts per cell.
        2. **log1p** transform.
        3. **HVG selection** (Seurat v3, scGPT's own scanpy-based step).
        4. **Binning** into ``n_bins`` discrete expression categories.

        This is deliberately *separate* from the baseline
        ``perturbgpt.data.preprocessing`` pipeline.

        Returns a new AnnData with ``.layers["X_binned"]`` containing the
        binned expression matrix.
        """
        from scgpt.preprocess import Preprocessor

        preprocessor = Preprocessor(
            use_key=None,
            filter_gene_by_counts=False,
            filter_cell_by_counts=False,
            normalize_total=1e4,
            result_normed_key="X_normed",
            log1p=True,
            result_log1p_key="X_log1p",
            subset_hvg=self.n_hvg,
            hvg_use_key=None,
            hvg_flavor="seurat_v3",
            binning=self.n_bins,
            result_binned_key="X_binned",
        )
        adata_pp = adata.copy()
        preprocessor(adata_pp)
        return adata_pp

    def _match_vocabulary(
        self, adata: ad.AnnData
    ) -> tuple[np.ndarray, np.ndarray]:
        """Map ``adata.var_names`` to scGPT vocabulary token IDs.

        Returns
        -------
        gene_ids : np.ndarray [n_matched_genes]
            Token IDs for genes found in the vocabulary.
        matched_indices : np.ndarray [n_matched_genes]
            Column indices in ``adata`` of the matched genes (in the same
            order as ``gene_ids``).
        """
        vocab = self.vocab
        var_names = list(adata.var_names)
        # var_names = adata.var['gene_name']
        matched = []
        indices = []
        for idx, gene in enumerate(var_names):
            if gene in vocab:
                matched.append(vocab[gene])
                indices.append(idx)
        if not matched:
            raise ValueError(
                "No genes from the input AnnData matched the scGPT vocabulary. "
                "Check that var_names uses gene symbols (not Ensembl IDs)."
            )
        logger.info(
            "Gene vocabulary match: %d / %d genes matched",
            len(matched),
            len(var_names),
        )
        return np.array(matched, dtype=np.int64), np.array(indices, dtype=np.int64)
        return np.array(matched, dtype=np.int64), np.array(indices, dtype=np.int64)

    # ------------------------------------------------------ cell embeddings

    def extract_cell_embeddings(self, adata: ad.AnnData) -> np.ndarray:
        """Extract a per-cell embedding for every cell in *adata*.

        The input AnnData is preprocessed with scGPT's own pipeline
        (normalise -> log1p -> HVG -> binning -> tokenize), then passed
        through the frozen transformer.  The ``<cls>`` token's output
        hidden state is returned as the cell embedding.

        Parameters
        ----------
        adata : AnnData
            Input dataset.  ``var_names`` must contain gene symbols that
            overlap with the scGPT vocabulary.

        Returns
        -------
        np.ndarray
            Shape ``(n_cells, d_model)``, dtype ``float32``.
            Row *i* corresponds to ``adata.obs_names[i]``.
        """
        import torch
        from scgpt.tokenizer.gene_tokenizer import tokenize_and_pad_batch

        # 1. scGPT preprocessing
        adata_pp = self._preprocess_for_scgpt(adata)

        # 2. Gene vocabulary matching
        gene_ids, matched_indices = self._match_vocabulary(adata_pp)

        # 3. Get the binned expression matrix (matched genes only)
        X_binned = adata_pp.layers["X_binned"]
        if hasattr(X_binned, "toarray"):
            X_binned = X_binned.toarray()
        X_binned = np.asarray(X_binned, dtype=np.float32)
        X_matched = X_binned[:, matched_indices]

        n_cells = X_matched.shape[0]
        max_len = len(gene_ids) + 1  # +1 for <cls>

        embeddings = np.zeros((n_cells, self.d_model), dtype=np.float32)

        with torch.no_grad():
            for start in range(0, n_cells, self.batch_size):
                end = min(start + self.batch_size, n_cells)
                batch_data = X_matched[start:end]

                tokenized = tokenize_and_pad_batch(
                    batch_data,
                    gene_ids,
                    max_len=max_len,
                    vocab=self.vocab,
                    pad_token=PAD_TOKEN,
                    pad_value=PAD_VALUE,
                    append_cls=True,
                    include_zero_gene=True,
                )

                input_gene_ids = tokenized["genes"].to(self.device)
                input_values = tokenized["values"].to(self.device)
                src_key_padding_mask = input_gene_ids.eq(
                    self.vocab[PAD_TOKEN]
                )

                output = self.model(
                    input_gene_ids,
                    input_values,
                    src_key_padding_mask=src_key_padding_mask,
                    batch_labels=None,
                    MVC=False,
                    ECS=False,
                )

                # cell_emb_style == "cls" -> the CLS token is the cell emb
                cell_emb = output["cell_emb"]  # [batch, d_model]
                embeddings[start:end] = cell_emb.cpu().numpy().astype(np.float32)

        return embeddings

    # ------------------------------------------------------ gene embeddings

    def get_gene_embedding(self, gene_symbol: str) -> np.ndarray:
        """Return scGPT's embedding vector for a single gene symbol.

        The embedding is the row of the model's gene encoder
        (``nn.Embedding``) corresponding to the gene's token ID.

        Parameters
        ----------
        gene_symbol : str
            Gene symbol, e.g. ``"KLF1"``.

        Returns
        -------
        np.ndarray
            Shape ``(d_model,)``, dtype ``float32``.

        Raises
        ------
        KeyError
            If *gene_symbol* is not in the scGPT vocabulary.
        """
        if gene_symbol not in self.vocab:
            raise KeyError(
                f"Gene symbol {gene_symbol!r} not found in scGPT vocabulary. "
                f"Vocabulary contains {len(self.vocab)} tokens."
            )
        token_id = self.vocab[gene_symbol]
        import torch

        with torch.no_grad():
            emb = self.model.encoder.embedding.weight[token_id]
        return emb.cpu().numpy().astype(np.float32)

    def get_combination_embedding(
        self,
        gene_symbols: Sequence[str],
        method: str = "sum",
    ) -> np.ndarray:
        """Combine per-gene embeddings for a combinatorial perturbation.

        Parameters
        ----------
        gene_symbols : Sequence[str]
            Gene symbols comprising the perturbation, e.g.
            ``["KLF1", "AHR"]``.
        method : str
            How to combine the individual gene embeddings.  One of:

            * ``"sum"`` (default) -- element-wise summation.
            * ``"mean"`` -- element-wise average.
            * ``"max"`` -- element-wise maximum.

            The parameter exists so the combination strategy can be swapped
            later without changing call sites.

        Returns
        -------
        np.ndarray
            Shape ``(d_model,)``, dtype ``float32``.

        Raises
        ------
        ValueError
            If *method* is not one of ``"sum"``, ``"mean"``, ``"max"``.
        KeyError
            If any gene symbol is not in the scGPT vocabulary.
        """
        if method not in ("sum", "mean", "max"):
            raise ValueError(
                f"Unknown combination method {method!r}. "
                f"Expected one of: 'sum', 'mean', 'max'."
            )
        embs = np.stack([self.get_gene_embedding(g) for g in gene_symbols])
        if method == "sum":
            combined = embs.sum(axis=0)
        elif method == "mean":
            combined = embs.mean(axis=0)
        else:  # max
            combined = embs.max(axis=0)
        return combined.astype(np.float32)