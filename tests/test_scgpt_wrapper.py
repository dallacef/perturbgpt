"""Unit tests for the scGPT foundation-model wrapper.

All tests use a tiny mock checkpoint (no real scGPT weights required).
The mock replaces scGPT's TransformerModel, GeneVocab, Preprocessor, and
tokenize_and_pad_batch with lightweight deterministic stand-ins so tests
run fast and without GPU or network access.
"""

from __future__ import annotations

import json
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

D_MODEL = 32
N_BINS = 5
N_HVG = 10
SEED = 0

GENES = [f"GENE{i}" for i in range(20)]
VOCAB_GENES = GENES[:15]


# ------------------------------------------------------------------ mock torch


class _NoGrad:
    """Minimal context manager to replace torch.no_grad."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _MockTensor:
    """Wrap a numpy array to look like a torch tensor for our purposes."""

    def __init__(self, arr):
        self._arr = np.asarray(arr)

    @property
    def shape(self):
        return self._arr.shape

    def __getitem__(self, idx):
        return _MockTensor(self._arr[idx])

    def eq(self, other):
        return _MockTensor(self._arr == other)

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self._arr

    def astype(self, dtype):
        return _MockTensor(self._arr.astype(dtype))


def _make_mock_torch():
    """Create a mock torch module with the minimal API the wrapper needs."""
    mock = MagicMock()
    mock.load = MagicMock(return_value={})
    mock.device = MagicMock(side_effect=lambda d: d)
    mock.no_grad = MagicMock(return_value=_NoGrad())
    mock.tensor = MagicMock(side_effect=lambda arr, dtype=None: _MockTensor(arr))
    mock.long = "long"
    mock.float32 = "float32"
    return mock


# ------------------------------------------------------------------ mock vocab


class MockVocab:
    """Minimal stand-in for scgpt.tokenizer.gene_tokenizer.GeneVocab."""

    def __init__(self, gene_list_or_vocab=None, specials=None, special_first=True,
                 default_token=None, **kwargs):
        gene_list = gene_list_or_vocab if isinstance(gene_list_or_vocab, list) else []
        all_tokens = list(specials or []) + list(gene_list)
        self._token_to_id = {t: i for i, t in enumerate(all_tokens)}
        self._id_to_token = {i: t for i, t in enumerate(all_tokens)}

    def __len__(self):
        return len(self._token_to_id)

    def __contains__(self, token):
        return token in self._token_to_id

    def __getitem__(self, token):
        return self._token_to_id[token]

    def get_itos(self):
        return list(self._id_to_token.values())


# ------------------------------------------------------------------ mock model


class MockEncoder:
    """Stand-in for scGPT's GeneEncoder."""

    def __init__(self, ntoken, d_model):
        self.embedding = MagicMock()
        rng = np.random.default_rng(42)
        weight = rng.standard_normal((ntoken, d_model)).astype(np.float32)
        self.embedding.weight = _MockTensor(weight)


class MockTransformerModel:
    """Minimal stand-in for scgpt.model.TransformerModel."""

    def __init__(self, **kwargs):
        self._d_model = kwargs.get("d_model", D_MODEL)
        ntoken = kwargs.get("ntoken", 100)
        self.encoder = MockEncoder(ntoken, self._d_model)
        self._params = [MagicMock(requires_grad=False)]
        self._training = False

    @property
    def d_model(self):
        return self._d_model

    def eval(self):
        self._training = False
        return self

    def parameters(self):
        return self._params

    def load_state_dict(self, sd):
        pass

    def to(self, device):
        return self

    def __call__(
        self,
        gene_ids,
        values,
        src_key_padding_mask=None,
        batch_labels=None,
        MVC=False,
        ECS=False,
    ):
        """Deterministic forward pass returning [batch, seq_len, d_model]."""
        batch_size = gene_ids.shape[0]
        seq_len = gene_ids.shape[1]
        d = self._d_model

        rng = np.random.default_rng(123)
        base = rng.standard_normal((seq_len, d)).astype(np.float32)
        output_np = np.tile(base[np.newaxis, :, :], (batch_size, 1, 1))
        return _MockTensor(output_np)


# ------------------------------------------------------------- mock preprocessor


def mock_tokenize_and_pad_batch(
    data,
    gene_ids,
    max_len,
    vocab,
    pad_token,
    pad_value,
    append_cls=True,
    include_zero_gene=False,
    **kwargs,
):
    """Stand-in for scgpt.tokenizer.gene_tokenizer.tokenize_and_pad_batch."""
    batch_size = data.shape[0]
    n_genes = len(gene_ids)

    padded_genes = np.full((batch_size, max_len), vocab[pad_token], dtype=np.int64)
    padded_values = np.full((batch_size, max_len), pad_value, dtype=np.float32)

    for i in range(batch_size):
        if append_cls:
            padded_genes[i, 0] = vocab["<cls>"]
            padded_values[i, 0] = 0
        padded_genes[i, 1 : 1 + n_genes] = gene_ids
        padded_values[i, 1 : 1 + n_genes] = data[i]

    return {
        "genes": _MockTensor(padded_genes),
        "values": _MockTensor(padded_values),
    }


class MockPreprocessor:
    """Stand-in for scgpt.preprocess.Preprocessor."""

    def __init__(self, **kwargs):
        self.n_bins = kwargs.get("binning", N_BINS)
        self.n_hvg = kwargs.get("subset_hvg", N_HVG)

    def __call__(self, adata, batch_key=None):
        """Simulate scGPT preprocessing: normalize, log1p, HVG, bin."""
        from scipy import sparse

        X = adata.X
        if sparse.issparse(X):
            X = X.toarray()
        X = np.asarray(X, dtype=np.float32)

        # Normalize total
        totals = X.sum(axis=1, keepdims=True)
        X_normed = np.divide(X, np.maximum(totals, 1e-12)) * 1e4

        # Log1p
        X_log1p = np.log1p(X_normed)

        # HVG: keep top n_hvg genes by variance
        variances = X_log1p.var(axis=0)
        top_idx = np.argsort(-variances)[: self.n_hvg]
        top_idx = np.sort(top_idx)
        adata_out = adata[:, top_idx].copy()
        X_hvg = X_log1p[:, top_idx]

        # Bin
        binned = np.zeros_like(X_hvg, dtype=np.int64)
        for i in range(X_hvg.shape[0]):
            row = X_hvg[i]
            nonzero = row > 0
            if nonzero.sum() == 0:
                continue
            vals = row[nonzero]
            bins = np.quantile(vals, np.linspace(0, 1, self.n_bins - 1))
            binned[i, nonzero] = np.clip(
                np.digitize(vals, bins), 1, self.n_bins - 1
            )

        adata_out.layers["X_binned"] = binned
        return adata_out


def mock_set_seed(seed):
    np.random.seed(seed)


# ------------------------------------------------------------------ fixtures


@pytest.fixture()
def mock_checkpoint_dir(tmp_path):
    """Create a fake checkpoint directory with args.json and vocab.json."""
    args = {
        "embsize": D_MODEL,
        "nhead": 2,
        "d_hid": 64,
        "nlayers": 2,
        "n_layers_cls": 2,
    }
    (tmp_path / "args.json").write_text(json.dumps(args))

    vocab = {g: i for i, g in enumerate(VOCAB_GENES)}
    (tmp_path / "vocab.json").write_text(json.dumps(vocab))

    # Dummy best_model.pt (just needs to exist for the path check)
    (tmp_path / "best_model.pt").write_text("dummy")

    return tmp_path


@pytest.fixture()
def synthetic_adata():
    """5 cells x 20 genes with known expression values."""
    rng = np.random.default_rng(99)
    X = rng.poisson(3, size=(5, 20)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(5)])
    var = pd.DataFrame(index=GENES)
    return AnnData(X, obs=obs, var=var)


@pytest.fixture()
def wrapper(mock_checkpoint_dir):
    """Create a ScGPTWrapper with all scGPT imports mocked.

    The real ``torch`` module is replaced in ``sys.modules`` so the wrapper's
    ``import torch`` statements pick up the mock instead of trying to load
    the real PyTorch (which is slow and may segfault on this machine).
    """
    mock_torch = _make_mock_torch()
    mock_modules = {
        "torch": mock_torch,
        "scgpt": MagicMock(),
        "scgpt.model": MagicMock(TransformerModel=MockTransformerModel),
        "scgpt.preprocess": MagicMock(Preprocessor=MockPreprocessor),
        "scgpt.tokenizer": MagicMock(),
        "scgpt.tokenizer.gene_tokenizer": MagicMock(
            GeneVocab=MockVocab,
            tokenize_and_pad_batch=mock_tokenize_and_pad_batch,
        ),
        "scgpt.utils": MagicMock(set_seed=mock_set_seed),
    }
    with patch.dict("sys.modules", mock_modules):
        from perturbgpt.foundation.scgpt_wrapper import ScGPTWrapper

        yield ScGPTWrapper(
            model_dir=mock_checkpoint_dir,
            device="cpu",
            n_bins=N_BINS,
            n_hvg=N_HVG,
            batch_size=3,
            seed=SEED,
        )


# ------------------------------------------------------------------ tests


class TestScGPTWrapperInit:
    """Test that the wrapper initialises correctly with mocked scGPT."""

    def test_wrapper_loads_successfully(self, wrapper):
        assert wrapper is not None
        assert wrapper.d_model == D_MODEL
        assert wrapper.n_bins == N_BINS
        assert wrapper.n_hvg == N_HVG
        assert wrapper.batch_size == 3

    def test_model_is_frozen(self, wrapper):
        for param in wrapper.model.parameters():
            assert not param.requires_grad

    def test_model_is_in_eval_mode(self, wrapper):
        assert not wrapper.model._training


class TestCellEmbeddings:
    """Test extract_cell_embeddings shape, dtype, and determinism."""

    def test_output_shape(self, wrapper, synthetic_adata):
        emb = wrapper.extract_cell_embeddings(synthetic_adata)
        assert emb.shape == (synthetic_adata.n_obs, D_MODEL)

    def test_output_dtype(self, wrapper, synthetic_adata):
        emb = wrapper.extract_cell_embeddings(synthetic_adata)
        assert emb.dtype == np.float32

    def test_output_finite(self, wrapper, synthetic_adata):
        emb = wrapper.extract_cell_embeddings(synthetic_adata)
        assert np.all(np.isfinite(emb))

    def test_deterministic_same_input(self, wrapper, synthetic_adata):
        """Re-running extraction on the same input must be deterministic."""
        emb1 = wrapper.extract_cell_embeddings(synthetic_adata)
        emb2 = wrapper.extract_cell_embeddings(synthetic_adata)
        np.testing.assert_array_equal(emb1, emb2)

    def test_row_count_matches_input(self, wrapper, synthetic_adata):
        emb = wrapper.extract_cell_embeddings(synthetic_adata)
        assert emb.shape[0] == synthetic_adata.n_obs


class TestGeneEmbeddings:
    """Test get_gene_embedding."""

    def test_gene_embedding_shape(self, wrapper):
        emb = wrapper.get_gene_embedding("GENE0")
        assert emb.shape == (D_MODEL,)

    def test_gene_embedding_dtype(self, wrapper):
        emb = wrapper.get_gene_embedding("GENE0")
        assert emb.dtype == np.float32

    def test_gene_embedding_finite(self, wrapper):
        emb = wrapper.get_gene_embedding("GENE0")
        assert np.all(np.isfinite(emb))

    def test_gene_embedding_deterministic(self, wrapper):
        emb1 = wrapper.get_gene_embedding("GENE0")
        emb2 = wrapper.get_gene_embedding("GENE0")
        np.testing.assert_array_equal(emb1, emb2)

    def test_gene_embedding_unknown_gene_raises(self, wrapper):
        with pytest.raises(KeyError, match="not found in scGPT vocabulary"):
            wrapper.get_gene_embedding("NONEXISTENT_GENE")

    def test_different_genes_different_embeddings(self, wrapper):
        emb_a = wrapper.get_gene_embedding("GENE0")
        emb_b = wrapper.get_gene_embedding("GENE1")
        assert not np.allclose(emb_a, emb_b)


class TestCombinationEmbeddings:
    """Test get_combination_embedding with different methods."""

    def test_sum_combination(self, wrapper):
        genes = ["GENE0", "GENE1", "GENE2"]
        emb = wrapper.get_combination_embedding(genes, method="sum")
        individual = np.stack([wrapper.get_gene_embedding(g) for g in genes])
        expected = individual.sum(axis=0)
        np.testing.assert_allclose(emb, expected, rtol=1e-6)

    def test_mean_combination(self, wrapper):
        genes = ["GENE0", "GENE1"]
        emb = wrapper.get_combination_embedding(genes, method="mean")
        individual = np.stack([wrapper.get_gene_embedding(g) for g in genes])
        expected = individual.mean(axis=0)
        np.testing.assert_allclose(emb, expected, rtol=1e-6)

    def test_max_combination(self, wrapper):
        genes = ["GENE0", "GENE1"]
        emb = wrapper.get_combination_embedding(genes, method="max")
        individual = np.stack([wrapper.get_gene_embedding(g) for g in genes])
        expected = individual.max(axis=0)
        np.testing.assert_allclose(emb, expected, rtol=1e-6)

    def test_default_is_sum(self, wrapper):
        genes = ["GENE0", "GENE1"]
        emb_default = wrapper.get_combination_embedding(genes)
        emb_sum = wrapper.get_combination_embedding(genes, method="sum")
        np.testing.assert_array_equal(emb_default, emb_sum)

    def test_combination_shape(self, wrapper):
        emb = wrapper.get_combination_embedding(["GENE0", "GENE1"])
        assert emb.shape == (D_MODEL,)

    def test_combination_dtype(self, wrapper):
        emb = wrapper.get_combination_embedding(["GENE0", "GENE1"])
        assert emb.dtype == np.float32

    def test_combination_deterministic(self, wrapper):
        genes = ["GENE0", "GENE1", "GENE2"]
        emb1 = wrapper.get_combination_embedding(genes)
        emb2 = wrapper.get_combination_embedding(genes)
        np.testing.assert_array_equal(emb1, emb2)

    def test_combination_invalid_method_raises(self, wrapper):
        with pytest.raises(ValueError, match="Unknown combination method"):
            wrapper.get_combination_embedding(["GENE0"], method="concat")

    def test_combination_unknown_gene_raises(self, wrapper):
        with pytest.raises(KeyError):
            wrapper.get_combination_embedding(["GENE0", "FAKE_GENE"])

    def test_single_gene_combination(self, wrapper):
        emb_single = wrapper.get_combination_embedding(["GENE0"], method="sum")
        emb_direct = wrapper.get_gene_embedding("GENE0")
        np.testing.assert_allclose(emb_single, emb_direct, rtol=1e-6)


class TestImportWithoutScgpt:
    """Test that the module can be imported even without scgpt installed."""

    def test_module_importable(self):
        """The module itself should be importable (lazy imports)."""
        import perturbgpt.foundation.scgpt_wrapper  # noqa: F401

    def test_require_scgpt_raises_import_error(self):
        """_require_scgpt should raise ImportError when scgpt is missing."""
        with patch.dict(
            "sys.modules",
            {
                "scgpt": None,
                "scgpt.model": None,
                "scgpt.preprocess": None,
                "scgpt.tokenizer": None,
                "scgpt.tokenizer.gene_tokenizer": None,
                "scgpt.utils": None,
            },
        ):
            from perturbgpt.foundation.scgpt_wrapper import _require_scgpt

            with pytest.raises(ImportError, match="scgpt"):
                _require_scgpt()
