"""Import-safety tests for the optional Geneformer wrapper."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest


class TestImportSafety:
    def test_module_importable_without_transformers(self):
        import perturbgpt.foundation.geneformer_wrapper  # noqa: F401

    def test_require_geneformer_raises_helpful_error(self):
        with patch.dict(
            sys.modules,
            {"transformers": None, "transformers.AutoModel": None},
        ):
            from perturbgpt.foundation.geneformer_wrapper import _require_geneformer

            with pytest.raises(ImportError, match="Geneformer"):
                _require_geneformer()


class TestMissingArtifacts:
    def test_missing_dictionary_is_reported(self, tmp_path):
        """Without a dictionary and without the Hub, the error is actionable."""
        import perturbgpt.foundation.geneformer_wrapper as gw

        class FakeTensor:  # minimal stand-in so construction fails at the dict step
            pass

        wrapper = gw.GeneformerWrapper.__new__(gw.GeneformerWrapper)
        wrapper.model_dir = tmp_path
        with patch.dict(sys.modules, {"huggingface_hub": None}):
            with pytest.raises(FileNotFoundError, match="gene dictionary"):
                wrapper._load_token_dictionary(None)
