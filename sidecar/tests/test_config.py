from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from oriens.config import load_api_key, load_config


class ConfigTests(unittest.TestCase):
    def test_default_config_keeps_model_details_out_of_business_code(self) -> None:
        config = load_config()
        provider, model = config.provider_for("advice")
        self.assertEqual(provider.region, "cn-beijing")
        self.assertTrue(provider.base_url.startswith("https://"))
        self.assertGreater(model.input_price_per_million_cny, 0)
        self.assertGreater(model.output_price_per_million_cny, 0)

    def test_loads_secret_from_env_file_without_mutating_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text("TEST_ORIENS_SECRET='sensitive-value'\n", encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(
                    load_api_key("TEST_ORIENS_SECRET", env_file), "sensitive-value"
                )
                self.assertNotIn("TEST_ORIENS_SECRET", os.environ)

    def test_missing_secret_is_normal_offline_condition(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.env"
            with patch.dict(os.environ, {}, clear=True):
                self.assertIsNone(load_api_key("TEST_ORIENS_SECRET", missing))

    def test_rag_backend_and_model_are_configuration_only(self) -> None:
        config = load_config()
        self.assertTrue(config.rag.vector_model_id)
        self.assertEqual(config.rag.vector_model_path.name, "bge-m3")
        self.assertEqual(config.rag.vector_dimension, 1024)
        self.assertEqual(config.rag.vector_min_similarity, 0.52)
        self.assertTrue(config.rag.source_path.is_file())
        self.assertTrue(config.rag.eval_path.is_file())

    def test_rag_v2_and_backend_comparison_configs_are_explicit(self) -> None:
        sqlite_config = load_config(Path("config/rag-v2.toml"))
        faiss_config = load_config(Path("config/rag-v2-faiss.toml"))
        self.assertEqual(sqlite_config.rag.pipeline_version, 2)
        self.assertEqual(sqlite_config.rag.vector_backend, "sqlite-vec")
        self.assertEqual(faiss_config.rag.vector_backend, "faiss")
        self.assertEqual(sqlite_config.rag.content_version, faiss_config.rag.content_version)
        self.assertEqual(sqlite_config.rag.raw_paths, faiss_config.rag.raw_paths)
        self.assertEqual(sqlite_config.rag.vector_min_similarity, 0.58)
        self.assertEqual(sqlite_config.rag.vector_max_sequence_length, 256)

    def test_rag_v21_uses_independent_data_corpus_and_indexes(self) -> None:
        sqlite_config = load_config(Path("config/rag-v2.1.toml"))
        faiss_config = load_config(Path("config/rag-v2.1-faiss.toml"))
        self.assertEqual(
            sqlite_config.rag.content_version,
            "rag-v2.1-huiji-data-2026-08-11",
        )
        self.assertEqual(sqlite_config.rag.raw_paths, faiss_config.rag.raw_paths)
        self.assertEqual(len(sqlite_config.rag.raw_paths), 3)
        self.assertEqual(sqlite_config.rag.vector_backend, "sqlite-vec")
        self.assertEqual(faiss_config.rag.vector_backend, "faiss")
        self.assertNotEqual(sqlite_config.rag.index_path, faiss_config.rag.index_path)
        self.assertIn("rag-v2.1", str(sqlite_config.rag.chunks_path))
        self.assertEqual(sqlite_config.rag.vector_device, "cuda")
        self.assertEqual(sqlite_config.rag.vector_build_timeout_seconds, 86400)


if __name__ == "__main__":
    unittest.main()
