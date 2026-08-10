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


if __name__ == "__main__":
    unittest.main()
