from __future__ import annotations

from threading import Event
import unittest
from unittest.mock import patch

from sidecar.tests.test_support import load_test_config as load_config
from oriens.modeling import ModelRequest, ModelTimeout, QwenOpenAIAdapter


class ModelingTests(unittest.TestCase):
    def test_qwen_timeout_is_sanitized(self) -> None:
        config = load_config()
        provider, model = config.provider_for("advice")
        secret = "test-secret-must-not-appear"
        adapter = QwenOpenAIAdapter(provider, secret)
        with patch("oriens.modeling.request.urlopen", side_effect=TimeoutError):
            with self.assertRaises(ModelTimeout) as raised:
                adapter.complete(
                    model,
                    ModelRequest("输出 JSON", "{}"),
                    Event(),
                )
        self.assertNotIn(secret, str(raised.exception))
        self.assertEqual(str(raised.exception), "模型请求超时")


if __name__ == "__main__":
    unittest.main()
