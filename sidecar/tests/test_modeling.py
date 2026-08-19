from __future__ import annotations

from threading import Event
import json
import unittest
from unittest.mock import patch

from sidecar.tests.test_support import load_test_config as load_config
from oriens.modeling import ModelImage, ModelRequest, ModelTimeout, QwenOpenAIAdapter


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps({
            "choices": [{"message": {"content": "{}"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }).encode("utf-8")


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

    def test_multimodal_request_preserves_plain_text_compatibility(self) -> None:
        config = load_config()
        provider, text_model = config.provider_for("advice")
        _provider, vision_model = config.provider_for("vision")
        adapter = QwenOpenAIAdapter(provider, "test-secret")
        outgoing = []

        def open_request(value, timeout):
            outgoing.append((value, timeout))
            return _Response()

        with patch("oriens.modeling.request.urlopen", side_effect=open_request):
            adapter.complete(text_model, ModelRequest("system", "plain"), Event())
            adapter.complete(
                vision_model,
                ModelRequest(
                    "system", "visual", images=(ModelImage("image/jpeg", b"jpeg"),)
                ),
                Event(),
            )
        plain = json.loads(outgoing[0][0].data)
        visual = json.loads(outgoing[1][0].data)
        self.assertEqual(plain["messages"][1]["content"], "plain")
        self.assertEqual(plain["model"], text_model.model_id)
        self.assertIn("response_format", plain)
        self.assertIsInstance(visual["messages"][1]["content"], list)
        self.assertEqual(visual["model"], vision_model.model_id)
        self.assertNotIn("response_format", visual)
        self.assertTrue(
            visual["messages"][1]["content"][1]["image_url"]["url"].startswith(
                "data:image/jpeg;base64,"
            )
        )


if __name__ == "__main__":
    unittest.main()
