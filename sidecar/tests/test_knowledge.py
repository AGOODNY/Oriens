from __future__ import annotations

import unittest

from sidecar.tests.test_support import load_test_config as load_config
from oriens.knowledge import LocalItemKnowledgeBase


class KnowledgeTests(unittest.TestCase):
    def test_fixed_stage1_items_have_traceable_sources(self) -> None:
        config = load_config()
        knowledge = LocalItemKnowledgeBase.load(config.app.knowledge_path)
        self.assertEqual(knowledge.known_ids(), (1, 3, 4, 12, 350))
        for collectible_id in knowledge.known_ids():
            item = knowledge.find(collectible_id)
            assert item is not None
            self.assertTrue(item.sources)
            self.assertTrue(all(source.url.startswith("https://") for source in item.sources))


if __name__ == "__main__":
    unittest.main()
