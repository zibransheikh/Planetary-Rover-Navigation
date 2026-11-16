import unittest

from src.config import EnvConfig
from src.environment.rover_env import LEGACY_FEATURES, ONBOARD_FEATURES
from tests.helpers import ROOT


class TestDocsSync(unittest.TestCase):
    def test_every_feature_is_documented(self):
        doc = (ROOT / "docs" / "OBSERVATION_SCHEMA.md").read_text()
        for name in ONBOARD_FEATURES[:6] + LEGACY_FEATURES[5:7]:
            self.assertIn(name, doc)
        for name in ONBOARD_FEATURES[6:]:
            self.assertIn(name, doc)  # patch names listed in the code-names line
        self.assertEqual(len(ONBOARD_FEATURES), 15)
        self.assertEqual(len(LEGACY_FEATURES), 17)


if __name__ == "__main__":
    unittest.main()
