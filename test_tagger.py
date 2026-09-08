import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from tagger.main import (
    DEFAULT_THRESHOLD,
    load_labels,
    parse_args,
    scores_to_caption,
    tag_directory,
)


class DummySession:
    def __init__(self, n_tags: int, hits: dict[int, float]):
        self.n_tags = n_tags
        self.hits = hits

    def get_inputs(self):
        return [type("I", (), {"name": "input"})()]

    def get_outputs(self):
        return [type("O", (), {"name": "output"})()]

    def get_providers(self):
        return ["CPUExecutionProvider"]

    def run(self, _output_names, feeds):
        batch = next(iter(feeds.values())).shape[0]
        out = np.zeros((batch, self.n_tags), dtype=np.float32)
        for idx, score in self.hits.items():
            out[:, idx] = score
        return [out]


class TaggerUnitTest(unittest.TestCase):
    def test_parse_cli(self):
        args = parse_args(["/tmp/alice", "-t", "0.5", "--json", "-b", "4"])
        self.assertEqual(args.directory, "/tmp/alice")
        self.assertEqual(args.threshold, 0.5)
        self.assertTrue(args.json)
        self.assertEqual(args.batch_size, 4)
        self.assertEqual(parse_args(["/tmp/alice"]).batch_size, 1)

    def test_default_threshold(self):
        args = parse_args(["/tmp/alice"])
        self.assertEqual(args.threshold, DEFAULT_THRESHOLD)

    def test_scores_to_caption_orders_by_confidence(self):
        names = ["alpha", "beta", "gamma"]
        scores = np.array([0.2, 0.9, 0.4], dtype=np.float32)
        self.assertEqual(scores_to_caption(scores, names, 0.35), "beta, gamma")
        self.assertEqual(scores_to_caption(scores, names, 0.95), "")

    def test_load_shipped_labels(self):
        tags = load_labels()
        self.assertGreater(len(tags), 1000)
        self.assertIn("1girl", tags)

    def test_writes_sidecar_captions(self):
        tags = ["1girl", "solo", "smile"]
        session = DummySession(n_tags=3, hits={0: 0.9, 1: 0.8, 2: 0.1})
        with tempfile.TemporaryDirectory() as raw:
            folder = Path(raw)
            img = Image.new("RGB", (64, 64), color=(12, 34, 56))
            img.save(folder / "0001.png")
            img.save(folder / "0002.jpg")
            (folder / "ignore.txt").write_text("orphan", encoding="utf-8")
            result = tag_directory(
                folder,
                threshold=0.35,
                session=session,
                tag_names=tags,
                warmup=False,
            )
            self.assertEqual(result["processed"], 2)
            self.assertEqual(result["failed"], 0)
            self.assertEqual((folder / "0001.txt").read_text(encoding="utf-8"), "1girl, solo")
            self.assertEqual((folder / "0002.txt").read_text(encoding="utf-8"), "1girl, solo")


if __name__ == "__main__":
    unittest.main()
