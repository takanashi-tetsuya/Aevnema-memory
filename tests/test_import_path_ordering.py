from __future__ import annotations

from pathlib import Path
import unittest

from memory_demo.ingestion.ordering import natural_path_sort_key


class ImportPathOrderingTests(unittest.TestCase):
    def test_numbered_flat_corpus_uses_natural_order(self) -> None:
        paths = [
            Path("100_第二部_序章.txt"),
            Path("10_序章.txt"),
            Path("02_学院与势力事实.txt"),
            Path("90_推测与考据.txt"),
            Path("11_Vol1_对策委员会篇.txt"),
        ]

        ordered = sorted(paths, key=natural_path_sort_key)

        self.assertEqual(
            [path.name for path in ordered],
            [
                "02_学院与势力事实.txt",
                "10_序章.txt",
                "11_Vol1_对策委员会篇.txt",
                "90_推测与考据.txt",
                "100_第二部_序章.txt",
            ],
        )


if __name__ == "__main__":
    unittest.main()
