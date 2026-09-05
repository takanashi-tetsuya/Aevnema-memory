from __future__ import annotations

import unittest

import numpy as np

from memory_demo.embeddings import (
    EmbeddingIndex,
    decode_embedding,
    encode_embedding,
)


class EmbeddingTests(unittest.TestCase):
    def test_float32_blob_round_trip(self):
        vector = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64)
        blob = encode_embedding(vector, 4)
        self.assertEqual(len(blob), 4 * 4)
        restored = decode_embedding(blob, 4)
        self.assertEqual(restored.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(restored)), 1.0, places=6)
        np.testing.assert_allclose(restored, [0.6, 0.8, 0.0, 0.0], atol=1e-6)

    def test_blob_length_is_strict(self):
        with self.assertRaises(ValueError):
            decode_embedding(b"\x00" * 8, 4)

    def test_index_upsert_search_and_expand(self):
        index = EmbeddingIndex(4, initial_capacity=1)
        index.upsert(10, [1, 0, 0, 0])
        index.upsert(20, [0, 1, 0, 0])
        self.assertGreaterEqual(index.capacity, 2)
        self.assertEqual(index.search([0.9, 0.1, 0, 0], 1)[0][0], 10)
        index.upsert(10, [0, 1, 0, 0])
        results = index.search([0, 1, 0, 0], 2)
        self.assertEqual({node_id for node_id, _ in results}, {10, 20})
        self.assertTrue(index.remove(20))
        self.assertFalse(index.remove(999))
        self.assertEqual(index.count, 1)
        self.assertEqual(index.search([0, 1, 0, 0], 2)[0][0], 10)
        index.upsert(30, [1, 0, 0, 0])
        self.assertEqual(index.count, 2)


if __name__ == "__main__":
    unittest.main()
