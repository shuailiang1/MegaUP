"""CPU tests for the shape rule and the static kernel configuration helpers."""

from __future__ import annotations

import importlib.util
import unittest

from mega_up import supports_shape, unsupported_shape_reason

HAS_CUTLASS = importlib.util.find_spec("cutlass") is not None


class ShapeRuleTest(unittest.TestCase):
    def test_head_dim_128_models(self) -> None:
        # (heads, UP) pairs with head_dim 128 that a Ulysses group can run.
        for heads in (12, 16, 20, 24, 28, 32, 40, 48, 64):
            for up in (2, 4, 8):
                if heads % up:
                    continue
                hidden = heads * 128
                with self.subTest(hidden=hidden, up=up):
                    self.assertTrue(supports_shape(hidden, up))

    def test_odd_number_of_256_column_tiles(self) -> None:
        for hidden in (1280, 2304, 3328, 4864):
            with self.subTest(hidden=hidden):
                self.assertTrue(supports_shape(hidden, 2))

    def test_rejections_name_the_violated_constraint(self) -> None:
        cases = {
            (4096, 3): "UP must be one of",
            (4096, 16): "UP must be one of",
            (2049, 2): "divisible by UP",
            (1920, 2): "multiple of 256",
            (1536, 8): "head shard",
            (1280, 8): "head shard",
            (0, 2): "positive",
        }
        for (hidden, up), message in cases.items():
            with self.subTest(hidden=hidden, up=up):
                reason = unsupported_shape_reason(hidden, up)
                self.assertIsNotNone(reason)
                self.assertIn(message, reason)
                self.assertFalse(supports_shape(hidden, up))


@unittest.skipUnless(HAS_CUTLASS, "requires nvidia-cutlass-dsl")
class PeerWriteRotationTest(unittest.TestCase):
    def test_each_rank_starts_in_its_own_column_block(self) -> None:
        from mega_up.kernels.peer_write_gemm import PEER_TILE_N, peer_n_rotation

        for hidden, up in ((2048, 8), (3072, 4), (4096, 2), (5120, 8), (2304, 2)):
            n_tiles = -(-hidden // PEER_TILE_N)
            shard = hidden // up
            starts = [peer_n_rotation(rank, up, hidden, PEER_TILE_N) for rank in range(up)]
            with self.subTest(hidden=hidden, up=up):
                self.assertEqual(starts, sorted(starts))
                for rank, start in enumerate(starts):
                    self.assertTrue(0 <= start < n_tiles)
                    # The first visited tile overlaps the rank's own shard.
                    first, last = start * PEER_TILE_N, (start + 1) * PEER_TILE_N
                    self.assertLess(first, (rank + 1) * shard)
                    self.assertGreater(last, rank * shard)


@unittest.skipUnless(HAS_CUTLASS, "requires nvidia-cutlass-dsl")
class PullConfigurationTest(unittest.TestCase):
    def setUp(self) -> None:
        from mega_up.kernels import headshard_pull_gemm

        self.kernel = headshard_pull_gemm

    def test_lane_width_tiles_the_shard(self) -> None:
        select = self.kernel._select_pull_elements_per_lane
        self.assertEqual(select(2048), 16)
        self.assertEqual(select(1280), 8)
        self.assertEqual(select(640), 4)
        for shard in range(128, 8193, 128):
            elements = select(shard)
            with self.subTest(shard=shard):
                self.assertIn(elements, (4, 8, 16))
                self.assertEqual(shard % (32 * elements), 0)

    def test_ticket_size_follows_the_shard_width(self) -> None:
        rows = self.kernel._vector_pull_rows_per_ticket
        self.assertEqual(rows(8, 128), 16)
        self.assertEqual(rows(8, 256), 16)
        self.assertEqual(rows(8, 384), 8)
        self.assertEqual(rows(8, 1280), 8)
        self.assertEqual(rows(8, 1408), 4)
        self.assertEqual(rows(8, 3072), 4)
        self.assertEqual(rows(4, 640), 4)

    def test_column_chunks_bound_the_registers(self) -> None:
        chunks = self.kernel._pull_column_chunks
        limit = self.kernel.PULL_MAX_ELEMENTS_IN_FLIGHT
        self.assertEqual(chunks(3072, 4), 1)
        self.assertEqual(chunks(4096, 4), 2)
        self.assertEqual(chunks(1280, 8), 1)
        for shard in range(128, 16385, 128):
            for rows_in_flight in (4, 8, 16):
                count = chunks(shard, rows_in_flight)
                chunk = shard // count
                with self.subTest(shard=shard, rows=rows_in_flight):
                    self.assertEqual(shard % count, 0)
                    self.assertEqual(chunk % 128, 0)
                    self.assertTrue(rows_in_flight * chunk <= limit or chunk == 128)
        with self.assertRaises(ValueError):
            chunks(200, 4)


if __name__ == "__main__":
    unittest.main()
