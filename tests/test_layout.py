"""CPU tests for the public BSHD layout contract."""

from __future__ import annotations

import unittest

import torch

from mega_up.ops import (
    _bshd_to_destination_major,
    _build_thd_layout,
    _source_major_to_bshd,
    _source_major_to_thd,
    _thd_to_destination_major,
)


def reference_regroup_thd(
    source_major: torch.Tensor,
    cu_seqlens: tuple[int, ...],
) -> torch.Tensor:
    """Slice-by-slice ``[source,T,C]`` to packed ``[UP*T,C]`` reference."""

    up = source_major.shape[0]
    pieces = []
    for sample in range(len(cu_seqlens) - 1):
        start, stop = cu_seqlens[sample], cu_seqlens[sample + 1]
        for source in range(up):
            pieces.append(source_major[source, start:stop])
    return torch.cat(pieces, dim=0)


class THDLayoutTest(unittest.TestCase):
    def test_indices_are_mutually_inverse_permutations(self) -> None:
        layout = _build_thd_layout((0, 5, 5, 12, 13), up=3, device="cpu")

        self.assertEqual(layout.tokens, 13)
        self.assertEqual(layout.num_samples, 4)
        self.assertEqual(layout.cu_seqlens_out, (0, 15, 15, 36, 39))
        total = layout.up * layout.tokens
        identity = torch.arange(total)
        torch.testing.assert_close(
            layout.to_thd_index.sort().values, identity
        )
        torch.testing.assert_close(
            layout.to_thd_index[layout.to_source_major_index], identity
        )
        torch.testing.assert_close(
            layout.to_source_major_index[layout.to_thd_index], identity
        )

    def test_source_major_to_thd_matches_slice_reference(self) -> None:
        up, cu_seqlens, shard_heads, head_dim = 4, (0, 3, 10, 11, 16), 2, 3
        layout = _build_thd_layout(cu_seqlens, up=up, device="cpu")
        source_major = torch.arange(
            up * layout.tokens * shard_heads * head_dim, dtype=torch.float32
        ).view(up, layout.tokens, shard_heads * head_dim)

        actual = _source_major_to_thd(
            source_major,
            layout,
            shard_heads=shard_heads,
            head_dim=head_dim,
        )
        expected = reference_regroup_thd(source_major, cu_seqlens).view(
            up * layout.tokens, shard_heads, head_dim
        )

        torch.testing.assert_close(actual, expected)
        # Each sample is contiguous at UP * cu_seqlens with chunks in rank order.
        for sample in range(layout.num_samples):
            start, stop = cu_seqlens[sample], cu_seqlens[sample + 1]
            length = stop - start
            for source in range(up):
                begin = layout.cu_seqlens_out[sample] + source * length
                torch.testing.assert_close(
                    actual[begin : begin + length].reshape(length, -1),
                    source_major[source, start:stop],
                )

    def test_equal_lengths_reduce_to_bshd(self) -> None:
        up, batch, sequence, shard_heads, head_dim = 3, 4, 5, 2, 2
        cu_seqlens = tuple(sequence * index for index in range(batch + 1))
        layout = _build_thd_layout(cu_seqlens, up=up, device="cpu")
        source_major = torch.randn(up, batch * sequence, shard_heads * head_dim)

        thd = _source_major_to_thd(
            source_major,
            layout,
            shard_heads=shard_heads,
            head_dim=head_dim,
        )
        bshd = _source_major_to_bshd(
            source_major,
            batch=batch,
            sequence=sequence,
            shard_heads=shard_heads,
            head_dim=head_dim,
        )
        torch.testing.assert_close(
            thd, bshd.reshape(batch * up * sequence, shard_heads, head_dim)
        )

        head_parallel = torch.randn(
            batch * up * sequence, shard_heads, head_dim
        )
        torch.testing.assert_close(
            _thd_to_destination_major(head_parallel, layout),
            _bshd_to_destination_major(
                head_parallel.view(batch, up * sequence, shard_heads, head_dim),
                up=up,
                sequence=sequence,
            ),
        )

    def test_thd_round_trip(self) -> None:
        up, cu_seqlens, shard_heads, head_dim = 8, (0, 7, 7, 30, 31, 45), 2, 4
        layout = _build_thd_layout(cu_seqlens, up=up, device="cpu")
        head_parallel = torch.randn(up * layout.tokens, shard_heads, head_dim)

        staged = _thd_to_destination_major(head_parallel, layout)
        restored = _source_major_to_thd(
            staged,
            layout,
            shard_heads=shard_heads,
            head_dim=head_dim,
        )

        self.assertEqual(tuple(staged.shape), (up, layout.tokens, 8))
        torch.testing.assert_close(restored, head_parallel)

    def test_accepts_tensor_cu_seqlens_and_rejects_invalid(self) -> None:
        from_tensor = _build_thd_layout(
            torch.tensor([0, 4, 9], dtype=torch.int32), up=2, device="cpu"
        )
        from_ints = _build_thd_layout([0, 4, 9], up=2, device="cpu")
        self.assertEqual(from_tensor.fingerprint, from_ints.fingerprint)
        torch.testing.assert_close(
            from_tensor.to_thd_index, from_ints.to_thd_index
        )
        self.assertNotEqual(
            from_ints.fingerprint,
            _build_thd_layout([0, 5, 9], up=2, device="cpu").fingerprint,
        )

        for invalid in ((1, 4), (0, 4, 3), (0,), (0, 0), [[0, 4]]):
            with self.assertRaises((ValueError, TypeError)):
                _build_thd_layout(invalid, up=2, device="cpu")
        with self.assertRaises(ValueError):
            _build_thd_layout(
                torch.tensor([0.0, 4.0]), up=2, device="cpu"
            )


class SBHDLayoutTest(unittest.TestCase):
    """Sequence-first data needs no regroup: the slots already hold the result."""

    def test_source_major_receive_is_the_sequence_first_output(self) -> None:
        up, sequence, batch, heads, head_dim = 3, 4, 2, 6, 2
        shard = heads // up
        chunks = [torch.randn(sequence, batch, heads, head_dim) for _ in range(up)]
        for rank in range(up):
            columns = slice(rank * shard, (rank + 1) * shard)
            # Ulysses result on ``rank``: its head shard of every chunk, in rank order.
            reference = torch.cat([chunk[:, :, columns] for chunk in chunks], dim=0)
            # Peer writes land source-major with sequence-major rows (s * B + b).
            slots = torch.stack(
                [
                    chunk[:, :, columns].reshape(sequence * batch, shard * head_dim)
                    for chunk in chunks
                ]
            )
            torch.testing.assert_close(
                slots.view(up * sequence, batch, shard, head_dim), reference
            )

    def test_head_parallel_input_is_destination_major(self) -> None:
        up, sequence, batch, shard_heads, head_dim = 4, 3, 2, 2, 3
        head_parallel = torch.randn(up * sequence, batch, shard_heads, head_dim)
        staged = head_parallel.view(up, sequence * batch, shard_heads * head_dim)
        for destination in range(up):
            torch.testing.assert_close(
                staged[destination].view(sequence, batch, shard_heads, head_dim),
                head_parallel[destination * sequence : (destination + 1) * sequence],
            )


class BSHDLayoutTest(unittest.TestCase):
    def test_source_major_receive_becomes_batch_major_sequence(self) -> None:
        up, batch, sequence, shard_heads, head_dim = 3, 2, 4, 2, 2
        source_major = torch.arange(
            up * batch * sequence * shard_heads * head_dim
        ).view(up, batch * sequence, shard_heads * head_dim)

        actual = _source_major_to_bshd(
            source_major,
            batch=batch,
            sequence=sequence,
            shard_heads=shard_heads,
            head_dim=head_dim,
        )
        expected = (
            source_major.view(
                up, batch, sequence, shard_heads, head_dim
            )
            .permute(1, 0, 2, 3, 4)
            .reshape(batch, up * sequence, shard_heads, head_dim)
        )

        torch.testing.assert_close(actual, expected)
        for batch_index in range(batch):
            for source in range(up):
                torch.testing.assert_close(
                    actual[
                        batch_index,
                        source * sequence : (source + 1) * sequence,
                    ],
                    source_major.view(
                        up, batch, sequence, shard_heads, head_dim
                    )[source, batch_index],
                )

    def test_batch_major_head_input_becomes_destination_major(self) -> None:
        up, batch, sequence, shard_heads, head_dim = 3, 2, 4, 2, 2
        head_parallel = torch.arange(
            batch * up * sequence * shard_heads * head_dim
        ).view(batch, up * sequence, shard_heads, head_dim)

        actual = _bshd_to_destination_major(
            head_parallel,
            up=up,
            sequence=sequence,
        )
        expected = (
            head_parallel.view(
                batch, up, sequence, shard_heads, head_dim
            )
            .permute(1, 0, 2, 3, 4)
            .reshape(
                up,
                batch * sequence,
                shard_heads * head_dim,
            )
        )

        torch.testing.assert_close(actual, expected)

    def test_layout_round_trip_preserves_batch_boundaries(self) -> None:
        up, batch, sequence, shard_heads, head_dim = 4, 3, 5, 2, 4
        head_parallel = torch.randn(
            batch, up * sequence, shard_heads, head_dim
        )

        staged = _bshd_to_destination_major(
            head_parallel,
            up=up,
            sequence=sequence,
        )
        restored = _source_major_to_bshd(
            staged,
            batch=batch,
            sequence=sequence,
            shard_heads=shard_heads,
            head_dim=head_dim,
        )

        torch.testing.assert_close(restored, head_parallel)


if __name__ == "__main__":
    unittest.main()
