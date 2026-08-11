from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dtw_checkpoint_transfer import (  # noqa: E402
    enforce_contiguous_boundaries,
    mapping_from_path,
    temporal_smooth,
    transfer_boundary_two_sided,
    validate_output_segments,
)


class DtwCheckpointTransferTests(unittest.TestCase):
    def test_temporal_smoothing_preserves_shape_and_unit_norm(self) -> None:
        features = np.arange(40, dtype=np.float32).reshape(10, 4) + 1.0
        smoothed = temporal_smooth(features, 3)
        self.assertEqual(smoothed.shape, features.shape)
        np.testing.assert_allclose(np.linalg.norm(smoothed, axis=1), 1.0, atol=1e-6)

    def test_two_sided_boundary_uses_before_and_after_votes(self) -> None:
        path = [(index, index * 2) for index in range(10)]
        mapping = mapping_from_path(path)
        boundary, diagnostic = transfer_boundary_two_sided(
            reference_boundary=5,
            mapping=mapping,
            reference_frames=10,
            target_frames=20,
            side_window=2,
        )
        self.assertEqual(boundary, 9)
        self.assertEqual(diagnostic["left_target_votes"], [6.0, 8.0])
        self.assertEqual(diagnostic["right_target_votes"], [10.0, 12.0])
        self.assertEqual(diagnostic["method"], "two_sided_median")

    def test_continuity_projection_removes_duplicate_and_extreme_boundaries(self) -> None:
        boundaries = enforce_contiguous_boundaries([10, 10, 220], target_frames=30, min_segment_frames=1)
        self.assertEqual(boundaries, [0, 10, 11, 29, 30])
        self.assertTrue(all(left < right for left, right in zip(boundaries, boundaries[1:])))

    def test_segment_validation_detects_complete_gap_free_coverage(self) -> None:
        boundaries = [0, 3, 7, 10]
        segments = [
            {
                "segment_id": index,
                "start_frame": start,
                "end_frame_exclusive": end,
                "end_frame_inclusive": end - 1,
                "num_frames": end - start,
            }
            for index, (start, end) in enumerate(zip(boundaries, boundaries[1:]))
        ]
        validation = validate_output_segments(segments, target_frames=10)
        self.assertTrue(validation["valid"])
        self.assertTrue(validation["coverage_is_contiguous"])
        self.assertTrue(validation["coverage_is_complete"])
        self.assertEqual(validation["num_labeled_frames"], 10)


if __name__ == "__main__":
    unittest.main()
