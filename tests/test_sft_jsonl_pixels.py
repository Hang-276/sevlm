"""SFT visual scale follows optional CLI limits and rejects invalid ranges."""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from open_r1.sft_jsonl import _configure_processor_pixels, _configure_sft_training_args
from open_r1.qwen_pixels import qwen_pixel_limits


class SftPixelTests(unittest.TestCase):
    def test_unset_limits_preserve_checkpoint_processor(self):
        image_processor = SimpleNamespace(min_pixels=3136, max_pixels=12845056)
        processor = SimpleNamespace(image_processor=image_processor)
        _configure_processor_pixels(processor, None, None)
        self.assertEqual((image_processor.min_pixels, image_processor.max_pixels),
                         (3136, 12845056))

    def test_configured_limits_replace_both_values(self):
        image_processor = SimpleNamespace(
            min_pixels=3136, max_pixels=12845056,
            size={"shortest_edge": 3136, "longest_edge": 12845056},
        )
        _configure_processor_pixels(SimpleNamespace(image_processor=image_processor),
                                    401408, 602112)
        self.assertEqual((image_processor.min_pixels, image_processor.max_pixels),
                         (401408, 602112))
        self.assertEqual(image_processor.size,
                         {"shortest_edge": 401408, "longest_edge": 602112})

    def test_real_qwen_fast_processor_applies_limits_and_saves_them(self):
        from PIL import Image
        try:
            from transformers import Qwen2VLImageProcessorFast
        except ImportError:
            self.skipTest("installed Transformers has no Qwen fast processor")

        image_processor = Qwen2VLImageProcessorFast()
        _configure_processor_pixels(SimpleNamespace(image_processor=image_processor),
                                    401408, 602112)
        self.assertEqual(qwen_pixel_limits(SimpleNamespace(image_processor=image_processor)),
                         {"min_pixels": 401408, "max_pixels": 602112})
        processed = image_processor(
            images=Image.new("RGB", (320, 240)), return_tensors="pt"
        )
        _, grid_h, grid_w = processed["image_grid_thw"][0].tolist()
        resized_pixels = grid_h * image_processor.patch_size * grid_w * image_processor.patch_size
        self.assertGreaterEqual(resized_pixels, 401408)
        self.assertLessEqual(resized_pixels, 602112)
        with tempfile.TemporaryDirectory() as directory:
            image_processor.save_pretrained(directory)
            saved = json.loads((Path(directory) / "preprocessor_config.json").read_text())
        self.assertEqual(saved["size"],
                         {"shortest_edge": 401408, "longest_edge": 602112})

    def test_real_qwen_slow_processor_uses_the_same_grid(self):
        from PIL import Image
        from transformers import Qwen2VLImageProcessor

        image_processor = Qwen2VLImageProcessor()
        _configure_processor_pixels(SimpleNamespace(image_processor=image_processor),
                                    401408, 602112)
        processed = image_processor(
            images=Image.new("RGB", (320, 240)), return_tensors="pt"
        )
        _, grid_h, grid_w = processed["image_grid_thw"][0].tolist()
        resized_pixels = grid_h * image_processor.patch_size * grid_w * image_processor.patch_size
        self.assertGreaterEqual(resized_pixels, 401408)
        self.assertLessEqual(resized_pixels, 602112)

    def test_invalid_ranges_fail_before_training(self):
        for minimum, maximum in ((0, 602112), (401408, -1), (602112, 401408)):
            with self.subTest(minimum=minimum, maximum=maximum):
                processor = SimpleNamespace(image_processor=SimpleNamespace(
                    min_pixels=3136, max_pixels=12845056))
                with self.assertRaises(ValueError):
                    _configure_processor_pixels(processor, minimum, maximum)

    def test_multimodal_records_are_not_cut_to_trl_default_length(self):
        args = SimpleNamespace(
            packing=False, max_length=1024, remove_unused_columns=True,
            dataset_kwargs={"existing": "kept"},
        )
        _configure_sft_training_args(args)
        self.assertIsNone(args.max_length)
        self.assertFalse(args.remove_unused_columns)
        self.assertEqual(args.dataset_kwargs,
                         {"existing": "kept", "skip_prepare_dataset": True})
        with self.assertRaises(ValueError):
            _configure_sft_training_args(SimpleNamespace(packing=True))


if __name__ == "__main__":
    unittest.main()
