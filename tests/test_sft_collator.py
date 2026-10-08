import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import open_r1.sft_jsonl as sft


CHAT = """{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' }}{% if m['content'] is string %}{{ m['content'] }}{% else %}{% for item in m['content'] %}{% if item['type'] == 'image' %}{{ '<|vision_start|><|image_pad|><|vision_end|>' }}{% else %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}{{ '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"""


def make_processor(padding_side="right"):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Qwen2TokenizerFast, Qwen2VLImageProcessor, Qwen2_5_VLProcessor, Qwen2VLVideoProcessor

    tokens = ['<unk>', '<|im_start|>', '<|im_end|>', '<|endoftext|>', '<|vision_start|>',
              '<|vision_end|>', '<|image_pad|>', '<|video_pad|>', 'user', 'assistant',
              'Describe', 'red', 'blue', 'short', 'long', 'response']
    tokenizer = Tokenizer(models.WordLevel({token: i for i, token in enumerate(tokens)}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    fast = Qwen2TokenizerFast(tokenizer_object=tokenizer, unk_token='<unk>',
                             eos_token='<|im_end|>', pad_token='<|im_end|>', padding_side=padding_side)
    fast.add_special_tokens({'additional_special_tokens': tokens[:8]})
    return Qwen2_5_VLProcessor(image_processor=Qwen2VLImageProcessor(min_pixels=784, max_pixels=3136),
                              tokenizer=fast, video_processor=Qwen2VLVideoProcessor(), chat_template=CHAT)


@pytest.mark.parametrize("padding_side", ["left", "right"])
def test_real_qwen_collator_mixed_image_counts_eos_and_prompt_mask(monkeypatch, tmp_path, padding_side):
    processor = make_processor(padding_side)
    monkeypatch.setattr(sft, "processor", processor)
    paths = []
    for index, color in enumerate(("red", "blue")):
        path = tmp_path / f"{color}.png"
        Image.new("RGB", (28 + 28 * index, 28), color).save(path)
        paths.append(str(path))
    examples = [dict(problem="Describe", completion="red response", image=paths),
                dict(problem="Describe", completion="blue short long response"),
                dict(problem="Describe", completion="blue response", image=paths[1:])]
    batch = sft.collate_fn(examples)
    assert batch["image_grid_thw"].shape[0] == 3
    tokenizer = processor.tokenizer
    for index, example in enumerate(examples):
        labels = batch["labels"][index]
        assert labels[batch["attention_mask"][index] == 0].eq(-100).all()
        expected = tokenizer(example["completion"], add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
        assert labels[labels != -100].tolist() == expected
    assert batch["labels"][batch["input_ids"] == processor.image_token_id].eq(-100).all()


def test_real_qwen_collator_text_only_and_empty_completion(monkeypatch):
    processor = make_processor()
    monkeypatch.setattr(sft, "processor", processor)
    batch = sft.collate_fn([dict(problem="Describe", completion="")])
    assert "pixel_values" not in batch
    labels = batch["labels"][0]
    assert labels[labels != -100].tolist() == [processor.tokenizer.eos_token_id]


def test_inconsistent_generation_template_rejected_before_supervising_user_tokens(monkeypatch):
    processor = make_processor()
    processor.chat_template = CHAT + "{% if add_generation_prompt %} response{% endif %}"
    monkeypatch.setattr(sft, "processor", processor)
    with pytest.raises(ValueError, match="prompt prefix"):
        sft.collate_fn([dict(problem="Describe", completion="red response")])
