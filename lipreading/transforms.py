import os

import sentencepiece
import torch
import torchvision

HERE = os.path.dirname(os.path.abspath(__file__))
SP_MODEL_PATH = os.path.join(HERE, "unigram5000.model")
DICT_PATH = os.path.join(HERE, "unigram5000_units.txt")


class Scale(torch.nn.Module):
    def forward(self, x):
        return x / 255.0


class VideoTransform:
    """Scales to 0-1, center-crops the mouth region to 88x88, converts to gray and normalizes."""

    def __init__(self):
        self.pipeline = torch.nn.Sequential(
            Scale(),
            torchvision.transforms.CenterCrop(88),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Normalize(0.421, 0.165),
        )

    def __call__(self, frames):
        # frames: T x C x H x W  ->  T x 1 x 88 x 88
        return self.pipeline(frames)


class TextTransform:
    """Converts between token ids and text using the SentencePiece vocabulary."""

    def __init__(self):
        self.spm = sentencepiece.SentencePieceProcessor(model_file=SP_MODEL_PATH)
        units = open(DICT_PATH, encoding="utf8").read().splitlines()
        self.hashmap = {unit.split()[0]: unit.split()[-1] for unit in units}
        # index 0 is the CTC "blank" token, the last one is end-of-sentence
        self.token_list = ["<blank>"] + list(self.hashmap.keys()) + ["<eos>"]

    def ids_to_text(self, token_ids):
        text = "".join(self.token_list[i] for i in token_ids)
        return text.replace("<space>", " ").replace("▁", " ").replace("<eos>", "").strip()
