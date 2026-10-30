import math

import torch

from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_conformer import E2E
from espnet.nets.scorers.length_bonus import LengthBonus

from .transforms import TextTransform

BEAM_SIZE = 40
CTC_WEIGHT = 0.1


class LipReadingModel(torch.nn.Module):
    """Conformer encoder-decoder that turns mouth-region frames into text."""

    def __init__(self, checkpoint_path):
        super().__init__()
        self.text = TextTransform()
        self.model = E2E(len(self.text.token_list), "video", ctc_weight=CTC_WEIGHT)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
        self.model.eval()
        self.beam_search = self._build_beam_search()

    def _build_beam_search(self):
        token_list = self.text.token_list
        scorers = self.model.scorers()
        scorers["lm"] = None
        scorers["length_bonus"] = LengthBonus(len(token_list))
        weights = {"decoder": 1.0 - CTC_WEIGHT, "ctc": CTC_WEIGHT, "lm": 0.0, "length_bonus": 0}
        return BatchBeamSearch(
            beam_size=BEAM_SIZE,
            vocab_size=len(token_list),
            weights=weights,
            scorers=scorers,
            sos=self.model.odim - 1,
            eos=self.model.odim - 1,
            token_list=token_list,
            pre_beam_score_key="decoder",
        )

    @torch.no_grad()
    def predict(self, frames, top_k=5):
        """frames: T x 1 x 88 x 88 tensor. Returns up to top_k phrases, best first.

        Each phrase has `text` and `confidence` (0-1). Confidence is the model score
        of each phrase divided by the total of the top_k scores, so the shown
        percentages are relative to each other, not absolute probabilities.
        """
        x = self.model.frontend(frames.unsqueeze(0))
        x = self.model.proj_encoder(x)
        encoded, _ = self.model.encoder(x, None)
        hypotheses = self.beam_search(encoded.squeeze(0))[:top_k]

        best = max(float(h.score) for h in hypotheses)
        weights = [math.exp(float(h.score) - best) for h in hypotheses]
        total = sum(weights)

        phrases = []
        for hypothesis, weight in zip(hypotheses, weights):
            token_ids = [int(i) for i in hypothesis.yseq[1:]]
            phrases.append({"text": self.text.ids_to_text(token_ids), "confidence": weight / total})
        return phrases
