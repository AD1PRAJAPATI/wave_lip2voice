import torch
import torchvision

from .detector.detector import LandmarksDetector
from .detector.video_process import VideoProcess
from .model import LipReadingModel
from .transforms import VideoTransform


class LipReader:
    """Video file in, ranked list of possible phrases out."""

    def __init__(self, checkpoint_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.landmarks_detector = LandmarksDetector()
        self.video_process = VideoProcess(convert_gray=False)
        self.video_transform = VideoTransform()
        self.model = LipReadingModel(checkpoint_path).to(self.device)

    def read(self, video_path, top_k=5):
        video = torchvision.io.read_video(video_path, pts_unit="sec")[0].numpy()
        landmarks = self.landmarks_detector(video)
        mouth_frames = self.video_process(video, landmarks)
        frames = torch.tensor(mouth_frames).permute(0, 3, 1, 2)
        frames = self.video_transform(frames).to(self.device)
        return self.model.predict(frames, top_k=top_k)
