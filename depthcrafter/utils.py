from typing import Union, List
import tempfile
import numpy as np
import PIL.Image
import matplotlib.cm as cm
import mediapy
import torch
from decord import VideoReader, cpu
import logging

logger = logging.getLogger(__name__)

dataset_res_dict = {
    "sintel": [448, 1024],
    "scannet": [640, 832],
    "KITTI": [384, 1280],
    "bonn": [512, 640],
    "NYUv2": [448, 640],
}

def read_video_frames(video_path, process_length, target_fps, max_res, dataset="open", return_original=False):
    logger.debug(f"Reading video: {video_path}, process_length={process_length}, target_fps={target_fps}, max_res={max_res}, dataset={dataset}, return_original={return_original}")
    vid = VideoReader(video_path, ctx=cpu(0))
    logger.debug(f"Original video shape: {(len(vid), *vid.get_batch([0]).shape[1:])}")
    original_height, original_width = vid.get_batch([0]).shape[1:3]

    if dataset == "open":
        height = round(original_height / 64) * 64
        width = round(original_width / 64) * 64
        if max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale / 64) * 64
            width = round(original_width * scale / 64) * 64
    else:
        height = dataset_res_dict[dataset][0]
        width = dataset_res_dict[dataset][1]

    # Read frames at scaled resolution
    vid_scaled = VideoReader(video_path, ctx=cpu(0), width=width, height=height)
    fps = vid_scaled.get_avg_fps() if target_fps == -1 else target_fps
    stride = round(vid_scaled.get_avg_fps() / fps)
    stride = max(stride, 1)
    frames_idx = list(range(0, len(vid_scaled), stride))
    logger.debug(f"Downsampled shape: {(len(frames_idx), *vid_scaled.get_batch([0]).shape[1:])}, stride: {stride}")
    
    if process_length != -1 and process_length < len(frames_idx):
        frames_idx = frames_idx[:process_length]
    logger.debug(f"Final processing shape: {(len(frames_idx), *vid_scaled.get_batch([0]).shape[1:])}")

    frames = vid_scaled.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
    logger.debug(f"Scaled frames shape: {frames.shape}")

    if return_original:
        # Read frames at original resolution
        original_frames = vid.get_batch(frames_idx).asnumpy().astype("float32") / 255.0
        logger.debug(f"Original frames shape: {original_frames.shape}")
        return frames, original_frames, fps

    return frames, fps

def save_video(
    video_frames: Union[List[np.ndarray], List[PIL.Image.Image]],
    output_video_path: str = None,
    fps: int = 10,
    crf: int = 18,
) -> str:
    if output_video_path is None:
        output_video_path = tempfile.NamedTemporaryFile(suffix=".mp4").name

    if isinstance(video_frames[0], np.ndarray):
        video_frames = [(frame * 255).astype(np.uint8) for frame in video_frames]
    elif isinstance(video_frames[0], PIL.Image.Image):
        video_frames = [np.array(frame) for frame in video_frames]
    mediapy.write_video(output_video_path, video_frames, fps=fps, crf=crf)
    return output_video_path

class ColorMapper:
    # a color mapper to map depth values to a certain colormap
    def __init__(self, colormap: str = "inferno"):
        self.colormap = torch.tensor(cm.get_cmap(colormap).colors)

    def apply(self, image: torch.Tensor, v_min=None, v_max=None):
        # assert len(image.shape) == 2
        if v_min is None:
            v_min = image.min()
        if v_max is None:
            v_max = image.max()
        image = (image - v_min) / (v_max - v_min)
        image = (image * 255).long()
        image = self.colormap[image]
        return image

def vis_sequence_depth(depths: np.ndarray, v_min=None, v_max=None):
    visualizer = ColorMapper()
    if v_min is None:
        v_min = depths.min()
    if v_max is None:
        v_max = depths.max()
    res = visualizer.apply(torch.tensor(depths), v_min=v_min, v_max=v_max).numpy()
    return res
