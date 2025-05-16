import logging
import gc
import os
import time
import numpy as np
import cv2
import spaces
import gradio as gr
import torch
from diffusers.training_utils import set_seed
from depthcrafter.depth_crafter_ppl import DepthCrafterPipeline
from depthcrafter.unet import DiffusersUNetSpatioTemporalConditionModelDepthCrafter
from depthcrafter.utils import read_video_frames, vis_sequence_depth, save_video
from moviepy.editor import VideoFileClip, AudioFileClip

# Configure logging to console and file
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('depthcrafter.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global variables
last_generation_time = None
pipe = None
last_xformers_state = None
last_offload_state = None

def initialize_pipeline(enable_xformers=False):
    global pipe
    logger.debug("Initializing pipeline")
    unet = DiffusersUNetSpatioTemporalConditionModelDepthCrafter.from_pretrained(
        "tencent/DepthCrafter",
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
    )
    pipe = DepthCrafterPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        unet=unet,
        torch_dtype=torch.float16,
        variant="fp16",
    )
    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.debug("Xformers enabled during initialization")
        except Exception as e:
            logger.error(f"Failed to enable xformers: {str(e)}")
            pipe.disable_xformers_memory_efficient_attention()
    else:
        pipe.disable_xformers_memory_efficient_attention()
        logger.debug("Xformers disabled during initialization")
    gc.collect()
    torch.cuda.empty_cache()
    return pipe

# Initialize pipeline at startup
initialize_pipeline()

examples = [
    ["examples/example_01.mp4", 5, 1.0, 1024, -1, -1, 110, 25, "Model", False, "./demo_output", 42, True, False, False],
    ["examples/example_02.mp4", 5, 1.0, 1024, -1, -1, 110, 25, "Model", False, "./demo_output", 42, True, False, False],
    ["examples/example_03.mp4", 5, 1.0, 1024, -1, -1, 110, 25, "Model", False, "./demo_output", 42, True, False, False],
    ["examples/example_04.mp4", 5, 1.0, 1024, -1, -1, 110, 25, "Model", False, "./demo_output", 42, True, False, False],
    ["examples/example_05.mp4", 5, 1.0, 1024, -1, -1, 110, 25, "Model", False, "./demo_output", 42, True, False, False],
]

@spaces.GPU(duration=120)
def infer_depth(
    video: str,
    num_denoising_steps: int,
    guidance_scale: float,
    max_res: int = 1024,
    process_length: int = -1,
    target_fps: int = -1,
    window_size: int = 110,
    overlap: int = 25,
    cpu_offload: str = "Model",
    enable_xformers: bool = False,
    save_folder: str = "./demo_output",
    save_npz: bool = False,
    seed: int = 42,
    track_time: bool = True,
    resize_to_original: bool = False,
):
    global last_generation_time, pipe, last_xformers_state, last_offload_state
    logger.debug(f"Input parameters: video={video}, num_denoising_steps={num_denoising_steps}, "
                 f"guidance_scale={guidance_scale}, max_res={max_res}, process_length={process_length}, "
                 f"target_fps={target_fps}, window_size={window_size}, overlap={overlap}, "
                 f"cpu_offload={cpu_offload}, enable_xformers={enable_xformers}, save_folder={save_folder}, "
                 f"seed={seed}, track_time={track_time}, save_npz={save_npz}, resize_to_original={resize_to_original}")

    # Validate inputs
    if not isinstance(video, str):
        logger.error(f"Invalid video input: expected string, got {type(video)}")
        raise ValueError("Video input must be a valid file path (string).")
    if not os.path.isfile(video):
        logger.error(f"Video file does not exist: {video}")
        raise ValueError(f"Video file not found: {video}")
    if not isinstance(save_folder, str):
        logger.error(f"Invalid save_folder: expected string, got {type(save_folder)}")
        raise ValueError("Save folder must be a valid directory path (string).")
    if window_size < 2:
        logger.error(f"Invalid window_size: {window_size}. Must be at least 2 for temporal consistency.")
        raise ValueError("Window size must be at least 2 to ensure temporal consistency.")

    start_time = time.time()
    logger.debug(f"Starting inference at {time.strftime('%Y-%m-%d %H:%M:%S')}")

    # Log initial VRAM usage
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.debug(f"Initial VRAM: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved, "
                     f"{vram_total:.2f} GB total")

    # Reinitialize pipeline if xformers or offload settings changed
    if last_xformers_state != enable_xformers or last_offload_state != cpu_offload:
        logger.debug("Reinitializing pipeline due to xformers or offload change")
        pipe = initialize_pipeline(enable_xformers)
        last_xformers_state = enable_xformers
        last_offload_state = cpu_offload

    # Configure pipeline optimizations
    logger.debug(f"Configuring pipeline with enable_xformers={enable_xformers}")
    xformers_start = time.time()
    if enable_xformers and last_xformers_state is None:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.debug("Xformers enabled")
        except Exception as e:
            logger.error(f"Failed to enable xformers: {str(e)}")
            pipe.disable_xformers_memory_efficient_attention()
            logger.debug("Xformers not enabled, continuing without")
    elif not enable_xformers and last_xformers_state is None:
        pipe.disable_xformers_memory_efficient_attention()
        logger.debug("Xformers explicitly disabled")
    xformers_time = time.time() - xformers_start
    logger.debug(f"Xformers configuration time: {xformers_time:.2f}s")
    pipe.enable_attention_slicing("max")
    logger.debug("Enabled maximum attention slicing")

    # Apply CPU offloading or move to GPU
    logger.debug(f"Applying CPU offload mode: {cpu_offload}")
    if cpu_offload == "Model":
        pipe.enable_model_cpu_offload()
        logger.debug("Model CPU offload enabled")
    elif cpu_offload == "Sequential":
        pipe.enable_sequential_cpu_offload()
        logger.debug("Sequential CPU offload enabled")
    else:
        pipe.to("cuda")
        logger.debug("No CPU offload, pipeline moved to CUDA")

    # Read video frames (scaled and original)
    logger.debug(f"Reading video frames with process_length={process_length}, target_fps={target_fps}, max_res={max_res}")
    try:
        frames, original_frames, effective_fps = read_video_frames(video, process_length, target_fps, max_res, return_original=True)
        logger.debug(f"Read {len(frames)} frames, effective FPS={effective_fps}, scaled frame shape={frames.shape}, "
                     f"original frame shape={original_frames.shape}")
    except Exception as e:
        logger.error(f"Failed to read video frames: {str(e)}")
        raise

    # Log VRAM after loading frames
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(f"VRAM after loading frames: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved")

    # Inference with fallback for CUDA errors
    logger.debug("Starting inference with torch.inference_mode")
    with torch.inference_mode():
        try:
            logger.debug(f"Running pipeline with {len(frames)} frames, window_size={window_size}, overlap={overlap}")
            res = pipe(
                frames,
                height=frames.shape[1],
                width=frames.shape[2],
                output_type="np",
                guidance_scale=guidance_scale,
                num_inference_steps=num_denoising_steps,
                window_size=window_size,
                overlap=overlap,
                track_time=track_time,
            ).frames[0]
            logger.debug(f"Inference completed, result shape={res.shape}")
        except RuntimeError as e:
            logger.error(f"Inference error: {str(e)}")
            if "CUDA error: invalid configuration argument" in str(e):
                logger.warning("CUDA error detected. Retrying without xformers...")
                pipe.disable_xformers_memory_efficient_attention()
                pipe.enable_attention_slicing("max")
                res = pipe(
                    frames,
                    height=frames.shape[1],
                    width=frames.shape[2],
                    output_type="np",
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_denoising_steps,
                    window_size=window_size,
                    overlap=overlap,
                    track_time=track_time,
                ).frames[0]
                logger.debug(f"Retry inference completed, result shape={res.shape}")
            else:
                raise e

    # Process depth map
    logger.debug("Processing depth map")
    res = res.sum(-1) / res.shape[-1]
    res = (res - res.min()) / (res.max() - res.min())

    # Upscale depth map to original resolution if enabled
    if resize_to_original:
        logger.debug(f"Upscaling depth map to original resolution: {original_frames.shape[1:3]}")
        upscaled_res = []
        for frame in res:
            frame = cv2.resize(frame, (original_frames.shape[2], original_frames.shape[1]), interpolation=cv2.INTER_CUBIC)
            upscaled_res.append(frame)
        res = np.array(upscaled_res)
        logger.debug(f"Upscaled depth map shape: {res.shape}")

    # Save results
    logger.debug(f"Saving results to {save_folder}")
    save_path = os.path.join(save_folder, os.path.splitext(os.path.basename(video))[0])
    logger.debug(f"Save path: {save_path}")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if save_npz:
        logger.debug("Saving depth map as .npz")
        np.savez_compressed(save_path + ".npz", depth=res)

    # Use original frames if resize_to_original is enabled, else use scaled frames
    output_frames = original_frames if resize_to_original else frames
    res_rgb = np.stack([res] * 3, axis=-1)
    stacked_frames = np.concatenate([output_frames, res_rgb], axis=1)

    # Calculate output video duration
    output_duration = len(output_frames) / effective_fps
    logger.debug(f"Output video duration: {output_duration:.2f}s with {len(output_frames)} frames at {effective_fps} FPS")

    # Save temporary videos without audio
    temp_input_path = save_path + "_input_temp.mp4"
    temp_depth_path = save_path + "_depth_temp.mp4"
    final_input_path = save_path + "_input.mp4"
    final_depth_path = save_path + "_depth.mp4"

    logger.debug(f"Saving temporary depth video: {temp_depth_path}, shape={stacked_frames.shape}")
    save_video(stacked_frames, temp_depth_path, fps=effective_fps)
    logger.debug(f"Saving temporary input video: {temp_input_path}")
    save_video(output_frames, temp_input_path, fps=effective_fps)

    # Extract and mux audio
    try:
        logger.debug(f"Extracting audio from input video: {video}")
        with VideoFileClip(video) as input_clip:
            if input_clip.audio:
                logger.debug(f"Trimming audio to match output duration: {output_duration:.2f}s")
                audio = input_clip.audio.subclip(0, min(output_duration, input_clip.duration))

                # Mux audio into input video
                logger.debug(f"Muxing audio into input video: {final_input_path}")
                with VideoFileClip(temp_input_path) as temp_input_clip:
                    temp_input_clip = temp_input_clip.set_audio(audio)
                    temp_input_clip.write_videofile(final_input_path, codec="libx264", audio_codec="aac")
                    temp_input_clip.close()

                # Mux audio into depth video
                logger.debug(f"Muxing audio into depth video: {final_depth_path}")
                with VideoFileClip(temp_depth_path) as temp_depth_clip:
                    temp_depth_clip = temp_depth_clip.set_audio(audio)
                    temp_depth_clip.write_videofile(final_depth_path, codec="libx264", audio_codec="aac")
                    temp_depth_clip.close()

                audio.close()
            else:
                logger.debug("No audio found in input video, saving videos without audio")
                os.rename(temp_input_path, final_input_path)
                os.rename(temp_depth_path, final_depth_path)
        input_clip.close()
    except Exception as e:
        logger.error(f"Failed to process audio: {str(e)}")
        logger.debug("Saving videos without audio due to audio processing error")
        os.rename(temp_input_path, final_input_path)
        os.rename(temp_depth_path, final_depth_path)

    # Clean up temporary files
    for temp_file in [temp_input_path, temp_depth_path]:
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
                logger.debug(f"Removed temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_file}: {str(e)}")

    # Log final VRAM usage
    if torch.cuda.is_available():
        vram_allocated = torch.cuda.memory_allocated() / 1e9
        vram_reserved = torch.cuda.memory_reserved() / 1e9
        logger.debug(f"Final VRAM: {vram_allocated:.2f} GB allocated, {vram_reserved:.2f} GB reserved")

    # Clear cache
    logger.debug("Clearing CUDA cache")
    gc.collect()
    torch.cuda.empty_cache()

    total_time = time.time() - start_time
    last_generation_time = total_time
    logger.debug(f"Total inference time: {total_time:.2f}s")

    return [
        final_input_path,
        final_depth_path,
    ]

def construct_demo():
    with gr.Blocks(analytics_enabled=False) as depthcrafter_iface:
        gr.Markdown(
            """
            <div align='center'> <h1> DepthCrafter: Generating Consistent Long Depth Sequences for Open-world Videos </span> </h1> \
                        <h2 style='font-weight: 450; font-size: 1rem; margin: 0rem'>\
                        <a href='https://wbhu.github.io'>Wenbo Hu</a>, \
                        <a href='https://scholar.google.com/citations?user=qgdesEcAAAAJ&hl=en'>Xiangjun Gao</a>, \
                        <a href='https://xiaoyu258.github.io/'>Xiaoyu Li</a>, \
                        <a href='https://scholar.google.com/citations?user=tZ3dS3MAAAAJ&hl=en'>Sijie Zhao</a>, \
                        <a href='https://vinthony.github.io/academic'> Xiaodong Cun</a>, \
                        <a href='https://yzhang2016.github.io'>Yong Zhang</a>, \
                        <a href='https://home.cse.ust.hk/~quan'>Long Quan</a>, \
                        <a href='https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en'>Ying Shan</a>\
                    </h2> \
                    <a style='font-size:18px;color: #000000'>If you find DepthCrafter useful, please help тнР the </a>\
                    <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>\
                    <a style='font-size:18px;color: #000000'>, which is important to Open-Source projects. Thanks!</a>\
                        <a style='font-size:18px;color: #000000' href='https://arxiv.org/abs/2409.02095'> [ArXiv] </a>\
                        <a style='font-size:18px;color: #000000' href='https://depthcrafter.github.io/'> [Project Page] </a> </div>
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video")

            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    output_video_1 = gr.Video(
                        label="Preprocessed video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )
                    output_video_2 = gr.Video(
                        label="Generated Depth Video (Input + Depth)",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row(equal_height=False):
                    with gr.Accordion("Advanced Settings", open=False):
                        num_denoising_steps = gr.Slider(
                            label="num denoising steps",
                            minimum=1,
                            maximum=25,
                            value=5,
                            step=1,
                        )
                        guidance_scale = gr.Slider(
                            label="cfg scale",
                            minimum=1.0,
                            maximum=1.2,
                            value=1.0,
                            step=0.1,
                        )
                        max_res = gr.Slider(
                            label="max resolution",
                            minimum=512,
                            maximum=2048,
                            value=1024,
                            step=64,
                            info="Resolutions above 1024 may cause VRAM errors or degraded depth map quality."
                        )
                        process_length = gr.Slider(
                            label="process length",
                            minimum=-1,
                            maximum=280,
                            value=60,
                            step=1,
                        )
                        process_target_fps = gr.Slider(
                            label="target FPS",
                            minimum=-1,
                            maximum=30,
                            value=15,
                            step=1,
                        )
                        window_size = gr.Slider(
                            label="Window Size",
                            minimum=2,
                            maximum=200,
                            value=110,
                            step=1,
                            info="Number of frames processed together (minimum 2 for temporal consistency, lower saves VRAM)."
                        )
                        overlap = gr.Slider(
                            label="Overlap",
                            minimum=0,
                            maximum=100,
                            value=25,
                            step=1,
                            info="Frame overlap between windows (lower saves VRAM)."
                        )
                        cpu_offload = gr.Dropdown(
                            label="CPU Offload Mode",
                            choices=["None", "Model", "Sequential"],
                            value="Model",
                            info="Select CPU offloading to save VRAM. 'Model' saves VRAM; 'Sequential' saves more but is slower."
                        )
                        enable_xformers = gr.Checkbox(
                            label="Enable Xformers",
                            value=False,
                            info="Enable xformers for memory-efficient attention (may increase runtime on some GPUs)."
                        )
                        save_folder = gr.Textbox(
                            label="Save Folder",
                            value="./demo_output",
                            info="Directory to save output videos."
                        )
                        seed = gr.Slider(
                            label="Seed",
                            minimum=0,
                            maximum=10000,
                            value=42,
                            step=1,
                            info="Random seed for reproducibility."
                        )
                        track_time = gr.Checkbox(
                            label="Track Time",
                            value=True,
                            info="Log inference time."
                        )
                        save_npz = gr.Checkbox(
                            label="Save NPZ",
                            value=False,
                            info="Save depth map as .npz file."
                        )
                        resize_to_original = gr.Checkbox(
                            label="Resize output to match original input video",
                            value=False,
                            info="Upscale depth map and use original input frames for output video."
                        )
                    generate_btn = gr.Button("Generate")
                    last_time_display = gr.Textbox(
                        label="Last Generation Time",
                        value="No generation run yet",
                        interactive=False,
                        lines=1,
                        max_lines=1,
                        show_copy_button=False
                    )

            with gr.Column(scale=2):
                pass

        def update_last_time_display():
            return f"Last generation time: {last_generation_time:.2f}s" if last_generation_time is not None else "No generation run yet"

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
                process_target_fps,
                window_size,
                overlap,
                cpu_offload,
                enable_xformers,
                save_folder,
                seed,
                track_time,
                save_npz,
                resize_to_original,
            ],
            outputs=[output_video_1, output_video_2, last_time_display],
            fn=lambda *args: infer_depth(*args) + [update_last_time_display()],
            cache_examples="lazy",
        )
        gr.Markdown(
            """
            <span style='font-size:18px;color: #E7CCCC'>Note: 
            For time quota consideration, we set the default parameters to be more efficient here,
            with a trade-off of shorter video length and slightly lower quality.
            You may adjust the parameters according to our 
            <a style='font-size:18px;color: #FF5DB0' href='https://github.com/Tencent/DepthCrafter'>[Github Repo]</a>
             for better results if you have enough time quota.
            The CPU Offload Mode ('Model' or 'Sequential') can reduce VRAM usage, with 'Sequential' being slower but saving more memory.
            Resolutions above 1024 may cause VRAM errors or degraded depth map quality; use lower window_size/overlap to mitigate.
            </span>
            """
        )

        generate_btn.click(
            fn=lambda *args: infer_depth(*args) + [update_last_time_display()],
            inputs=[
                input_video,
                num_denoising_steps,
                guidance_scale,
                max_res,
                process_length,
                process_target_fps,
                window_size,
                overlap,
                cpu_offload,
                enable_xformers,
                save_folder,
                seed,
                track_time,
                save_npz,
                resize_to_original,
            ],
            outputs=[output_video_1, output_video_2, last_time_display],
        )

    return depthcrafter_iface

if __name__ == "__main__":
    logger.debug("Launching Gradio interface")
    demo = construct_demo()
    demo.queue()
    demo.launch(share=True)
