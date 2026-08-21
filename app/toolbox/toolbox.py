import os
import gc
import sys
import subprocess
import types
import torch
import re
import numpy as np
import imageio
import gradio as gr
import shutil
import traceback
import math
import json
from datetime import datetime
from pathlib import Path

import devicetorch

# Local imports for RIFE and ESRGAN
from toolbox.rife_core import RIFEHandler
from naming_utils import (
    STAGE_INTERP,
    STAGE_POSTED,
    strip_stage,
    with_stage,
    promote_stage,
    step2_filename,
    move_to_bin,
)

device_name_str = devicetorch.get(torch)

class ToolboxProcessor:
    """
    A processor for handling upscale, frame adjustment, and export operations.
    """
    def __init__(self, autosave_enabled=True):
        self.device_obj = torch.device(device_name_str)
        # toolbox.py is in /app/toolbox, so parent.parent gets us to /app
        app_dir = Path(__file__).parent.parent.absolute()
        
        self.output_dir = app_dir / "outputs" / "toolbox"
        self.temp_dir = app_dir / "_temp" / "toolbox"
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.temp_dir, exist_ok=True)

        self.ffmpeg_exe, self.ffprobe_exe, self.has_ffmpeg = self._initialize_ffmpeg()
        self.autosave_enabled = autosave_enabled
        self.rife_handler = RIFEHandler()
        # Encoder prefs (quality-preserving speed). Override via webui queue params.
        self.export_preset = "medium"       # was "slow" — same CRF, ~2–3× faster
        self.prefer_nvenc = True            # use GPU encode when available
        self._nvenc_ok = None               # lazy probe

    def _nvenc_available(self) -> bool:
        """True if this ffmpeg build can encode h264_nvenc (one-time probe)."""
        if self._nvenc_ok is not None:
            return self._nvenc_ok
        if not self.has_ffmpeg or not self.ffmpeg_exe:
            self._nvenc_ok = False
            return False
        try:
            r = subprocess.run(
                [self.ffmpeg_exe, "-hide_banner", "-encoders"],
                capture_output=True, text=True, timeout=8, errors="ignore",
            )
            self._nvenc_ok = "h264_nvenc" in (r.stdout or "")
        except Exception:
            self._nvenc_ok = False
        if self._nvenc_ok:
            print("INFO: h264_nvenc available — final exports will use GPU encode (same quality CQ).")
        return self._nvenc_ok

    def _intermediate_ffmpeg_params(self, output_quality: int):
        """
        Fast intermediate encode after RIFE (will be re-encoded on Export).
        Use ultrafast + lower CRF so quality is preserved for the next stage
        without spending time on a slow preset twice.
        """
        # Quality slider 0-100 → intermediate CRF (keep high fidelity for re-encode)
        crf_final = int(35 - (output_quality / 100) * 20)
        crf_inter = max(10, min(crf_final, 14))  # never softer than final target
        return ['-crf', str(crf_inter), '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-loglevel', 'error']

    def _export_video_args(self, quality: int, max_width: int, video_path: str):
        """
        Build ffmpeg video codec args for final export.
        Prefer NVENC CQ on NVIDIA (huge speedup, visually matched at high CQ);
        else libx264 medium (not slow) at same CRF — quality parity, much faster.
        """
        crf = int(35 - (quality / 100) * 20)
        preset = (self.export_preset or "medium").strip().lower()
        if preset not in (
            "ultrafast", "superfast", "veryfast", "faster", "fast",
            "medium", "slow", "slower", "veryslow",
        ):
            preset = "medium"

        # Only scale when needed (avoids pointless filter work)
        need_scale = True
        try:
            w, h = self._probe_dims(video_path)
            if w and w <= int(max_width):
                need_scale = False
        except Exception:
            need_scale = True
        vf = f"scale='min({int(max_width)},iw)':-2:flags=lanczos" if need_scale else None

        if self.prefer_nvenc and self._nvenc_available():
            # Map quality% → NVENC CQ (lower = better). Align near x264 CRF band.
            cq = max(12, min(28, crf + 1))
            # p5/p6 = high quality presets on modern NVENC (4090 etc.)
            args = [
                "-c:v", "h264_nvenc",
                "-preset", "p5",
                "-rc", "vbr",
                "-cq", str(cq),
                "-b:v", "0",
                "-profile:v", "high",
                "-pix_fmt", "yuv420p",
            ]
            return args, vf, f"h264_nvenc cq={cq}"

        args = [
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
            "-pix_fmt", "yuv420p",
        ]
        return args, vf, f"libx264 preset={preset} crf={crf}"

    def _probe_dims(self, video_path):
        if not self.has_ffmpeg:
            return 0, 0
        try:
            cmd = [
                self.ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
                str(video_path),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=8).stdout.strip()
            if "x" in out:
                w, h = out.split("x", 1)
                return int(w), int(h)
        except Exception:
            pass
        return 0, 0

    def _normalize_input_path(self, video_path):
        """
        Normalizes input video path to prevent filename length issues.
        If the path is too long or in a problematic location (like Gradio's temp with hash subdirs),
        copies the video to toolbox temp directory with a cleaned filename.
        """
        if not video_path or not os.path.exists(video_path):
            return video_path
        
        video_path_obj = Path(video_path).resolve()
        
        # Check if path is too long (Windows has 260 char limit, be conservative)
        # Or if it's in a Gradio temp directory with hash subdirectories
        path_str = str(video_path_obj)
        is_too_long = len(path_str) > 200
        is_gradio_temp = 'gradio' in path_str.lower() or len(video_path_obj.parts) > 10
        
        if is_too_long or is_gradio_temp:
            # Create a normalized copy in toolbox temp directory
            original_name = video_path_obj.stem
            extension = video_path_obj.suffix
            
            # Clean the filename aggressively
            clean_name = self._clean_filename(original_name, max_base_length=40)
            
            # Generate a short unique identifier to avoid collisions
            import hashlib
            file_hash = hashlib.md5(path_str.encode()).hexdigest()[:8]
            
            normalized_filename = f"{clean_name}_{file_hash}{extension}"
            normalized_path = self.temp_dir / normalized_filename
            
            # Copy if not already normalized
            if not normalized_path.exists():
                # Ensure temp directory exists (may have been deleted by cleanup)
                os.makedirs(self.temp_dir, exist_ok=True)
                print(f"Normalizing input path (length: {len(path_str)} chars)")
                print(f"  From: {video_path_obj.name}")
                print(f"  To: {normalized_filename}")
                shutil.copy2(video_path, normalized_path)
            
            return str(normalized_path)
        
        return video_path
        
    def set_autosave_mode(self, is_enabled):
        """Updates the autosave mode."""
        self.autosave_enabled = is_enabled
        status = "ON" if is_enabled else "OFF"
        return f"✅ Autosave is now {status}."

    def save_video_from_any_source(self, video_source_path):
        """
        Copies a video from the toolbox temp directory to the permanent output folder,
        preserving its filename. This is the backend for the Manual Save button.
        """
        try:
            # Get the filename from the source path
            source_filename = Path(video_source_path).name
            
            # Create the destination path
            destination_path = self.output_dir / source_filename
            
            print(f"Copying video from '{video_source_path}' to '{destination_path}'")
            
            # Copy the file to preserve it in temp for further operations
            shutil.copy2(video_source_path, destination_path)
            
            return str(destination_path)
            
        except Exception as e:
            print(f"Error during manual save: {e}\n{traceback.format_exc()}")
            return None
            
    def open_output_folder(self):
        """Opens the toolbox output folder in the system's file explorer."""
        folder_path = os.path.abspath(self.output_dir)
        try:
            if sys.platform == "win32":
                os.startfile(folder_path)
            elif sys.platform == "darwin":
                subprocess.Popen(["open", folder_path])
            else:
                subprocess.Popen(["xdg-open", folder_path])
            return f"Opened output folder: {folder_path}"
        except Exception as e:
            return f"❌ Error opening folder: {e}"

    def analyze_video_html(self, video_path):
        """Analyzes video file and returns detailed HTML information for display."""
        if not video_path:
            return '<div style="padding: 12px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 6px; color: #856404;">⚠️ No video provided for analysis</div>'
        
        # Normalize input path to prevent filename length issues
        video_path = self._normalize_input_path(video_path)
        
        resolved_path = str(Path(video_path).resolve())
        
        # Get file size
        file_size_display = "N/A"
        try:
            if os.path.exists(resolved_path):
                size_bytes = os.path.getsize(resolved_path)
                if size_bytes < 1024**2:
                    file_size_display = f"{size_bytes/1024:.1f} KB"
                elif size_bytes < 1024**3:
                    file_size_display = f"{size_bytes/1024**2:.1f} MB"
                else:
                    file_size_display = f"{size_bytes/1024**3:.2f} GB"
        except Exception as e:
            print(f"Warning: Could not get file size: {e}")
        
        # Initialize variables
        video_width, video_height = 0, 0
        duration = 0.0
        calculated_fps = 0.0
        num_frames_value = 0
        duration_display, fps_display, resolution_display = "N/A", "N/A", "N/A"
        nframes_display, has_audio_str = "N/A", "No"
        codec_info = "N/A"
        bitrate_display = "N/A"
        analysis_source = "imageio"
        
        # Try ffprobe first if available
        if self.has_ffmpeg and self.ffprobe_exe:
            try:
                probe_cmd = [
                    self.ffprobe_exe, "-v", "error", "-show_format", "-show_streams",
                    "-of", "json", resolved_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                probe_data = json.loads(result.stdout)
                
                video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
                audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)
                
                if video_stream:
                    analysis_source = "ffprobe"
                    
                    # Duration
                    duration_str = probe_data.get("format", {}).get("duration", "0")
                    duration = float(duration_str) if duration_str and duration_str.replace('.', '', 1).isdigit() else 0.0
                    duration_display = f"{duration:.2f}s"
                    
                    # Bitrate
                    bitrate_str = probe_data.get("format", {}).get("bit_rate", "0")
                    if bitrate_str and bitrate_str.isdigit():
                        bitrate_mbps = int(bitrate_str) / 1_000_000
                        bitrate_display = f"{bitrate_mbps:.1f} Mbps"
                    
                    # Codec
                    codec_name = video_stream.get("codec_name", "N/A")
                    codec_long = video_stream.get("codec_long_name", "")
                    if "H.264" in codec_long or "AVC" in codec_long:
                        codec_info = "H.264/AVC"
                    elif "H.265" in codec_long or "HEVC" in codec_long:
                        codec_info = "H.265/HEVC"
                    elif "VP9" in codec_long:
                        codec_info = "VP9"
                    elif "VP8" in codec_long:
                        codec_info = "VP8"
                    else:
                        codec_info = codec_name.upper()
                    
                    # FPS
                    def parse_fps(fps_s):
                        if isinstance(fps_s, (int, float)):
                            return float(fps_s)
                        if isinstance(fps_s, str) and "/" in fps_s:
                            try:
                                num, den = map(float, fps_s.split('/'))
                                return num / den if den != 0 else 0.0
                            except ValueError:
                                return 0.0
                        try:
                            return float(fps_s)
                        except ValueError:
                            return 0.0
                    
                    r_frame_rate = video_stream.get("r_frame_rate", "0/0")
                    avg_frame_rate = video_stream.get("avg_frame_rate", "0/0")
                    r_fps = parse_fps(r_frame_rate)
                    avg_fps = parse_fps(avg_frame_rate)
                    
                    if r_fps > 0:
                        calculated_fps = r_fps
                        fps_display = f"{r_fps:.1f} FPS"
                    if avg_fps > 0 and abs(r_fps - avg_fps) > 0.01:
                        calculated_fps = avg_fps
                        fps_display = f"{avg_fps:.1f} FPS"
                    elif avg_fps > 0 and r_fps <= 0:
                        calculated_fps = avg_fps
                        fps_display = f"{avg_fps:.1f} FPS"
                    
                    # Resolution
                    video_width = video_stream.get("width", 0)
                    video_height = video_stream.get("height", 0)
                    resolution_display = f"{video_width}×{video_height}"
                    
                    # Frame count
                    nframes_str = video_stream.get("nb_frames")
                    if nframes_str and nframes_str.isdigit():
                        num_frames_value = int(nframes_str)
                        nframes_display = str(num_frames_value)
                    elif duration > 0 and calculated_fps > 0:
                        num_frames_value = int(duration * calculated_fps)
                        nframes_display = f"{num_frames_value}"
                    
                    # Audio
                    if audio_stream:
                        audio_codec = audio_stream.get('codec_name', 'N/A').upper()
                        audio_channels = audio_stream.get('channels', 'N/A')
                        audio_rate = audio_stream.get('sample_rate', 'N/A')
                        if audio_rate != 'N/A':
                            audio_rate = f"{int(audio_rate)/1000:.1f}kHz"
                        has_audio_str = f"Yes ({audio_codec}, {audio_channels}ch, {audio_rate})"
                    else:
                        has_audio_str = "No"
                    
                    print("Video analysis complete (using ffprobe).")
            except Exception as e:
                print(f"ffprobe analysis failed, falling back to imageio: {e}")
                analysis_source = "imageio"
        
        # Fallback to imageio
        if analysis_source == "imageio":
            reader = None
            try:
                reader = imageio.get_reader(resolved_path)
                meta = reader.get_meta_data()
                
                # Duration
                duration_val = meta.get('duration')
                duration = float(duration_val) if duration_val is not None else 0.0
                duration_display = f"{duration:.2f}s"
                
                # FPS
                fps_val = meta.get('fps')
                calculated_fps = float(fps_val) if fps_val is not None else 0.0
                fps_display = f"{calculated_fps:.1f} FPS"
                
                # Resolution
                size_val = meta.get('size')
                if isinstance(size_val, tuple) and len(size_val) == 2:
                    video_width, video_height = int(size_val[0]), int(size_val[1])
                    resolution_display = f"{video_width}×{video_height}"
                
                # Frame count
                nframes_val = meta.get('nframes')
                if nframes_val not in [float('inf'), "N/A", None] and isinstance(nframes_val, (int, float)):
                    num_frames_value = int(nframes_val)
                    nframes_display = str(num_frames_value)
                elif duration > 0 and calculated_fps > 0:
                    num_frames_value = int(duration * calculated_fps)
                    nframes_display = f"{num_frames_value}"
                
                has_audio_str = "Unknown"
                print("Video analysis complete (using imageio).")
            except Exception as e:
                print(f"Error analyzing video with imageio: {e}")
                return f'<div style="padding: 12px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 6px; color: #721c24;">❌ Error analyzing video: {str(e)}</div>'
            finally:
                if reader:
                    reader.close()
        
        # Build HTML display with toolbox-specific details (3x2 layout)
        # Combine related stats for better readability
        duration_fps_display = f"{duration_display} @ {fps_display}"
        codec_bitrate_display = f"{codec_info} • {bitrate_display}"
        
        html = f'''
        <div style="padding: 16px; background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); border: 1px solid #667eea40; border-radius: 8px; font-family: 'Segoe UI', sans-serif;">
            <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 8px;">
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, rgba(209, 236, 241, 0.3) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">RESOLUTION</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{resolution_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #bbc1f2 0%, rgba(187, 193, 242, 0.3) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FRAMES</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{nframes_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, rgba(209, 236, 241, 0.3) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">DURATION & FPS</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{duration_fps_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #bbc1f2 0%, rgba(187, 193, 242, 0.3) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">FILE SIZE</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #362e54;">{file_size_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #d1ecf1 0%, rgba(209, 236, 241, 0.3) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #667eea;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">CODEC & BITRATE</div>
                    <div style="font-size: 1.1em; font-weight: 600; color: #415e78;">{codec_bitrate_display}</div>
                </div>
                <div style="background: linear-gradient(135deg, #bbc1f2 0%, rgba(187, 193, 242, 0.3) 100%); padding: 10px; border-radius: 6px; border-left: 3px solid #764ba2;">
                    <div style="font-size: 0.8em; color: #292626; margin-bottom: 4px;">AUDIO</div>
                    <div style="font-size: 1.0em; font-weight: 600; color: #362e54;">{has_audio_str}</div>
                </div>
            </div>
        </div>
        '''
        return html

    def analyze_video(self, video_path):
        """Analyzes video file and returns detailed information."""
        if not video_path:
            return "⚠️ No video provided for analysis."
        
        # Normalize input path to prevent filename length issues
        video_path = self._normalize_input_path(video_path)
        
        resolved_path = str(Path(video_path).resolve())
        report = []
        
        # Get file size
        file_size_display = "N/A"
        try:
            if os.path.exists(resolved_path):
                size_bytes = os.path.getsize(resolved_path)
                if size_bytes < 1024:
                    file_size_display = f"{size_bytes} B"
                elif size_bytes < 1024**2:
                    file_size_display = f"{size_bytes/1024:.2f} KB"
                elif size_bytes < 1024**3:
                    file_size_display = f"{size_bytes/1024**2:.2f} MB"
                else:
                    file_size_display = f"{size_bytes/1024**3:.2f} GB"
        except Exception as e:
            print(f"Warning: Could not get file size: {e}")
        
        # Initialize variables
        video_width, video_height = 0, 0
        num_frames_value = None
        duration_display, fps_display, resolution_display = "N/A", "N/A", "N/A"
        nframes_display, has_audio_str = "N/A", "No"
        analysis_source = "imageio"
        
        # Try ffprobe first if available
        if self.has_ffmpeg and self.ffprobe_exe:
            try:
                probe_cmd = [
                    self.ffprobe_exe, "-v", "error", "-show_format", "-show_streams",
                    "-of", "json", resolved_path
                ]
                result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True, errors='ignore')
                probe_data = json.loads(result.stdout)
                
                video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
                audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)
                
                if video_stream:
                    analysis_source = "ffprobe"
                    
                    # Duration
                    duration_str = probe_data.get("format", {}).get("duration", "0")
                    duration = float(duration_str) if duration_str and duration_str.replace('.', '', 1).isdigit() else 0.0
                    duration_display = f"{duration:.2f} seconds"
                    
                    # FPS
                    def parse_fps(fps_s):
                        if isinstance(fps_s, (int, float)):
                            return float(fps_s)
                        if isinstance(fps_s, str) and "/" in fps_s:
                            try:
                                num, den = map(float, fps_s.split('/'))
                                return num / den if den != 0 else 0.0
                            except ValueError:
                                return 0.0
                        try:
                            return float(fps_s)
                        except ValueError:
                            return 0.0
                    
                    r_frame_rate = video_stream.get("r_frame_rate", "0/0")
                    avg_frame_rate = video_stream.get("avg_frame_rate", "0/0")
                    r_fps = parse_fps(r_frame_rate)
                    avg_fps = parse_fps(avg_frame_rate)
                    
                    calculated_fps = 0.0
                    if r_fps > 0:
                        calculated_fps = r_fps
                        fps_display = f"{r_fps:.2f} FPS"
                    if avg_fps > 0 and abs(r_fps - avg_fps) > 0.01:
                        calculated_fps = avg_fps
                        fps_display = f"{avg_fps:.2f} FPS (Avg, r: {r_fps:.2f})"
                    elif avg_fps > 0 and r_fps <= 0:
                        calculated_fps = avg_fps
                        fps_display = f"{avg_fps:.2f} FPS (Average)"
                    
                    # Resolution
                    video_width = video_stream.get("width", 0)
                    video_height = video_stream.get("height", 0)
                    resolution_display = f"{video_width}x{video_height}" if video_width and video_height else "N/A"
                    
                    # Frame count
                    nframes_str = video_stream.get("nb_frames")
                    if nframes_str and nframes_str.isdigit():
                        num_frames_value = int(nframes_str)
                        nframes_display = str(num_frames_value)
                    elif duration > 0 and calculated_fps > 0:
                        num_frames_value = int(duration * calculated_fps)
                        nframes_display = f"{num_frames_value} (Calculated)"
                    
                    # Audio
                    if audio_stream:
                        has_audio_str = (
                            f"Yes (Codec: {audio_stream.get('codec_name', 'N/A')}, "
                            f"Channels: {audio_stream.get('channels', 'N/A')}, "
                            f"Rate: {audio_stream.get('sample_rate', 'N/A')} Hz)"
                        )
                    
                    print("Video analysis complete (using ffprobe).")
            except Exception as e:
                print(f"ffprobe analysis failed, falling back to imageio: {e}")
                analysis_source = "imageio"
        
        # Fallback to imageio
        if analysis_source == "imageio":
            reader = None
            try:
                reader = imageio.get_reader(resolved_path)
                meta = reader.get_meta_data()
                
                # Duration
                duration_val = meta.get('duration')
                duration_display = f"{float(duration_val):.2f} seconds" if duration_val is not None else "N/A"
                
                # FPS
                fps_val = meta.get('fps')
                fps_display = f"{float(fps_val):.2f} FPS" if fps_val is not None else "N/A"
                
                # Resolution
                size_val = meta.get('size')
                if isinstance(size_val, tuple) and len(size_val) == 2:
                    video_width, video_height = int(size_val[0]), int(size_val[1])
                    resolution_display = f"{video_width}x{video_height}"
                
                # Frame count
                nframes_val = meta.get('nframes')
                if nframes_val not in [float('inf'), "N/A", None] and isinstance(nframes_val, (int, float)):
                    num_frames_value = int(nframes_val)
                    nframes_display = str(num_frames_value)
                elif hasattr(reader, 'count_frames'):
                    try:
                        nframes_count = reader.count_frames()
                        if nframes_count != float('inf'):
                            num_frames_value = int(nframes_count)
                            nframes_display = f"{num_frames_value} (Counted)"
                        else:
                            nframes_display = "Unknown (Stream)"
                    except Exception:
                        nframes_display = "Unknown"
                
                has_audio_str = "(Audio info not available via imageio)"
                print("Video analysis complete (using imageio).")
            except Exception as e:
                print(f"Error analyzing video with imageio: {e}")
                return f"❌ Error analyzing video: {e}"
            finally:
                if reader:
                    reader.close()
        
        # Build report
        report.append(f"📊 Video Analysis ({analysis_source})")
        report.append(f"File: {os.path.basename(video_path)}")
        report.append("─" * 50)
        report.append(f"📦 File Size: {file_size_display}")
        report.append(f"⏱️  Duration: {duration_display}")
        report.append(f"🎬 Frame Rate: {fps_display}")
        report.append(f"📐 Resolution: {resolution_display}")
        report.append(f"🎞️  Frames: {nframes_display}")
        report.append(f"🔊 Audio: {has_audio_str}")
        
        # Add upscale advisory
        # if video_width > 0 and video_height > 0:
            # HD_WIDTH = 1920
            # FOUR_K_WIDTH = 3800
            
            # is_hd_or_larger = video_width >= HD_WIDTH or video_height >= (HD_WIDTH * 9/16 * 0.95)
            # is_4k_or_larger = video_width >= FOUR_K_WIDTH or video_height >= (FOUR_K_WIDTH * 9/16 * 0.95)
            
            # warnings = []
            # if is_4k_or_larger:
                # warnings.append("This video is 4K+ resolution. Upscaling will be very slow and memory-intensive.")
            # elif is_hd_or_larger:
                # warnings.append("This video is HD or larger. Upscaling will be resource-intensive.")
            
            # if num_frames_value and num_frames_value > 900:
                # warnings.append(f"With {num_frames_value} frames, processing will be time-consuming.")
            
            # if warnings:
                # report.append("\n⚠️  PROCESSING ADVISORY")
                # report.append("─" * 50)
                # for warning in warnings:
                    # report.append(f"• {warning}")
        
        return "\n".join(report)

        
    def _initialize_ffmpeg(self):
        """Finds FFmpeg/FFprobe and sets status flags."""
        ffmpeg_path, ffprobe_path = self._find_ffmpeg_executables()
        has_ffmpeg = bool(ffmpeg_path) and bool(ffprobe_path)
        if not has_ffmpeg: print("WARNING: FFmpeg or FFprobe not found. Audio handling and some export formats will be disabled.")
        return ffmpeg_path, ffprobe_path, has_ffmpeg

    def _find_ffmpeg_executables(self):
        """Finds ffmpeg and ffprobe, prioritizing system PATH then imageio."""
        ffmpeg_path = shutil.which("ffmpeg") or shutil.which("ffmpeg.exe")
        ffprobe_path = shutil.which("ffprobe") or shutil.which("ffprobe.exe")
        if not ffmpeg_path:
            try:
                imageio_ffmpeg_exe = imageio.plugins.ffmpeg.get_exe()
                if os.path.isfile(imageio_ffmpeg_exe): ffmpeg_path = imageio_ffmpeg_exe
            except Exception: pass
        return ffmpeg_path, ffprobe_path

    def _clean_filename(self, filename, max_base_length=60):
        """
        Aggressively cleans filenames to prevent path length issues.
        - REMOVES all timestamps to prevent length accumulation
        - REMOVES duplicate operation suffixes (e.g., multiple _exported_)
        - TRUNCATES base filename to max_base_length to ensure path compatibility
        - PRESERVES pipeline stage tags _1 / _2 / _3 at end of stem
        """
        bare, stage = strip_stage(filename)
        filename = bare

        # Remove all timestamps (format: _YYYYMMDD_HHMMSS and _YYYYMMDD-HHMMSS)
        filename = re.sub(r'_\d{8}[-_]\d{6}', '', filename)

        # Also remove shorter timestamps (format: _HHMMSS) — not single-digit stages
        filename = re.sub(r'_\d{6}(?=_|$)', '', filename)

        # Remove duplicate operation suffixes to prevent accumulation
        # e.g., "exported_3840w_85q_exported_1920w_85q" -> "exported_1920w_85q"
        filename = re.sub(r'(_exported_\d+w_\d+q)+(?=_exported_)', '', filename)
        filename = re.sub(r'(_frames_[^_]+)+(?=_frames_)', '', filename)
        filename = re.sub(r'(_loop_[^_]+)+(?=_loop_)', '', filename)

        # Clean up multiple underscores that may result from removals
        filename = re.sub(r'_+', '_', filename)
        filename = filename.strip('_')

        # Truncate if still too long (keep room for _1/_2/_3)
        reserve = 2 if stage else 0
        limit = max(10, max_base_length - reserve)
        if len(filename) > limit:
            truncated = filename[:limit]
            last_underscore = truncated.rfind('_')
            if last_underscore > limit // 2:
                filename = truncated[:last_underscore]
            else:
                filename = truncated
            filename = filename.rstrip('_')

        if stage:
            filename = with_stage(filename, stage)
        return filename

    def _generate_output_path(self, input_path, suffix, ext=".mp4", is_temp=False, batch_folder=None, stage=None):
        """Generates a unique output path for processed videos with aggressive length management.

        stage: optional pipeline stage (1=upscale, 2=interp, 3=posted/export).
        Always placed as the LAST token before the extension for at-a-glance status.
        """
        try:
            raw_stem = Path(input_path).stem if Path(str(input_path)).suffix else str(input_path)
        except Exception:
            raw_stem = str(input_path)

        base_name = self._clean_filename(raw_stem)
        bare, existing_stage = strip_stage(base_name)

        # Use shorter timestamp format (HHMMSS) to save characters
        timestamp = datetime.now().strftime("%H%M%S")

        middle = f"{bare}_{suffix}_{timestamp}" if suffix else f"{bare}_{timestamp}"

        final_stage = None
        if stage is not None:
            final_stage = max(int(stage), existing_stage or 0)
        elif existing_stage:
            final_stage = existing_stage

        if final_stage in (1, 2, 3):
            filename = f"{middle}_{final_stage}{ext}"
        else:
            filename = f"{middle}{ext}"

        # Final safety check: if filename is still too long, truncate bare further
        max_filename_length = 150  # Conservative limit for Windows compatibility
        if len(filename) > max_filename_length:
            excess = len(filename) - max_filename_length
            new_base_length = max(20, len(bare) - excess - 10)
            bare = bare[:new_base_length].rstrip('_')
            middle = f"{bare}_{suffix}_{timestamp}" if suffix else f"{bare}_{timestamp}"
            if final_stage in (1, 2, 3):
                filename = f"{middle}_{final_stage}{ext}"
            else:
                filename = f"{middle}{ext}"
            print(f"WARNING: Filename truncated to fit path limits: {filename}")

        if is_temp:
            os.makedirs(self.temp_dir, exist_ok=True)
            return self.temp_dir / filename
        if batch_folder:
            target_dir = self.output_dir / batch_folder
            os.makedirs(target_dir, exist_ok=True)
            return target_dir / filename
        os.makedirs(self.output_dir, exist_ok=True)
        return self.output_dir / filename

    def _copy_to_permanent_storage(self, temp_path, final_path):
        """Copies a temp file to permanent storage and cleans up the source temp file."""
        try:
            # Check if source and destination are the same file
            temp_path_resolved = Path(temp_path).resolve()
            final_path_resolved = Path(final_path).resolve()
            
            if temp_path_resolved == final_path_resolved:
                # File is already in the correct location, no need to copy
                print(f"File already in permanent storage: {final_path}")
                return str(final_path)
            
            shutil.copy(temp_path, final_path)
            os.remove(temp_path)
            return str(final_path)
        except Exception as e:
            print(f"Error moving file to permanent storage: {e}")
            return str(temp_path)

    def _get_video_frame_count(self, video_path):
        """Uses ffprobe to get an accurate frame count."""
        if not self.has_ffmpeg: return None
        try:
            cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "v:0", "-count_frames",
                   "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", video_path]
            return int(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip())
        except Exception: return None

    def _has_audio_stream(self, video_path):
        """Checks if a video file has an audio stream using ffprobe."""
        if not self.has_ffmpeg: return False
        try:
            cmd = [self.ffprobe_exe, "-v", "error", "-select_streams", "a:0",
                   "-show_entries", "stream=codec_type", "-of", "csv=p=0", video_path]
            return "audio" in subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip().lower()
        except Exception: return False

    def _has_video_stream(self, video_path):
        """True if ffprobe sees a video stream (not audio-only)."""
        if not video_path or not os.path.exists(video_path):
            return False
        if not self.has_ffmpeg:
            return True
        try:
            cmd = [
                self.ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                str(video_path),
            ]
            out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=12)
            return "video" in (out.stdout or "").strip().lower()
        except Exception:
            return False

    def _probe_fps(self, video_path) -> float:
        """Best-effort source FPS (ffprobe, then imageio). 0 if unknown."""
        if not video_path or not os.path.exists(video_path):
            return 0.0
        if self.has_ffmpeg:
            try:
                cmd = [
                    self.ffprobe_exe, "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=avg_frame_rate,r_frame_rate",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    str(video_path),
                ]
                out = subprocess.run(
                    cmd, capture_output=True, text=True, check=False, timeout=12
                ).stdout.strip().splitlines()
                for line in out:
                    line = (line or "").strip()
                    if not line or line in ("0/0", "N/A"):
                        continue
                    if "/" in line:
                        a, b = line.split("/", 1)
                        den = float(b)
                        if den:
                            val = float(a) / den
                            if 1.0 <= val <= 480.0:
                                return val
                    else:
                        val = float(line)
                        if 1.0 <= val <= 480.0:
                            return val
            except Exception:
                pass
        try:
            with imageio.get_reader(video_path) as reader:
                fps = float((reader.get_meta_data() or {}).get("fps") or 0)
            if 1.0 <= fps <= 480.0:
                return fps
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _choose_interp_factor(src_fps: float, requested: int, max_out_fps: float) -> int:
        """
        Never 4× a 60fps clip into 240. Drop 4→2→1 until output stays at/under the cap.
        """
        factor = int(requested or 1)
        if factor < 1:
            factor = 1
        src = float(src_fps or 0)
        cap = float(max_out_fps or 120)
        if src <= 0:
            return factor
        while factor > 1 and (src * factor) > (cap + 1.0):
            factor //= 2
        return factor

    def adjust_frames(self, video_path, fps_mode, speed_factor, use_streaming, output_quality=90, progress=gr.Progress()):
        if not video_path: print("No input video for frame adjustment."); return None
        
        # Normalize input path to prevent filename length issues
        video_path = self._normalize_input_path(video_path)
        if not self._has_video_stream(video_path):
            print(
                f"ERROR: No video stream in {Path(video_path).name} "
                f"(audio-only or corrupt). RIFE cannot interpolate this file."
            )
            return None
        
        # Convert quality (0-100) to CRF (15-35)
        crf = int(35 - (output_quality / 100) * 20)
        
        interpolation_factor = 1
        if "2x" in fps_mode: interpolation_factor = 2
        elif "4x" in fps_mode: interpolation_factor = 4
        try:
            max_out_fps = float(os.environ.get("FLASHVSR_MAX_OUT_FPS", "120") or 120)
        except (TypeError, ValueError):
            max_out_fps = 120.0
        src_fps = self._probe_fps(video_path)
        if src_fps >= max(160.0, max_out_fps * 1.25):
            print(
                f"ERROR: {Path(video_path).name} is already {src_fps:.1f} FPS "
                f"(over {max_out_fps:.0f} cap). Skipping RIFE — move to HighFPS."
            )
            return None
        requested = interpolation_factor
        interpolation_factor = self._choose_interp_factor(src_fps, interpolation_factor, max_out_fps)
        if interpolation_factor != requested:
            print(
                f"INFO: Capped interpolation {requested}x → {interpolation_factor}x "
                f"(source {src_fps:.1f} FPS, max out {max_out_fps:.0f}). "
                f"60fps sources stay at 2x (120), never 4x (240)."
            )
        should_interpolate = interpolation_factor > 1

        if not should_interpolate and speed_factor == 1.0:
            print(
                "INFO: No frame interpolation needed "
                f"(source {src_fps:.1f} FPS already at/under cap). Skipping RIFE."
            )
            return video_path

        temp_video_path = None
        try:
            print(f"Adjusting frames: Mode={fps_mode}, Speed={speed_factor}x, Streaming: {use_streaming}, Quality: {output_quality}/100 (CRF {crf})")
            
            # Suppress verbose imageio/ffmpeg warnings about frame rate estimation
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=UserWarning)
                reader = imageio.get_reader(video_path)
                fps = float((reader.get_meta_data() or {}).get("fps") or src_fps or 30)
            if src_fps > 0:
                fps = src_fps
            
            output_fps = fps * interpolation_factor
            print(f"Input: {fps:.2f} FPS → Output: {output_fps:.2f} FPS")
            if use_streaming and speed_factor != 1.0:
                print("Note: Speed adjustment is ignored in RIFE streaming mode.")
                speed_factor = 1.0
            
            if use_streaming and should_interpolate:
                # Stream 2x per pass (true 4x = two streaming passes). Never load all frames into RAM.
                try:
                    reader.close()
                except Exception:
                    pass
                if not self.rife_handler._ensure_model_downloaded_and_loaded():
                    raise gr.Error(
                        "❌ RIFE model failed to download or load. Please check your internet connection "
                        "and try again. If the problem persists, try manually deleting the "
                        "'toolbox/model_rife' folder and restarting."
                    )
                num_passes = int(math.log2(interpolation_factor))
                current_input = video_path
                current_fps = float(fps)
                intermediate_paths = []
                for p in range(num_passes):
                    pass_out = self._generate_output_path(
                        video_path, f"frames_stream_p{p + 1}", is_temp=True
                    )
                    pass_fps = current_fps * 2.0
                    print(
                        f"INFO: Streaming RIFE pass {p + 1}/{num_passes} "
                        f"({current_fps:.2f} → {pass_fps:.2f} FPS)..."
                    )
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=UserWarning)
                        r_in = imageio.get_reader(current_input)
                        w_out = imageio.get_writer(
                            pass_out,
                            fps=pass_fps,
                            macro_block_size=1,
                            # Intermediate only — ultrafast + low CRF; Export does the quality encode once
                            ffmpeg_params=self._intermediate_ffmpeg_params(output_quality),
                        )
                        frame_iterator = iter(r_in)
                        frame1 = next(frame_iterator, None)
                        if frame1 is None:
                            r_in.close()
                            w_out.close()
                            raise RuntimeError("Video has no frames for RIFE streaming")
                        desc = f"RIFE stream pass {p + 1}/{num_passes}"
                        for frame2 in progress.tqdm(frame_iterator, desc=desc):
                            w_out.append_data(frame1)
                            middle = self.rife_handler.interpolate_between_frames(frame1, frame2)
                            if middle is not None:
                                w_out.append_data(middle)
                            else:
                                w_out.append_data(frame1)
                            frame1 = frame2
                        w_out.append_data(frame1)
                        w_out.close()
                        r_in.close()
                    # Drop previous intermediate (keep source video_path)
                    if current_input != video_path and os.path.exists(current_input):
                        try:
                            os.remove(current_input)
                        except OSError:
                            pass
                        if current_input in intermediate_paths:
                            intermediate_paths.remove(current_input)
                    intermediate_paths.append(pass_out)
                    current_input = pass_out
                    current_fps = pass_fps
                temp_video_path = current_input
            else:
                frames = [frame for frame in reader]
                print(f"Loaded {len(frames)} frames from video")
                
                # Check if we have enough frames
                if len(frames) < 2:
                    print("WARNING: Video has less than 2 frames. Cannot interpolate.")
                    reader.close()
                    return video_path
                
                processed_frames = frames
                if speed_factor != 1.0:
                    print(f"Adjusting speed by {speed_factor}x (in-memory)...")
                    new_len = int(len(frames) / speed_factor)
                    indices = np.linspace(0, len(frames) - 1, new_len).astype(int)
                    processed_frames = [frames[i] for i in indices]
                    print(f"Speed adjustment: {len(frames)} → {len(processed_frames)} frames")
                
                if should_interpolate and len(processed_frames) > 1:
                    if not self.rife_handler._ensure_model_downloaded_and_loaded():
                        raise gr.Error("❌ RIFE model failed to download or load. Please check your internet connection and try again. If the problem persists, try manually deleting the 'toolbox/model_rife' folder and restarting.")
                    num_passes = int(math.log2(interpolation_factor))
                    for p in range(num_passes):
                        print(f"INFO: Starting RIFE interpolation pass {p + 1}/{num_passes}...")
                        print(f"      Input frames: {len(processed_frames)}")
                        interpolated_this_pass = []
                        desc = f"RIFE Pass {p+1}/{num_passes}"
                        frame_iterator = progress.tqdm(range(len(processed_frames) - 1), desc=desc)
                        failed_interpolations = 0
                        for i in frame_iterator:
                            interpolated_this_pass.append(processed_frames[i])
                            middle = self.rife_handler.interpolate_between_frames(processed_frames[i], processed_frames[i+1])
                            if middle is not None:
                                interpolated_this_pass.append(middle)
                            else:
                                interpolated_this_pass.append(processed_frames[i])  # Fallback to original frame
                                failed_interpolations += 1
                        interpolated_this_pass.append(processed_frames[-1])
                        processed_frames = interpolated_this_pass
                        print(f"      Output frames: {len(processed_frames)}")
                        if failed_interpolations > 0:
                            print(f"      WARNING: {failed_interpolations} frame interpolations failed (used fallback frames)")
                temp_video_path = self._generate_output_path(video_path, "frames_temp", is_temp=True)
                print(f"Writing {len(processed_frames)} frames to video...")
                
                # Suppress verbose imageio/ffmpeg warnings during video writing
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    # Intermediate write — keep quality high, spend time on Export only
                    imageio.mimwrite(
                        temp_video_path,
                        processed_frames,
                        fps=output_fps,
                        macro_block_size=1,
                        ffmpeg_params=self._intermediate_ffmpeg_params(output_quality),
                    )
                reader.close()

            # --- Suffix and Final Path Generation ---
            suffix_parts = []
            if should_interpolate: suffix_parts.append(fps_mode.replace(' ', ''))
            if speed_factor != 1.0: suffix_parts.append(f"{speed_factor}x")
            suffix = f"frames_{'_'.join(suffix_parts)}"
            # Stage 2 = interpolation / frame-adjust done (at end of filename)
            stage_tag = STAGE_INTERP if should_interpolate else None
            final_temp_output = self._generate_output_path(
                video_path, suffix, is_temp=True, stage=stage_tag
            )

            # --- CORRECTED AUDIO MUXING LOGIC ---
            if self.has_ffmpeg and self._has_audio_stream(video_path):
                print("Muxing audio into processed video...")
                mux_cmd = [
                    self.ffmpeg_exe, "-y",
                    "-i", str(temp_video_path),
                    "-i", video_path,
                    "-c:v", "copy"
                ]
                
                # Conditionally apply the atempo filter ONLY if speed is changed
                if speed_factor != 1.0:
                    print(f"Applying atempo speed filter: {speed_factor}x")
                    audio_filters = [f"atempo={speed_factor}"]
                    if speed_factor > 2.0: audio_filters = [f"atempo=2.0,atempo={speed_factor/2.0}"]
                    if speed_factor < 0.5: audio_filters = [f"atempo=0.5,atempo={speed_factor/0.5}"]
                    mux_cmd.extend(["-filter:a", ",".join(audio_filters)])
                    mux_cmd.extend(["-c:a", "aac", "-b:a", "192k"]) # Re-encode when filtering
                else:
                    # If just interpolating, copy the audio directly
                    print("Copying original audio without speed change.")
                    mux_cmd.extend(["-c:a", "copy"])

                mux_cmd.extend([
                    "-map", "0:v:0", "-map", "1:a:0?",
                    "-shortest", str(final_temp_output)
                ])
                
                subprocess.run(mux_cmd, check=True, capture_output=True, text=True)
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except OSError:
                        pass
                temp_video_path = None  # owned by final_temp_output now
            else:
                # This block runs if there's no FFmpeg or no original audio
                shutil.move(temp_video_path, final_temp_output)
                temp_video_path = None

            return str(final_temp_output)
        except gr.Error:
            # Re-raise Gradio errors so they can be caught by process_pipeline for console display
            raise
        except Exception as e:
            print(f"Error during frame adjustment: {e}\n{traceback.format_exc()}")
            return None
        finally:
            try:
                self.rife_handler.unload_model()
            except Exception:
                pass
            # Only delete leftover temps on failure (success clears temp_video_path above)
            if temp_video_path and os.path.exists(temp_video_path):
                try:
                    os.remove(temp_video_path)
                except OSError:
                    pass
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def create_loop(self, video_path, loop_type, num_loops, output_quality=90, progress=gr.Progress()):
        """Creates a looped or ping-pong version of the video."""
        if video_path is None:
            print("No input video for loop creation.")
            return None
        if not self.has_ffmpeg:
            print("FFmpeg is required for creating video loops. This operation cannot proceed.")
            return video_path
        if loop_type == "none":
            print("Loop type 'none'. No action.")
            return video_path

        # Normalize input path to prevent filename length issues
        video_path = self._normalize_input_path(video_path)
        
        # Convert quality (0-100) to CRF (15-35)
        crf = int(35 - (output_quality / 100) * 20)
        
        progress(0, desc="Initializing loop creation...")
        resolved_video_path = str(Path(video_path).resolve())
        output_path = self._generate_output_path(
            resolved_video_path, 
            suffix=f"loop_{loop_type}_{num_loops}x",
            is_temp=True
        )
        
        print(f"Creating {loop_type} ({num_loops}x) for {os.path.basename(resolved_video_path)}... Quality: {output_quality}/100 (CRF {crf})")
        
        ping_pong_unit_path = None 
        original_video_has_audio = self._has_audio_stream(resolved_video_path)

        try:
            progress(0.2, desc=f"Preparing {loop_type} loop...")
            if loop_type == "ping-pong":
                ping_pong_unit_path = self._generate_output_path(
                    resolved_video_path, 
                    suffix="pingpong_unit_temp",
                    is_temp=True
                )
                # Create video-only ping-pong unit first
                ffmpeg_pp_unit_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-i", resolved_video_path,
                    "-vf", "split[main][tmp];[tmp]reverse[rev];[main][rev]concat=n=2:v=1:a=0",
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                    "-an", str(ping_pong_unit_path)
                ]
                subprocess.run(ffmpeg_pp_unit_cmd, check=True, capture_output=True, text=True)
                print(f"Created ping-pong unit (video-only): {ping_pong_unit_path}")

                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(num_loops - 1),
                    "-i", str(ping_pong_unit_path)
                ]

                if original_video_has_audio:
                    print("Original video has audio. Will loop audio for ping-pong.")
                    audio_loop_count_for_ffmpeg = (num_loops * 2) - 1
                    ffmpeg_cmd.extend([
                        "-i", resolved_video_path,
                        "-filter_complex", f"[1:a]areverse[areva];[1:a][areva]concat=n=2:v=0:a=1[ppa];[ppa]aloop=loop={num_loops-1}:size=2147483647[a_looped]",
                        "-map", "0:v:0", "-map", "[a_looped]",
                        "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
                        "-c:a", "aac", "-b:a", "192k", "-shortest"
                    ])
                else:
                    print("No audio in original or detection issue. Creating video-only ping-pong loop.")
                    ffmpeg_cmd.extend(["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-an"])

                ffmpeg_cmd.append(str(output_path))

            else:  # Regular 'loop'
                ffmpeg_stream_loop_value = num_loops 
                
                if ffmpeg_stream_loop_value < 0: 
                    ffmpeg_stream_loop_value = 0

                total_plays = ffmpeg_stream_loop_value + 1
                print(f"Regular loop: original video + {ffmpeg_stream_loop_value} additional repeat(s). Total {total_plays} plays.")
                
                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-loglevel", "error",
                    "-stream_loop", str(ffmpeg_stream_loop_value),
                    "-i", resolved_video_path,
                    "-c:v", "libx264", "-crf", str(crf), "-preset", "medium"
                ]
                if original_video_has_audio:
                    print("Original video has audio. Re-encoding to AAC for looped MP4 (if not already AAC).")
                    ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", "-map", "0:v:0", "-map", "0:a:0?"])
                else:
                    print("No audio in original or detection issue. Looped video will be silent.")
                    ffmpeg_cmd.extend(["-an", "-map", "0:v:0"])
                ffmpeg_cmd.append(str(output_path))
            
            print(f"Processing video {loop_type} with FFmpeg...")
            progress(0.5, desc=f"Running FFmpeg for {loop_type}...")
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, errors='ignore')

            progress(1.0, desc=f"{loop_type.capitalize()} loop created successfully.")
            print(f"Loop creation complete: {output_path}")
            return str(output_path)
        except subprocess.CalledProcessError as e_loop:
            print(f"FFmpeg error during {loop_type} creation: {e_loop}")
            if e_loop.stderr:
                print(f"FFmpeg stderr: {e_loop.stderr}")
            progress(1.0, desc=f"Error creating {loop_type}.")
            return None
        except Exception as e:
            print(f"Error creating loop: {e}")
            print(traceback.format_exc())
            progress(1.0, desc="Error creating loop.")
            return None
        finally:
            if ping_pong_unit_path and os.path.exists(ping_pong_unit_path):
                try:
                    os.remove(ping_pong_unit_path)
                except Exception as e_clean_pp:
                    print(f"Could not remove temp ping-pong unit: {e_clean_pp}")
            gc.collect()
            
    def export_video(self, video_path, export_format, quality, max_width, output_name, two_pass=False, progress=gr.Progress()):
        if not video_path: print("No input video to export."); return None
        if not self.has_ffmpeg: print("FFmpeg is required for export."); return None
        
        # Normalize input path to prevent filename length issues
        video_path = self._normalize_input_path(video_path)
        
        print(f"Exporting video to {export_format} with quality {quality} and max width {max_width}px (Two-pass: {two_pass}).")
        try:
            # Determine file extension based on format
            ext_map = {
                "MP4 (H.264)": ".mp4",
                "MP4 (H.265)": ".mp4",
                "WebM (VP9)": ".webm",
                "GIF": ".gif"
            }
            ext = ext_map.get(export_format, ".mp4")
            
            # Prefer clean Step-2 name: <step1>_30fps.ext (approx FPS from source)
            is_temp_save = export_format != "GIF" and not self.autosave_enabled
            if output_name and str(output_name).strip():
                base_name = str(output_name).strip()
                output_path = self._generate_output_path(
                    base_name, "", ext=ext, is_temp=is_temp_save, stage=None
                )
            else:
                fps_est = 30
                try:
                    # imageio may already be imported at module level
                    import imageio as _iio
                    fps_est = float(_iio.get_reader(video_path).get_meta_data().get("fps") or 30)
                except Exception:
                    fps_est = 30
                nice = step2_filename(Path(video_path).stem, fps_est, ext=ext)
                if is_temp_save:
                    os.makedirs(self.temp_dir, exist_ok=True)
                    output_path = self.temp_dir / nice
                else:
                    os.makedirs(self.output_dir, exist_ok=True)
                    output_path = self.output_dir / nice
                    # avoid overwrite
                    if Path(output_path).exists():
                        stem = Path(nice).stem
                        output_path = self.output_dir / f"{stem}_{datetime.now().strftime('%H%M%S')}{ext}"
            if export_format == "GIF": print(f"INFO: GIF format selected. Output will be saved to permanent folder: {output_path}")

            # Common video filter (may be None if already under max_width)
            vf_scale = f"scale='min({max_width},iw)':-2:flags=lanczos"
            
            if export_format == "MP4 (H.264)":
                # CRF range: 0=lossless, 23=default, 51=worst
                # Quality slider: 100%→15 (near-lossless), 50%→23 (default), 0%→35 (low quality)
                crf = int(35 - (quality / 100) * 20)
                vcodec_args, vf_auto, enc_label = self._export_video_args(quality, max_width, video_path)
                vf_use = vf_auto if vf_auto is not None else None
                # two-pass only applies to software x264 bitrate mode
                use_nvenc = "h264_nvenc" in vcodec_args

                if two_pass and not use_nvenc:
                    # Two-pass encoding for better compression efficiency
                    progress(0.2, desc="Encoding pass 1/2 (analyzing)...")
                    pass1_cmd = [
                        self.ffmpeg_exe, "-y", "-i", video_path,
                    ]
                    if vf_use:
                        pass1_cmd.extend(["-vf", vf_use])
                    pass1_cmd.extend([
                        "-c:v", "libx264",
                        "-preset", (self.export_preset or "medium"),
                        "-b:v", f"{self._calculate_target_bitrate(video_path, quality, max_width)}k",
                        "-pass", "1",
                        "-passlogfile", str(self.temp_dir / "ffmpeg2pass"),
                        "-an",
                        "-f", "null",
                        "NUL" if os.name == 'nt' else "/dev/null"
                    ])
                    subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
                    
                    progress(0.5, desc="Encoding pass 2/2 (final)...")
                    pass2_cmd = [
                        self.ffmpeg_exe, "-y", "-i", video_path,
                    ]
                    if vf_use:
                        pass2_cmd.extend(["-vf", vf_use])
                    pass2_cmd.extend([
                        "-c:v", "libx264",
                        "-preset", (self.export_preset or "medium"),
                        "-b:v", f"{self._calculate_target_bitrate(video_path, quality, max_width)}k",
                        "-pass", "2",
                        "-passlogfile", str(self.temp_dir / "ffmpeg2pass"),
                        "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "192k",
                        str(output_path)
                    ])
                    subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
                    
                    # Clean up pass log files
                    for f in self.temp_dir.glob("ffmpeg2pass*"):
                        try: f.unlink()
                        except: pass
                else:
                    # Single-pass: NVENC CQ or libx264 medium (not slow) — same visual quality, much faster
                    ffmpeg_cmd = [
                        self.ffmpeg_exe, "-y", "-i", video_path,
                    ]
                    if vf_use:
                        ffmpeg_cmd.extend(["-vf", vf_use])
                    ffmpeg_cmd.extend(vcodec_args)
                    ffmpeg_cmd.extend([
                        "-c:a", "aac", "-b:a", "192k",
                        str(output_path)
                    ])
                    progress(0.3, desc=f"Encoding {export_format} ({enc_label})...")
                    print(f"INFO: Export encode: {enc_label}" + (f" + scale≤{max_width}" if vf_use else " (no scale)"))
                    try:
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                    except subprocess.CalledProcessError as e_enc:
                        # NVENC can fail on some drivers — fall back to software once
                        if use_nvenc:
                            print(f"WARNING: NVENC failed ({e_enc}); falling back to libx264 medium")
                            self._nvenc_ok = False
                            vcodec_args, vf_use, enc_label = self._export_video_args(quality, max_width, video_path)
                            ffmpeg_cmd = [self.ffmpeg_exe, "-y", "-i", video_path]
                            if vf_use:
                                ffmpeg_cmd.extend(["-vf", vf_use])
                            ffmpeg_cmd.extend(vcodec_args)
                            ffmpeg_cmd.extend(["-c:a", "aac", "-b:a", "192k", str(output_path)])
                            progress(0.4, desc=f"Encoding {export_format} ({enc_label})...")
                            subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                        else:
                            raise
                
            elif export_format == "MP4 (H.265)":
                # H.265/HEVC: 30-50% better compression than H.264 at same quality
                crf = int(35 - (quality / 100) * 20)
                
                if two_pass:
                    progress(0.2, desc="Encoding pass 1/2 (analyzing)...")
                    pass1_cmd = [
                        self.ffmpeg_exe, "-y", "-i", video_path,
                        "-vf", vf_scale,
                        "-c:v", "libx265",
                        "-preset", "slow",
                        "-b:v", f"{self._calculate_target_bitrate(video_path, quality, max_width, hevc=True)}k",
                        "-x265-params", f"pass=1:log-level=error",
                        "-an",
                        "-f", "null",
                        "NUL" if os.name == 'nt' else "/dev/null"
                    ]
                    subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
                    
                    progress(0.5, desc="Encoding pass 2/2 (final)...")
                    pass2_cmd = [
                        self.ffmpeg_exe, "-y", "-i", video_path,
                        "-vf", vf_scale,
                        "-c:v", "libx265",
                        "-preset", "slow",
                        "-b:v", f"{self._calculate_target_bitrate(video_path, quality, max_width, hevc=True)}k",
                        "-x265-params", f"pass=2:log-level=error",
                        "-pix_fmt", "yuv420p",
                        "-tag:v", "hvc1",
                        "-c:a", "aac", "-b:a", "96k",
                        str(output_path)
                    ]
                    subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
                    
                    # Clean up x265 log files
                    for f in Path.cwd().glob("x265_*pass.log*"):
                        try: f.unlink()
                        except: pass
                else:
                    ffmpeg_cmd = [
                        self.ffmpeg_exe, "-y", "-i", video_path,
                        "-vf", vf_scale,
                        "-c:v", "libx265",
                        "-preset", "slow",
                        "-crf", str(crf),
                        "-pix_fmt", "yuv420p",
                        "-tag:v", "hvc1",
                        "-c:a", "aac", "-b:a", "96k",
                        str(output_path)
                    ]
                    progress(0.3, desc=f"Encoding {export_format}...")
                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                
            elif export_format == "WebM (VP9)":
                # VP9: Two-pass is highly recommended for VP9
                crf = int(45 - (quality / 100) * 25)
                target_bitrate = self._calculate_target_bitrate(video_path, quality, max_width, vp9=True)
                
                progress(0.2, desc="Encoding pass 1/2 (analyzing)...")
                pass1_cmd = [
                    self.ffmpeg_exe, "-y", "-i", video_path,
                    "-vf", vf_scale,
                    "-c:v", "libvpx-vp9",
                    "-b:v", f"{target_bitrate}k",
                    "-crf", str(crf),
                    "-pass", "1",
                    "-passlogfile", str(self.temp_dir / "ffmpeg2pass"),
                    "-row-mt", "1",
                    "-an",
                    "-f", "null",
                    "NUL" if os.name == 'nt' else "/dev/null"
                ]
                subprocess.run(pass1_cmd, check=True, capture_output=True, text=True)
                
                progress(0.5, desc="Encoding pass 2/2 (final)...")
                pass2_cmd = [
                    self.ffmpeg_exe, "-y", "-i", video_path,
                    "-vf", vf_scale,
                    "-c:v", "libvpx-vp9",
                    "-b:v", f"{target_bitrate}k",
                    "-crf", str(crf),
                    "-pass", "2",
                    "-passlogfile", str(self.temp_dir / "ffmpeg2pass"),
                    "-row-mt", "1",
                    "-c:a", "libopus", "-b:a", "64k",
                    str(output_path)
                ]
                subprocess.run(pass2_cmd, check=True, capture_output=True, text=True)
                
                # Clean up pass log files
                for f in self.temp_dir.glob("ffmpeg2pass*"):
                    try: f.unlink()
                    except: pass
                
            elif export_format == "GIF":
                progress(0.2, desc="Generating GIF palette (Pass 1/2)...")
                palette_path = self.temp_dir / "palette.png"
                palette_cmd = [self.ffmpeg_exe, "-y", "-i", video_path, "-vf", f"{vf_scale},palettegen", str(palette_path)]
                subprocess.run(palette_cmd, check=True, capture_output=True, text=True)
                
                progress(0.5, desc="Encoding GIF (Pass 2/2)...")
                ffmpeg_cmd = [
                    self.ffmpeg_exe, "-y", "-i", video_path, "-i", str(palette_path),
                    "-filter_complex", f"[0:v]{vf_scale}[v];[v][1:v]paletteuse",
                    "-an",
                    str(output_path)
                ]
                subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
                
            progress(1.0, desc="Export complete!")
            
            # Log file size for user feedback
            if os.path.exists(output_path):
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                print(f"Export complete: {output_path} ({size_mb:.2f} MB)")
            
            return str(output_path)
        except subprocess.CalledProcessError as e:
            print(f"ERROR: FFmpeg failed during export to {export_format}.\nCmd: {' '.join(e.cmd)}\nStderr: {e.stderr}"); return video_path
        except Exception as e: print(f"Error during export: {e}\n{traceback.format_exc()}"); return video_path
    
    def _calculate_target_bitrate(self, video_path, quality, max_width, hevc=False, vp9=False):
        """Calculate target bitrate based on resolution and quality for two-pass encoding."""
        try:
            # Get video info
            reader = imageio.get_reader(video_path)
            meta = reader.get_meta_data()
            width = meta.get('size', (1920, 1080))[0]
            height = meta.get('size', (1920, 1080))[1]
            fps = meta.get('fps', 30)
            reader.close()
            
            # Calculate output dimensions
            if width > max_width:
                scale_factor = max_width / width
                output_width = max_width
                output_height = int(height * scale_factor)
            else:
                output_width = width
                output_height = height
            
            # Calculate pixels per frame
            pixels = output_width * output_height
            
            # Base bitrate calculation (bits per pixel per frame)
            # Quality 100 = 0.15 bpp, Quality 50 = 0.08 bpp, Quality 0 = 0.03 bpp
            bpp = 0.03 + (quality / 100) * 0.12
            
            # Adjust for codec efficiency
            if hevc:
                bpp *= 0.6  # H.265 is ~40% more efficient
            elif vp9:
                bpp *= 0.65  # VP9 is ~35% more efficient
            
            # Calculate bitrate in kbps
            bitrate = int((pixels * fps * bpp) / 1000)
            
            # Clamp to reasonable ranges
            min_bitrate = 500
            max_bitrate = 50000
            bitrate = max(min_bitrate, min(bitrate, max_bitrate))
            
            print(f"Calculated target bitrate: {bitrate}k for {output_width}x{output_height} @ {fps}fps")
            return bitrate
            
        except Exception as e:
            print(f"Error calculating bitrate, using default: {e}")
            # Fallback bitrates based on quality
            return int(2000 + (quality / 100) * 8000)

    def process_pipeline(self, input_path, operations, params, progress=gr.Progress()):
        """Processes a single video through a pipeline of operations."""
        current_video_path = input_path
        messages = [f"🚀 Starting pipeline for '{Path(input_path).name}'..."]
        execution_order = ["Frame Adjust", "Video Loop", "Export"]
        for op_name in execution_order:
            if op_name in operations:
                messages.append(f"  -> Starting '{op_name}' step...")
                original_path = current_video_path
                try:
                    if op_name == "Frame Adjust":
                        current_video_path = self.adjust_frames(current_video_path, **params["frame_adjust"], progress=progress)
                    elif op_name == "Video Loop":
                        current_video_path = self.create_loop(current_video_path, **params["loop"], progress=progress)
                    elif op_name == "Export":
                        current_video_path = self.export_video(current_video_path, **params["export"], progress=progress)
                except gr.Error as e:
                    # Capture Gradio errors and add to messages for display in console
                    messages.append(f"❌ {str(e)}")
                    messages.append(f"❌ Operation '{op_name}' failed. Aborting pipeline.")
                    return None, "\n".join(messages)
                if not current_video_path:
                    messages.append(
                        f"❌ Operation '{op_name}' failed (no output — "
                        f"audio-only, already 200+ FPS, or corrupt). Aborting pipeline."
                    )
                    return None, "\n".join(messages)
                if current_video_path == original_path:
                    if op_name == "Frame Adjust":
                        messages.append(
                            "  -> 'Frame Adjust' skipped (source already at/under FPS cap)."
                        )
                        continue
                    messages.append(f"❌ Operation '{op_name}' failed. Aborting pipeline.")
                    return None, "\n".join(messages)
                messages.append(f"  -> '{op_name}' step completed.")
        return current_video_path, "\n".join(messages)

    def process_batch(self, input_paths, operations, params, progress=gr.Progress()):
        """Processes a batch of videos through the pipeline."""
        total_videos, final_video_path = len(input_paths), None
        if total_videos == 0: return None, "No videos provided for batch processing."
        batch_messages = [f"🚀 Starting batch process for {total_videos} videos..."]
        batch_folder_name = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        for i, video_path in enumerate(input_paths):
            progress(i / total_videos, desc=f"Processing video {i+1}/{total_videos}: {os.path.basename(video_path)}")
            batch_messages.append(f"\n--- Video {i+1}/{total_videos}: {os.path.basename(video_path)} ---")
            temp_result_path, messages = self.process_pipeline(video_path, operations, params, progress)
            batch_messages.append(messages)
            if temp_result_path:
                temp_path = Path(temp_result_path)
                final_path = self.output_dir / batch_folder_name / temp_path.name
                os.makedirs(final_path.parent, exist_ok=True)
                final_video_path = self._copy_to_permanent_storage(temp_result_path, final_path)
                batch_messages.append(f"✅ Batch result saved to: {final_path}")
            else: batch_messages.append(f"❌ Pipeline failed for {os.path.basename(video_path)}. Skipping.")
        batch_messages.append("\n--- ✅ Batch processing complete. ---")
        return final_video_path, "\n".join(batch_messages)
