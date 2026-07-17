"""
🎬 Automated Story-to-Video Generator
--------------------------------------
A single-file Streamlit web app. Upload your pictures, write (or paste) the
matching story text for each one right next to it, pick a voice, and get a
fully narrated video — all from one page.

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit edge-tts moviepy imageio-ffmpeg requests
    pip install fal-client   # only needed for the AI Full Motion option
"""
import os
import re
import shutil
import random
import asyncio
import traceback
import requests

# fal_client is optional — only needed if the user turns on AI Full Motion.
try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
except ModuleNotFoundError:
    FAL_CLIENT_AVAILABLE = False

import imageio_ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

import streamlit as st
import edge_tts

# Handle MoviePy 1.x vs 2.x imports
try:
    from moviepy.editor import (
        ImageClip,
        AudioFileClip,
        AudioClip,
        VideoFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
    )
except ModuleNotFoundError:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        AudioClip,
        VideoFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
    )

# MoviePy 1.x vs 2.x method compatibility helpers
def _with_duration(clip, duration):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)

def _with_audio(clip, audio_clip):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)

def _resized(clip, factor):
    if hasattr(clip, "resized"):
        return clip.resized(factor)
    return clip.resize(factor)

def _positioned(clip, pos):
    if hasattr(clip, "with_position"):
        return clip.with_position(pos)
    return clip.set_position(pos)

def _with_start(clip, start_time):
    if hasattr(clip, "with_start"):
        return clip.with_start(start_time)
    return clip.set_start(start_time)

def _with_volume(clip, factor):
    if hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    return clip.volumex(factor)

def _subclip(clip, start, end):
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


DIALOGUE_PAUSE_SECONDS = 0.35
TEMP_DIR = "temp_assets"
OUTPUT_FILENAME = "final_output.mp4"

def _silence(duration_seconds, fps=44100):
    return AudioClip(lambda t: 0, duration=duration_seconds, fps=fps)

KEN_BURNS_EFFECTS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]

def apply_ken_burns(image_clip, duration, effect=None, zoom_ratio=1.18):
    w, h = image_clip.size
    effect = effect or random.choice(KEN_BURNS_EFFECTS)
    def scale_at(t):
        progress = min(max(t / duration, 0), 1) if duration > 0 else 0
        if effect == "zoom_in":
            return 1 + (zoom_ratio - 1) * progress
        elif effect == "zoom_out":
            return zoom_ratio - (zoom_ratio - 1) * progress
        else:
            return zoom_ratio
    moving_clip = _resized(image_clip, scale_at)
    def pos_at(t):
        progress = min(max(t / duration, 0), 1) if duration > 0 else 0
        cur_scale = scale_at(t)
        cur_w, cur_h = w * cur_scale, h * cur_scale
        max_x = cur_w - w
        max_y = cur_h - h
        if effect == "pan_left":
            x, y = -max_x * progress, -max_y / 2
        elif effect == "pan_right":
            x, y = -max_x * (1 - progress), -max_y / 2
        elif effect == "pan_up":
            x, y = -max_x / 2, -max_y * progress
        elif effect == "pan_down":
            x, y = -max_x / 2, -max_y * (1 - progress)
        else:
            x, y = -max_x / 2, -max_y / 2
        return (x, y)
    moving_clip = _positioned(moving_clip, pos_at)
    framed_clip = CompositeVideoClip([moving_clip], size=(w, h))
    return framed_clip

# --------------------------------------------------------------------------
# AI Full Motion pipeline (fal.ai)
# --------------------------------------------------------------------------
LTX_DURATION_BUCKETS = [6, 8, 10]

def _pick_duration_bucket(target_seconds: float) -> str:
    for bucket in LTX_DURATION_BUCKETS:
        if target_seconds <= bucket:
            return str(bucket)
    return str(LTX_DURATION_BUCKETS[-1])

def generate_ai_motion_clip(image_path: str, audio_path: str, motion_prompt: str,
                            output_path: str, fal_key: str = None,
                            status_callback=None) -> str:
    if not FAL_CLIENT_AVAILABLE:
        raise RuntimeError("fal-client is not installed. Run: pip install fal-client")
    if fal_key:
        os.environ["FAL_KEY"] = fal_key
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("No FAL_KEY set. Get one at https://fal.ai/dashboard/keys")
    
    def report(msg):
        if status_callback:
            status_callback(msg)

    with AudioFileClip(audio_path) as probe:
        target_seconds = probe.duration
    duration_bucket = _pick_duration_bucket(target_seconds)

    report("Uploading photo...")
    image_url = fal_client.upload_file(image_path)
    report("Generating motion (hands/body/background)...")
    motion_result = fal_client.subscribe(
        "fal-ai/ltx-2.3/image-to-video",
        arguments={
            "image_url": image_url,
            "prompt": motion_prompt or "subtle natural motion, gentle hand and body movement, slight background movement, realistic",
            "duration": duration_bucket,
        },
        with_logs=False,
    )
    motion_video_url = motion_result["video"]["url"]

    report("Uploading narration audio...")
    audio_url = fal_client.upload_file(audio_path)
    report("Syncing lips to narration...")
    lipsync_result = fal_client.subscribe(
        "fal-ai/latentsync",
        arguments={
            "video_url": motion_video_url,
            "audio_url": audio_url,
        },
        with_logs=False,
    )
    final_video_url = lipsync_result["video"]["url"]

    report("Downloading generated clip...")
    response = requests.get(final_video_url, timeout=120)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
    return output_path

# --------------------------------------------------------------------------
# Sound library (SFX / BGM) 
# --------------------------------------------------------------------------
SOUND_LIBRARY_DIR = "sound_library"
SFX_DIR = os.path.join(SOUND_LIBRARY_DIR, "sfx")
BGM_DIR = os.path.join(SOUND_LIBRARY_DIR, "bgm")
SOUND_FILE_EXTENSIONS = (".mp3", ".wav", ".ogg", ".m4a")

SFX_MAX_SECONDS = 4
BGM_VOLUME = 0.22
SFX_VOLUME = 0.9

SOUND_KEYWORD_MAP = {
    r"(हंसने|हंसा|मजाक|ठिठोली|खिलखिला)": "[sfx:laugh]",
    r"(सांप|नाग|नागिन|फुंकार|डसने)": "[sfx:hiss]",
    r"(बिजली|तूफान|बादल|गर्जना|कड़क)": "[sfx:thunder]",
    r"(अचानक|चौंक|तभी|एकदम|पलक झपकते)": "[sfx:whoosh]",
    r"(हवा|सन्नाटा|अंधेरा|जंगल|शमशान)": "[sfx:wind]",
    r"(सोचा|बुद्धि|विचार|आइडिया|तरकीब)": "[sfx:ding]",
    r"(रोने|रोया|आंसू|सिसकने|विलाप|रोना)": "[sfx:crying]",
    r"(डर|कांप|सहमा|खौफ|भयानक|भूत)": "[sfx:fear]",
    r"(हांफने|हांफा|सांस फूल|थक)": "[sfx:panting]",
    r"(शेर|दहाड़|सिंह|वनराज)": "[sfx:lion_roar]",
    r"(कुत्ता|भोंकने|भौ-भौ|श्वान)": "[sfx:dog_bark]",
    r"(बिल्ली|म्याऊ|म्यॉंऊ)": "[sfx:cat_meow]",
    r"(भेड़िया|हुआँ|चीख)": "[sfx:wolf_howl]",
    r"(दर्द|कराहा|चोट|आह|उफ्)": "[sfx:pain_groan]",
    r"(प्यार|मोहब्बत|सुंदर|रूप|खूबसूरत|रोमांटिक)": "[bgm:love]",
    r"(भगवान|शिव|मंदिर|पूजा|प्रार्थना|भक्ति|आशीर्वाद)": "[bgm:devotional]",
    r"(रहस्य|राज|सस्पेंस|छुपा|खोज)": "[bgm:suspense]",
}

FALLBACK_VOICE_OPTIONS = {
    "Male (English-India)": "en-IN-PrabhatNeural",
    "Female (English-India)": "en-IN-NeerjaNeural",
    "Male (Hindi)": "hi-IN-MadhurNeural",
    "Female (Hindi)": "hi-IN-SwaraNeural",
}

def _parse_sound_tag(tag_str: str):
    m = re.match(r"\[(sfx|bgm):(\w+)\]", tag_str)
    return (m.group(1), m.group(2)) if m else (None, None)

def find_sound_cues(text: str):
    cues = []
    for pattern, tag in SOUND_KEYWORD_MAP.items():
        kind, name = _parse_sound_tag(tag)
        if not kind:
            continue
        for m in re.finditer(pattern, text):
            cues.append({
                "start": m.start(),
                "match": m.group(0),
                "kind": kind,
                "name": name,
                "tag": tag,
            })
    cues.sort(key=lambda c: c["start"])
    return cues

def sound_file_path(kind: str, name: str):
    folder = SFX_DIR if kind == "sfx" else BGM_DIR
    for ext in SOUND_FILE_EXTENSIONS:
        candidate = os.path.join(folder, name + ext)
        if os.path.exists(candidate):
            return candidate
    return None

def all_known_sound_tags():
    seen = []
    for tag in SOUND_KEYWORD_MAP.values():
        kind, name = _parse_sound_tag(tag)
        if kind and (kind, name) not in seen:
            seen.append((kind, name))
    return seen

def list_sound_library():
    found = {}
    for kind, name in all_known_sound_tags():
        path = sound_file_path(kind, name)
        if path:
            found[(kind, name)] = path
    return found

def save_sound_file(uploaded_file, kind: str, name: str):
    folder = SFX_DIR if kind == "sfx" else BGM_DIR
    os.makedirs(folder, exist_ok=True)
    for ext in SOUND_FILE_EXTENSIONS:
        old_path = os.path.join(folder, name + ext)
        if os.path.exists(old_path):
            os.remove(old_path)
    ext = os.path.splitext(uploaded_file.name)[1].lower() or ".mp3"
    if ext not in SOUND_FILE_EXTENSIONS:
        ext = ".mp3"
    dest_path = os.path.join(folder, name + ext)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path

# --------------------------------------------------------------------------
# Core Pipeline Processing Logic
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_voices():
    try:
        all_voices = asyncio.run(edge_tts.list_voices())
    except Exception:
        return dict(FALLBACK_VOICE_OPTIONS)
    wanted_locales = {"hi-IN", "en-IN"}
    filtered = [
        v for v in all_voices
        if v.get("Locale") in wanted_locales or "Multilingual" in v.get("ShortName", "")
    ]
    if not filtered:
        return dict(FALLBACK_VOICE_OPTIONS)
    
    def _sort_key(v):
        is_hindi_native = v["Locale"] != "hi-IN"
        is_multilingual = "Multilingual" not in v["ShortName"]
        return (is_hindi_native, is_multilingual, v.get("Gender", ""), v["ShortName"])
    filtered.sort(key=_sort_key)
    options = {}
    for v in filtered:
        short_name = v["ShortName"]
        persona = short_name.split("-")[-1].replace("Neural", "").replace("Multilingual", "")
        if v["Locale"] == "hi-IN":
            lang_label = "Hindi"
        elif "Multilingual" in short_name:
            lang_label = "Multilingual — speaks Hindi"
        else:
            lang_label = "English-India"
        label = f"{v.get('Gender', '')} ({lang_label}) — {persona}"
        options[label] = short_name
    return options or dict(FALLBACK_VOICE_OPTIONS)

def setup_workspace():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}
def is_video_file(path_or_name: str) -> bool:
    return os.path.splitext(path_or_name)[1].lower() in VIDEO_EXTENSIONS

def save_uploaded_media(uploaded_file):
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

async def generate_audio_file(text: str, voice: str, output_path: str,
                              rate: str = "+0%", pitch: str = "+0Hz"):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    boundaries = []
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                boundaries.append({
                    "audio_time": chunk["offset"] / 10_000_000,
                    "duration": chunk["duration"] / 10_000_000,
                    "text": chunk["text"],
                })
    return boundaries

def _align_cues_to_audio(text: str, boundaries: list, cues: list):
    word_spans = []
    search_pos = 0
    for b in boundaries:
        idx = text.find(b["text"], search_pos)
        if idx == -1:
            idx = search_pos
        word_spans.append((idx, b["audio_time"]))
        search_pos = idx + max(1, len(b["text"]))
    
    def time_for_char(char_index):
        chosen = None
        for start, t in word_spans:
            if start <= char_index:
                chosen = t
            else:
                break
        if chosen is not None:
            return chosen
        return (char_index / len(text)) * boundaries[-1]["audio_time"] if text and boundaries else 0.0
    
    for cue in cues:
        cue["audio_time"] = time_for_char(cue["start"])
    return cues

def mix_scene_audio(narration_path: str, cues: list, output_path: str, scene_duration: float):
    narration_clip = AudioFileClip(narration_path)
    layers = [narration_clip]
    extra_clips = []
    for cue in cues:
        path = sound_file_path(cue["kind"], cue["name"])
        if not path:
            continue
        try:
            raw_clip = AudioFileClip(path)
        except Exception:
            continue
        start_t = min(cue.get("audio_time", 0.0), max(0.0, scene_duration - 0.1))
        if cue["kind"] == "sfx":
            clip = _subclip(raw_clip, 0, min(SFX_MAX_SECONDS, raw_clip.duration))
            clip = _with_volume(clip, SFX_VOLUME)
            clip = _with_start(clip, start_t)
        else:
            remain = max(0.5, scene_duration - start_t)
            bgm_clip = raw_clip
            if bgm_clip.duration < remain:
                loops_needed = int(remain // bgm_clip.duration) + 1
                bgm_clip = concatenate_audioclips([raw_clip] * loops_needed)
            clip = _subclip(bgm_clip, 0, remain)
            clip = _with_volume(clip, BGM_VOLUME)
            clip = _with_start(clip, start_t)
        layers.append(clip)
        extra_clips.append(raw_clip)
        
    if len(layers) == 1:
        narration_clip.close()
        return narration_path
    
    composite = CompositeAudioClip(layers)
    composite = _with_duration(composite, scene_duration)
    composite.write_audiofile(output_path, fps=44100, logger=None)
    composite.close()
    narration_clip.close()
    for c in extra_clips:
        try:
            c.close()
        except Exception:
            pass
    return output_path

async def generate_all_audio(story_items, progress_callback=None):
    audio_paths = []
    scene_cues = []
    total = len(story_items)
    for index, (_, dialogue_lines) in enumerate(story_items):
        line_clips = []
        combined_cues = []
        cumulative_offset = 0.0

        for line_index, (line_text, line_voice, rate, pitch) in enumerate(dialogue_lines):
            line_path = os.path.join(TEMP_DIR, f"audio_{index}_line{line_index}.mp3")
            boundaries = await generate_audio_file(line_text, line_voice, line_path, rate=rate, pitch=pitch)

            line_cues = find_sound_cues(line_text)
            _align_cues_to_audio(line_text, boundaries, line_cues)
            for cue in line_cues:
                cue["audio_time"] = cue.get("audio_time", 0.0) + cumulative_offset
            combined_cues.extend(line_cues)

            line_clip = AudioFileClip(line_path)
            line_clips.append(line_clip)
            cumulative_offset += line_clip.duration

            if line_index < len(dialogue_lines) - 1:
                pause_clip = _silence(DIALOGUE_PAUSE_SECONDS)
                line_clips.append(pause_clip)
                cumulative_offset += DIALOGUE_PAUSE_SECONDS

        raw_scene_clip = concatenate_audioclips(line_clips)
        raw_scene_path = os.path.join(TEMP_DIR, f"audio_raw_{index}.mp3")
        raw_scene_clip.write_audiofile(raw_scene_path, fps=44100, logger=None)
        scene_duration = raw_scene_clip.duration
        raw_scene_clip.close()
        for c in line_clips:
            try:
                c.close()
            except Exception:
                pass

        mixed_path = os.path.join(TEMP_DIR, f"audio_{index}.mp3")
        final_path = mix_scene_audio(raw_scene_path, combined_cues, mixed_path, scene_duration)
        audio_paths.append(final_path)
        scene_cues.append(combined_cues)

        if progress_callback:
            progress_callback((index + 1) / total, f"Generating audio {index + 1}/{total}...")
    return audio_paths, scene_cues

# --------------------------------------------------------------------------
# Uniform Dimensions Video Generation Fix
# --------------------------------------------------------------------------
def build_video(story_items, audio_paths, progress_callback=None, motion_effect="random",
                motion_prompts=None, fal_key=None):
    video_segments = []
    audio_clips_to_close = []
    image_clips_to_close = []
    ai_generated_paths = []
    total = len(story_items)
    
    # Force a standard uniform resolution canvas (Standard Full HD Widescreen)
    TARGET_SIZE = (1920, 1080)  
    
    try:
        for index, (image_path, _) in enumerate(story_items):
            audio_path = audio_paths[index]
            audio_clip = AudioFileClip(audio_path)
            audio_clips_to_close.append(audio_clip)
            duration = audio_clip.duration
            segment_clip = None
            
            if is_video_file(image_path):
                raw_clip = VideoFileClip(image_path)
                image_clips_to_close.append(raw_clip)

                if raw_clip.duration <= duration:
                    loops_needed = int(duration // raw_clip.duration) + 1
                    looped_clip = concatenate_videoclips([raw_clip] * loops_needed, method="compose")
                    image_clips_to_close.append(looped_clip)
                    segment_clip = _subclip(looped_clip, 0, duration)
                    segment_clip = _with_audio(segment_clip, audio_clip)
                else:
                    scene_duration = raw_clip.duration
                    narration_layer = _with_start(audio_clip, 0)
                    audio_layers = [narration_layer]
                    tail_duration = scene_duration - duration
                    if raw_clip.audio is not None and tail_duration > 0.05:
                        tail_audio = _subclip(raw_clip.audio, duration, scene_duration)
                        tail_audio = _with_start(tail_audio, duration)
                        audio_layers.append(tail_audio)
                    combined_audio = (
                        CompositeAudioClip(audio_layers) if len(audio_layers) > 1 else narration_layer
                    )
                    combined_audio = _with_duration(combined_audio, scene_duration)
                    segment_clip = _with_audio(raw_clip, combined_audio)

                image_clips_to_close.append(segment_clip)

            elif motion_effect == "ai_motion":
                try:
                    if progress_callback:
                        progress_callback(
                            (index) / total,
                            f"Generating AI motion for picture {index + 1}/{total} (this can take a minute)...",
                        )
                    prompt = (motion_prompts or {}).get(index, "")
                    clip_path = os.path.join(TEMP_DIR, f"ai_motion_{index}.mp4")
                    generate_ai_motion_clip(
                        image_path, audio_path, prompt, clip_path, fal_key=fal_key,
                        status_callback=lambda msg: progress_callback(
                            (index) / total, f"Picture {index + 1}/{total}: {msg}"
                        ) if progress_callback else None,
                    )
                    ai_generated_paths.append(clip_path)
                    video_clip = VideoFileClip(clip_path)
                    image_clips_to_close.append(video_clip)
                    segment_clip = video_clip
                except Exception as ai_error:
                    if progress_callback:
                        progress_callback(
                            (index + 1) / total,
                            f"⚠️ AI motion failed for picture {index + 1} ({ai_error}); using Ken Burns instead...",
                        )
                    segment_clip = None

            if segment_clip is None:
                image_clip = ImageClip(image_path)
                if motion_effect not in ("none", "ai_motion"):
                    effect = None if motion_effect == "random" else motion_effect
                    image_clip = apply_ken_burns(image_clip, duration, effect=effect)
                elif motion_effect == "ai_motion":
                    image_clip = apply_ken_burns(image_clip, duration, effect=None)
                image_clip = _with_duration(image_clip, duration)
                image_clip = _with_audio(image_clip, audio_clip)
                image_clips_to_close.append(image_clip)
                segment_clip = image_clip

            # Normalize dimensions by resizing and cropping the segment down to target dimensions
            if hasattr(segment_clip, "cropped"):
                segment_clip = segment_clip.cropped(width=TARGET_SIZE[0], height=TARGET_SIZE[1])
            elif hasattr(segment_clip, "crop"):
                segment_clip = segment_clip.crop(x_center=segment_clip.w / 2, y_center=segment_clip.h / 2,
                                                 width=TARGET_SIZE[0], height=TARGET_SIZE[1])
            else:
                segment_clip = _resized(segment_clip, TARGET_SIZE)

            video_segments.append(segment_clip)
            if progress_callback:
                progress_callback(
                    (index + 1) / total, f"Assembling segment {index + 1}/{total}..."
                )

        final_video = concatenate_videoclips(video_segments, method="compose")
        output_path = os.path.join(os.getcwd(), OUTPUT_FILENAME)
        try:
            final_video.write_videofile(
                output_path,
                fps=24,
                audio_codec="aac",
                codec="libx264",
                threads=4,
                verbose=False,
                logger=None,
            )
        except TypeError:
            final_video.write_videofile(
                output_path,
                fps=24,
                audio_codec="aac",
                codec="libx264",
                threads=4,
                logger=None,
            )
        final_video.close()
        return output_path
    finally:
        for clip in image_clips_to_close:
            try:
                clip.close()
            except Exception:
                pass
        for clip in audio_clips_to_close:
            try:
                clip.close()
            except Exception:
                pass

# --------------------------------------------------------------------------
# Password Gate
# --------------------------------------------------------------------------
def check_password() -> bool:
    correct_password = None
    if hasattr(st, "secrets"):
        correct_password = st.secrets.get("APP_PASSWORD")
    if not correct_password:
        correct_password = os.environ.get("APP_PASSWORD")
    if not correct_password:
        st.warning(
            "⚠️ No APP_PASSWORD is configured, so this app is currently open to "
            "anyone with the link. Set APP_PASSWORD in Streamlit secrets to lock it."
        )
        return True
    if st.session_state.get("authenticated", False):
        return True
    st.title("🔒 Story-to-Video Generator")
    st.caption("This app is password protected.")
    password_attempt = st.text_input("Enter password", type="password", key="password_attempt")
    submit = st.button("Unlock")
    if submit:
        if password_attempt == correct_password:
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("❌ Incorrect password.")
    return False

# --------------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Story-to-Video Generator", page_icon="🎬", layout="centered")
    if not check_password():
        return
    st.title("🎬 Automated Story-to-Video Generator")
    st.caption(
        "Upload your pictures, write the story for each one right below it, "
        "pick a voice, and generate a fully narrated video — all in one place."
    )
    st.divider()

    st.subheader("1️⃣ Upload Your Pictures or Video Clips")
    uploaded_images = st.file_uploader(
        "Upload images/videos (order below = order in the video)",
        type=["jpg", "jpeg", "png", "mp4", "mov", "m4v", "webm"],
        accept_multiple_files=True,
    )
    voice_options = get_available_voices()
    voice_labels = list(voice_options.keys())

    st.subheader("2️⃣ Sound Library — Upload SFX & Music (optional)")
    with st.expander("🎵 Upload / manage sound files", expanded=False):
        library_now = list_sound_library()
        known_tags = all_known_sound_tags()
        if library_now:
            saved_labels = ", ".join(f"{k}:{n}" for (k, n) in library_now.keys())
            st.success(f"✅ Currently saved: {saved_labels}")
        else:
            st.info("No sound files saved yet — everything below will be auto-detected but silent.")
        sound_uploads = st.file_uploader(
            "Choose one or more audio files (mp3/wav/ogg/m4a)",
            type=["mp3", "wav", "ogg", "m4a"],
            accept_multiple_files=True,
            key="sound_uploads",
        )
        if sound_uploads:
            tag_display = [f"{'🔊' if k == 'sfx' else '🎵'} {k}:{n}" for k, n in known_tags]
            for u_index, up_file in enumerate(sound_uploads):
                stem = os.path.splitext(up_file.name)[0].lower()
                guessed_index = next((i for i, (k, n) in enumerate(known_tags) if n == stem), 0)
                chosen_label = st.selectbox(
                    f"'{up_file.name}' is the sound for:",
                    options=tag_display,
                    index=guessed_index,
                    key=f"sound_tag_choice_{u_index}",
                )
                st.session_state[f"sound_tag_kind_{u_index}"] = known_tags[tag_display.index(chosen_label)][0]
                st.session_state[f"sound_tag_name_{u_index}"] = known_tags[tag_display.index(chosen_label)][1]
            if st.button("💾 Save uploaded sounds to library"):
                saved = []
                for u_index, up_file in enumerate(sound_uploads):
                    kind = st.session_state.get(f"sound_tag_kind_{u_index}")
                    name = st.session_state.get(f"sound_tag_name_{u_index}")
                    if kind and name:
                        save_sound_file(up_file, kind, name)
                        saved.append(f"{kind}:{name}")
                if saved:
                    st.success("Saved: " + ", ".join(saved) + " — reload page to refresh summary library visualization.")

    st.subheader("3️⃣ Default Voice")
    default_voice_label = st.selectbox("Default voice", options=voice_labels, key="default_voice_label")
    selected_voice = voice_options[default_voice_label]
    default_voice_index = voice_labels.index(default_voice_label)

    if uploaded_images:
        st.subheader("4️⃣ Write the Story — Voice & Sounds Auto-Detect Per Scene")
        with st.expander("✏️ Optional: paste the whole story at once (split by blank lines)"):
            bulk_text = st.text_area(
                "Paste your full story here — separate each picture's sentence with a blank line",
                height=150,
                placeholder="This is the first sentence of my story.\n\nThen, an unexpected event took place.",
                key="bulk_text",
            )
            apply_bulk = st.button("Apply to pictures below")
            if apply_bulk and bulk_text.strip():
                chunks = [c.strip() for c in bulk_text.split("\n\n") if c.strip()]
                for i, chunk in enumerate(chunks[: len(uploaded_images)]):
                    st.session_state[f"story_text_{i}_0"] = chunk

        VOICE_STYLE_PRESETS = {
            "Natural (no change)": (0, 0),
            "Deep & Slow (serious/dramatic)": (-20, -15),
            "Young & Energetic": (15, 10),
            "Warm & Soft": (-10, 5),
            "Bright & Fast (excited)": (15, 15),
        }
        
        def _apply_voice_preset(scene_index, line_index):
            preset_name = st.session_state.get(f"story_style_preset_{scene_index}_{line_index}")
            if preset_name in VOICE_STYLE_PRESETS:
                rate, pitch = VOICE_STYLE_PRESETS[preset_name]
                st.session_state[f"story_rate_{scene_index}_{line_index}"] = rate
                st.session_state[f"story_pitch_{scene_index}_{line_index}"] = pitch

        for index, uploaded_file in enumerate(uploaded_images):
            col1, col2 = st.columns([1, 2])
            with col1:
                if is_video_file(uploaded_file.name):
                    st.video(uploaded_file)
                    st.caption(f"🎬 {uploaded_file.name} (video clip)")
                else:
                    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            with col2:
                num_lines = st.session_state.get(f"num_lines_{index}", 1)
                st.caption(f"**Scene {index + 1}** — {num_lines} dialogue line" + f"{'s' if num_lines != 1 else ''}.")
                for line_i in range(num_lines):
                    line_cols = st.columns([2, 1])
                    with line_cols[0]:
                        st.text_area(
                            f"Line {line_i + 1} — scene {index + 1}",
                            key=f"story_text_{index}_{line_i}",
                            height=90,
                            placeholder=("e.g. राम ने कहा, 'चलो चलते हैं'" if line_i > 0 or num_lines > 1 else "Write story sentence here..."),
                        )
                    with line_cols[1]:
                        st.selectbox(
                            f"Voice — line {line_i + 1}",
                            options=voice_labels,
                            index=default_voice_index,
                            key=f"story_voice_{index}_{line_i}",
                        )
                        with st.expander("🎙️ Style", expanded=False):
                            st.selectbox(
                                "Preset",
                                options=list(VOICE_STYLE_PRESETS.keys()),
                                key=f"story_style_preset_{index}_{line_i}",
                                on_change=_apply_voice_preset,
                                args=(index, line_i),
                            )
                            st.slider("Rate", -50, 50, 0, step=5, key=f"story_rate_{index}_{line_i}")
                            st.slider("Pitch", -50, 50, 0, step=5, key=f"story_pitch_{index}_{line_i}")

                btn_cols = st.columns(2)
                with btn_cols[0]:
                    if st.button(f"➕ Add line", key=f"add_line_{index}"):
                        st.session_state[f"num_lines_{index}"] = num_lines + 1
                        st.rerun()
                with btn_cols[1]:
                    if num_lines > 1 and st.button(f"➖ Remove last line", key=f"remove_line_{index}"):
                        last = num_lines - 1
                        for suffix in ("story_text", "story_voice", "story_style_preset", "story_rate", "story_pitch"):
                            st.session_state.pop(f"{suffix}_{index}_{last}", None)
                        st.session_state[f"num_lines_{index}"] = last
                        st.rerun()

                combined_scene_text = " ".join(st.session_state.get(f"story_text_{index}_{li}", "") for li in range(num_lines))
                scene_cues_preview = find_sound_cues(combined_scene_text) if combined_scene_text.strip() else []
                if scene_cues_preview:
                    bits = []
                    seen = set()
                    for cue in scene_cues_preview:
                        key = (cue["kind"], cue["name"])
                        if key in seen:
                            continue
                        seen.add(key)
                        icon = "🎵" if cue["kind"] == "bgm" else "🔊"
                        has_file = sound_file_path(cue["kind"], cue["name"]) is not None
                        bits.append(f"{icon} {cue['name']}" if has_file else f"{icon} {cue['name']} ⚠️ no file")
                    st.caption("Auto-detected sounds: " + " · ".join(bits))
            st.divider()

    st.subheader("5️⃣ Motion Effect")
    motion_choice = st.selectbox(
        "Motion style",
        options=[
            "Random (different per picture)",
            "Zoom In",
            "Zoom Out",
            "Pan Left",
            "Pan Right",
            "Pan Up",
            "Pan Down",
            "AI Full Motion — lips, hands, body (requires fal.ai API key)",
            "None (static, original behavior)",
        ],
    )
    motion_effect_map = {
        "Random (different per picture)": "random",
        "Zoom In": "zoom_in",
        "Zoom Out": "zoom_out",
        "Pan Left": "pan_left",
        "Pan Right": "pan_right",
        "Pan Up": "pan_up",
        "Pan Down": "pan_down",
        "AI Full Motion — lips, hands, body (requires fal.ai API key)": "ai_motion",
        "None (static, original behavior)": "none",
    }
    selected_motion_effect = motion_effect_map[motion_choice]
    fal_key_input = ""
    motion_prompts = {}
    if selected_motion_effect == "ai_motion":
        with st.expander("⚙️ AI Full Motion setup", expanded=True):
            if not FAL_CLIENT_AVAILABLE:
                st.error("Missing dependency. Run: `pip install fal-client` and restart the app.")
            secret_key = st.secrets.get("FAL_KEY") if hasattr(st, "secrets") else None
            if secret_key:
                fal_key_input = secret_key
                st.success("Using fal.ai API key from app secrets ✅")
            else:
                fal_key_input = st.text_input("fal.ai API key", type="password")
            if uploaded_images:
                for index, uploaded_file in enumerate(uploaded_images):
                    if is_video_file(uploaded_file.name):
                        continue
                    motion_prompts[index] = st.text_input(
                        f"Motion for picture {index + 1} ({uploaded_file.name})",
                        key=f"motion_prompt_{index}",
                    )

    st.subheader("6️⃣ Generate")
    generate_clicked = st.button("🚀 Generate Video", type="primary", use_container_width=True)
    if generate_clicked:
        if not uploaded_images:
            st.error("❌ Please upload at least one picture before generating the video.")
            return
        missing_text_indexes = [
            i for i in range(len(uploaded_images))
            if not any(st.session_state.get(f"story_text_{i}_{li}", "").strip() for li in range(st.session_state.get(f"num_lines_{i}", 1)))
        ]
        if missing_text_indexes:
            st.error(f"❌ Please add at least one dialogue line for all scenes.")
            return
        if selected_motion_effect == "ai_motion":
            if not FAL_CLIENT_AVAILABLE or (not fal_key_input and not os.environ.get("FAL_KEY")):
                st.error("❌ Setup check failure for AI Full Motion config variables.")
                return

        try:
            with st.spinner("Setting up workspace..."):
                setup_workspace()
                story_items = []
                for index, uploaded_file in enumerate(uploaded_images):
                    image_path = save_uploaded_media(uploaded_file)
                    num_lines = st.session_state.get(f"num_lines_{index}", 1)
                    dialogue_lines = []
                    for line_i in range(num_lines):
                        line_text = st.session_state.get(f"story_text_{index}_{line_i}", "").strip()
                        if not line_text:
                            continue
                        line_voice_label = st.session_state.get(f"story_voice_{index}_{line_i}", default_voice_label)
                        line_voice = voice_options.get(line_voice_label, selected_voice)
                        rate_pct = st.session_state.get(f"story_rate_{index}_{line_i}", 0)
                        pitch_hz = st.session_state.get(f"story_pitch_{index}_{line_i}", 0)
                        dialogue_lines.append((line_text, line_voice, f"{rate_pct:+d}%", f"{pitch_hz:+d}Hz"))
                    story_items.append((image_path, dialogue_lines))

            audio_progress = st.progress(0, text="Starting audio generation...")
            audio_paths, scene_cues = asyncio.run(generate_all_audio(story_items, lambda f, m: audio_progress.progress(f, text=m)))
            audio_progress.progress(1.0, text="Audio generation complete ✅")

            all_cues = [c for cues in scene_cues for c in cues]
            if all_cues:
                found_list = sorted({f"{c['kind']}:{c['name']}" for c in all_cues if sound_file_path(c["kind"], c["name"]) is not None})
                if found_list:
                    st.info("🔊 Sounds mixed in: " + ", ".join(found_list))

            video_progress = st.progress(0, text="Starting video assembly...")
            with st.spinner("Compiling final video... this may take a moment."):
                output_path = build_video(
                    story_items, audio_paths, lambda f, m: video_progress.progress(f, text=m),
                    motion_effect=selected_motion_effect, motion_prompts=motion_prompts, fal_key=fal_key_input
                )
            video_progress.progress(1.0, text="Video assembly complete ✅")

            st.success("✅ Video generated successfully!")
            st.video(output_path)
            with open(output_path, "rb") as f:
                st.download_button("⬇️ Download Video", data=f, file_name=OUTPUT_FILENAME, mime="video/mp4", use_container_width=True)
        except Exception as e:
            st.error(f"❌ Something went wrong while generating the video: {e}")
            with st.expander("Show detailed error traceback"):
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
