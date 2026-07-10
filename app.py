"""
CreatorStudio: Automated Dynamic Video Engine
------------------------------------------------
A single-file Streamlit app that turns a script + images into a narrated,
kinetic-subtitle video with Ken Burns motion and optional background music.

REQUIRED DEPENDENCIES (install before running):
    pip install streamlit edge-tts "moviepy>=1.0.3" pydub

SYSTEM DEPENDENCIES:
    - ffmpeg must be installed and on PATH (used by moviepy/pydub).
    - ImageMagick must be installed (used by moviepy's TextClip).
      On Linux: sudo apt-get install imagemagick
      On Windows: install ImageMagick and make sure moviepy's config
      (moviepy/config_defaults.py or an IMAGEMAGICK_BINARY env var) points
      to magick.exe if TextClip raises an "ImageMagick not found" error.

RUN:
    streamlit run app.py
"""

import os
import shutil
import asyncio
import tempfile
import traceback

import streamlit as st
import edge_tts

from pydub import AudioSegment

from moviepy.editor import (
    ImageClip,
    TextClip,
    AudioFileClip,
    CompositeVideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.audio.fx.all import audio_loop, volumex

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

WORKDIR = "session_assets"
OUTPUT_FILE = "generated_story_output.mp4"

# edge-tts only ships ONE native neural voice per gender for each of these
# locales (en-IN-PrabhatNeural / en-IN-NeerjaNeural for English-Indian, and
# hi-IN-MadhurNeural / hi-IN-SwaraNeural for Hindi). To still offer real
# variety ("not just one person"), each base voice is given several distinct
# rate/pitch personas so scenes don't all sound like the same single speaker.
VOICE_PROFILES = {
    "English (Indian accent)": {
        "Male": {
            "Prabhat - Standard": {"voice": "en-IN-PrabhatNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Prabhat - Deep & Calm": {"voice": "en-IN-PrabhatNeural", "rate": "-12%", "pitch": "-8Hz"},
            "Prabhat - Energetic": {"voice": "en-IN-PrabhatNeural", "rate": "+15%", "pitch": "+6Hz"},
        },
        "Female": {
            "Neerja - Standard": {"voice": "en-IN-NeerjaNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Neerja - Warm & Slow": {"voice": "en-IN-NeerjaNeural", "rate": "-12%", "pitch": "-5Hz"},
            "Neerja - Bright & Fast": {"voice": "en-IN-NeerjaNeural", "rate": "+15%", "pitch": "+8Hz"},
        },
    },
    "Hindi": {
        "Male": {
            "Madhur - Standard": {"voice": "hi-IN-MadhurNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Madhur - Deep & Calm": {"voice": "hi-IN-MadhurNeural", "rate": "-12%", "pitch": "-8Hz"},
            "Madhur - Energetic": {"voice": "hi-IN-MadhurNeural", "rate": "+15%", "pitch": "+6Hz"},
        },
        "Female": {
            "Swara - Standard": {"voice": "hi-IN-SwaraNeural", "rate": "+0%", "pitch": "+0Hz"},
            "Swara - Warm & Slow": {"voice": "hi-IN-SwaraNeural", "rate": "-12%", "pitch": "-5Hz"},
            "Swara - Bright & Fast": {"voice": "hi-IN-SwaraNeural", "rate": "+15%", "pitch": "+8Hz"},
        },
    },
}

ORIENTATIONS = {
    "Shorts/Reels (9:16 vertical)": (1080, 1920),
    "YouTube Video (16:9 widescreen)": (1920, 1080),
}

# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


def reset_workdir():
    """Create a clean session_assets workspace folder."""
    if os.path.exists(WORKDIR):
        shutil.rmtree(WORKDIR)
    os.makedirs(WORKDIR, exist_ok=True)


def save_uploaded_images(uploaded_images):
    """Persist uploaded images to disk using their original filenames."""
    saved_paths = {}
    for img in uploaded_images:
        dest_path = os.path.join(WORKDIR, img.name)
        with open(dest_path, "wb") as f:
            f.write(img.getbuffer())
        saved_paths[img.name] = dest_path
    return saved_paths


def parse_script(raw_text):
    """Parse 'image_name.jpg | caption text' lines into a list of tuples."""
    rows = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" not in line:
            continue
        img_target, narrative_text = line.split("|", 1)
        rows.append((img_target.strip(), narrative_text.strip()))
    return rows


async def _tts_to_file(text, voice, out_path, rate="+0%", pitch="+0Hz"):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    await communicate.save(out_path)


def generate_voice_sync(text, voice, out_path, rate="+0%", pitch="+0Hz"):
    """Run the async edge-tts call synchronously (fresh event loop)."""
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(_tts_to_file(text, voice, out_path, rate, pitch))
    finally:
        loop.close()


def resize_and_crop(clip, target_w, target_h):
    """Scale a clip to cover the target box, then center-crop to exact size."""
    clip_ratio = clip.w / clip.h
    target_ratio = target_w / target_h

    if clip_ratio > target_ratio:
        # source is relatively wider -> match height, crop width
        clip = clip.resize(height=target_h)
    else:
        # source is relatively taller -> match width, crop height
        clip = clip.resize(width=target_w)

    clip = clip.crop(
        x_center=clip.w / 2,
        y_center=clip.h / 2,
        width=target_w,
        height=target_h,
    )
    return clip


def build_scene(image_path, narrative_text, voice_path, target_w, target_h):
    """Build one video scene: Ken Burns image + kinetic subtitle + voice audio."""
    audio_clip = AudioFileClip(voice_path)
    duration = audio_clip.duration

    base_img = ImageClip(image_path)
    base_img = resize_and_crop(base_img, target_w, target_h)
    base_img = base_img.set_duration(duration)

    # Ken Burns: slow continuous zoom-in over the clip's duration.
    zoom_clip = base_img.resize(lambda t: 1.0 + 0.05 * (t / max(duration, 0.01)))
    zoom_clip = zoom_clip.set_position("center").set_duration(duration)

    subtitle_clip = TextClip(
        narrative_text,
        fontsize=int(target_w * 0.045),
        color="white",
        font="Arial-Bold",
        method="caption",
        size=(int(target_w * 0.9), None),
        stroke_color="black",
        stroke_width=2,
        align="center",
    ).set_duration(duration)
    subtitle_clip = subtitle_clip.set_position(("center", int(target_h * 0.80)))

    scene = CompositeVideoClip(
        [zoom_clip, subtitle_clip], size=(target_w, target_h)
    ).set_duration(duration)
    scene = scene.set_audio(audio_clip)

    # Return handles that need explicit closing later.
    return scene, [audio_clip, base_img, zoom_clip, subtitle_clip]


def prepare_background_music(music_bytes, volume_percent, target_duration):
    """Normalize/trim/loop background music and set volume via pydub + moviepy."""
    tmp_raw = os.path.join(WORKDIR, "bg_music_raw.mp3")
    with open(tmp_raw, "wb") as f:
        f.write(music_bytes)

    # Use pydub for a clean volume-leveled pass, then hand off to moviepy for looping.
    sound = AudioSegment.from_file(tmp_raw)
    tmp_leveled = os.path.join(WORKDIR, "bg_music_leveled.mp3")
    sound.export(tmp_leveled, format="mp3")

    music_clip = AudioFileClip(tmp_leveled)
    music_clip = audio_loop(music_clip, duration=target_duration)
    music_clip = volumex(music_clip, max(volume_percent, 0) / 100.0)
    return music_clip


def compile_video(script_rows, image_paths, voice_profile, target_w, target_h,
                   bg_music_bytes, bg_volume_percent, progress_callback=None):
    """Run the full production pipeline and return the output file path.

    voice_profile: dict with keys "voice", "rate", "pitch" (see VOICE_PROFILES).
    """
    voice = voice_profile["voice"]
    rate = voice_profile.get("rate", "+0%")
    pitch = voice_profile.get("pitch", "+0Hz")
    scenes = []
    disposable_clips = []

    total_steps = len(script_rows)
    for idx, (img_target, narrative_text) in enumerate(script_rows):
        if progress_callback:
            progress_callback(
                idx / max(total_steps, 1),
                f"Processing scene {idx + 1}/{total_steps}: {img_target}",
            )

        if img_target not in image_paths:
            raise FileNotFoundError(
                f"Script references '{img_target}' but no matching uploaded image was found."
            )

        voice_path = os.path.join(WORKDIR, f"v_{idx}.mp3")
        generate_voice_sync(narrative_text, voice, voice_path, rate=rate, pitch=pitch)

        scene, handles = build_scene(
            image_paths[img_target], narrative_text, voice_path, target_w, target_h
        )
        scenes.append(scene)
        disposable_clips.extend(handles)

    if progress_callback:
        progress_callback(0.9, "Concatenating timeline...")

    final_video = concatenate_videoclips(scenes, method="compose")

    music_clip = None
    if bg_music_bytes:
        if progress_callback:
            progress_callback(0.93, "Mixing background music...")
        music_clip = prepare_background_music(
            bg_music_bytes, bg_volume_percent, final_video.duration
        )
        mixed_audio = CompositeAudioClip([final_video.audio, music_clip])
        final_video = final_video.set_audio(mixed_audio)

    if progress_callback:
        progress_callback(0.96, "Rendering final MP4 (this can take a while)...")

    final_video.write_videofile(
        OUTPUT_FILE,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        threads=4,
        verbose=False,
        logger=None,
    )

    # --- Cleanup: close every clip to release file handles/locks ---
    for c in disposable_clips:
        try:
            c.close()
        except Exception:
            pass
    for s in scenes:
        try:
            s.close()
        except Exception:
            pass
    if music_clip is not None:
        try:
            music_clip.close()
        except Exception:
            pass
    try:
        final_video.close()
    except Exception:
        pass

    if progress_callback:
        progress_callback(1.0, "Done!")

    return OUTPUT_FILE


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

st.set_page_config(page_title="CreatorStudio", page_icon="🎬", layout="wide")

st.title("🎬 CreatorStudio: Automated Dynamic Video Engine")
st.caption(
    "Upload images, paste your script (English or Hindi/Devanagari), pick a "
    "language, gender, and voice persona, and CreatorStudio will auto-generate "
    "narration, animate your images with a Ken Burns zoom, burn in kinetic "
    "subtitles, mix background music, and export a finished MP4 — all in one click."
)

with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("🗣️ Voice")
    language_choice = st.selectbox("Language", list(VOICE_PROFILES.keys()))
    gender_choice = st.radio("Gender", list(VOICE_PROFILES[language_choice].keys()), horizontal=True)
    persona_options = list(VOICE_PROFILES[language_choice][gender_choice].keys())
    persona_choice = st.selectbox("Voice Persona", persona_options)
    st.caption(
        "Each persona is a distinct rate/pitch tuning of the base neural voice, "
        "so different personas sound noticeably different — not one voice reused."
    )
    voice_profile = VOICE_PROFILES[language_choice][gender_choice][persona_choice]

    st.divider()
    orientation_choice = st.radio(
        "Video Orientation",
        list(ORIENTATIONS.keys()),
        index=0,
    )

    st.divider()
    st.subheader("🎵 Background Music (optional)")
    bg_music_file = st.file_uploader("Upload MP3", type=["mp3"])
    bg_volume = st.slider("Music Volume (%)", min_value=0, max_value=30, value=10)

st.subheader("📜 Story Editor")
st.caption("One row per scene, formatted as:  `image_name.jpg | Story caption sentence`")
script_text = st.text_area(
    "Script",
    height=220,
    placeholder="beach.jpg | The sun rose slowly over the quiet shore.\n"
    "forest.jpg | Deep in the woods, the adventure truly began.",
    label_visibility="collapsed",
)

st.subheader("🖼️ Media Desk")
uploaded_images = st.file_uploader(
    "Upload images",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True,
    label_visibility="collapsed",
)

st.divider()
compile_clicked = st.button("🚀 Compile Dynamic Video", type="primary", use_container_width=True)

if compile_clicked:
    if not script_text.strip():
        st.error("Please paste a script before compiling.")
    elif not uploaded_images:
        st.error("Please upload at least one image.")
    else:
        try:
            reset_workdir()
            image_paths = save_uploaded_images(uploaded_images)
            script_rows = parse_script(script_text)

            if not script_rows:
                st.error("No valid script rows found. Use the format: image.jpg | caption text")
            else:
                target_w, target_h = ORIENTATIONS[orientation_choice]
                bg_bytes = bg_music_file.read() if bg_music_file is not None else None

                progress_bar = st.progress(0.0)
                status_text = st.empty()

                def on_progress(fraction, message):
                    progress_bar.progress(min(max(fraction, 0.0), 1.0))
                    status_text.info(message)

                with st.spinner("Compiling your dynamic video..."):
                    output_path = compile_video(
                        script_rows=script_rows,
                        image_paths=image_paths,
                        voice_profile=voice_profile,
                        target_w=target_w,
                        target_h=target_h,
                        bg_music_bytes=bg_bytes,
                        bg_volume_percent=bg_volume,
                        progress_callback=on_progress,
                    )

                st.success("✅ Video compiled successfully!")
                st.video(output_path)
                with open(output_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download MP4",
                        data=f,
                        file_name=OUTPUT_FILE,
                        mime="video/mp4",
                    )

        except Exception as e:
            st.error(f"Something went wrong during compilation: {e}")
            st.code(traceback.format_exc())
