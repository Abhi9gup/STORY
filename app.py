"""
🎬 Automated Story-to-Video Generator
--------------------------------------
A single-file Streamlit web app. Upload your pictures, write (or paste) the
matching story text for each one right next to it, pick a voice, and get a
fully narrated video — all from one page.

Run with:
    streamlit run app.py

Dependencies:
    pip install streamlit edge-tts moviepy
"""

import os
import shutil
import asyncio
import traceback

# Ensure MoviePy can find a working ffmpeg binary even if it isn't on the
# system PATH (e.g. on locked-down office laptops where PATH editing or
# direct ffmpeg downloads are blocked by security policy). imageio-ffmpeg
# ships its own ffmpeg binary as part of the normal pip install.
import imageio_ffmpeg
os.environ["IMAGEIO_FFMPEG_EXE"] = imageio_ffmpeg.get_ffmpeg_exe()
os.environ["FFMPEG_BINARY"] = imageio_ffmpeg.get_ffmpeg_exe()

import streamlit as st
import edge_tts

# MoviePy 2.x removed the `moviepy.editor` module — everything now imports
# directly from `moviepy`. This try/except keeps the app working whether
# you have MoviePy 1.x or 2.x installed.
try:
    from moviepy.editor import (
        ImageClip,
        AudioFileClip,
        concatenate_videoclips,
    )
except ModuleNotFoundError:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        concatenate_videoclips,
    )

# MoviePy 2.x renamed several clip methods (set_duration -> with_duration,
# set_audio -> with_audio). These small helpers call whichever one exists,
# so the app works on both MoviePy 1.x and 2.x.
def _with_duration(clip, duration):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _with_audio(clip, audio_clip):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
TEMP_DIR = "temp_assets"
OUTPUT_FILENAME = "final_output.mp4"

VOICE_OPTIONS = {
    "Male (English-India)": "en-IN-PrabhatNeural",
    "Female (English-India)": "en-IN-NeerjaNeural",
    "Male (Hindi)": "hi-IN-MadhurNeural",
    "Female (Hindi)": "hi-IN-SwaraNeural",
}


# --------------------------------------------------------------------------
# Helper Functions
# --------------------------------------------------------------------------
def setup_workspace():
    """Create a fresh temp_assets directory, wiping any previous run."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)


def save_uploaded_image(uploaded_file):
    """Persist a single uploaded image to the temp workspace and return its path."""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


async def generate_audio_file(text: str, voice: str, output_path: str):
    """Generate a single TTS audio file asynchronously using edge-tts."""
    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(output_path)


async def generate_all_audio(story_items, voice: str, progress_callback=None):
    """
    Generate TTS audio sequentially for every (image_path, text) pair.
    Returns a list of audio file paths aligned with story_items order.
    """
    audio_paths = []
    total = len(story_items)
    for index, (_, text) in enumerate(story_items):
        audio_path = os.path.join(TEMP_DIR, f"audio_{index}.mp3")
        await generate_audio_file(text, voice, audio_path)
        audio_paths.append(audio_path)
        if progress_callback:
            progress_callback((index + 1) / total, f"Generating audio {index + 1}/{total}...")
    return audio_paths


def build_video(story_items, audio_paths, progress_callback=None):
    """
    Build the final video by pairing each image with its corresponding audio clip.
    Returns the path to the exported video file.
    """
    video_segments = []
    audio_clips_to_close = []
    image_clips_to_close = []

    total = len(story_items)
    try:
        for index, (image_path, _) in enumerate(story_items):
            audio_path = audio_paths[index]
            audio_clip = AudioFileClip(audio_path)
            audio_clips_to_close.append(audio_clip)

            duration = audio_clip.duration

            image_clip = ImageClip(image_path)
            image_clip = _with_duration(image_clip, duration)
            image_clip = _with_audio(image_clip, audio_clip)
            image_clips_to_close.append(image_clip)

            video_segments.append(image_clip)

            if progress_callback:
                progress_callback(
                    (index + 1) / total, f"Assembling segment {index + 1}/{total}..."
                )

        final_video = concatenate_videoclips(video_segments, method="compose")
        output_path = os.path.join(os.getcwd(), OUTPUT_FILENAME)
        try:
            # MoviePy 1.x accepts `verbose`; MoviePy 2.x removed it.
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
        # Explicitly close all clips to release file handles and avoid
        # locking/memory issues on Windows/Linux.
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
# Streamlit UI
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Story-to-Video Generator", page_icon="🎬", layout="centered")
    st.title("🎬 Automated Story-to-Video Generator")
    st.caption(
        "Upload your pictures, write the story for each one right below it, "
        "pick a voice, and generate a fully narrated video — all in one place."
    )

    st.divider()

    # ---------------- Step 1: Upload Images ----------------
    st.subheader("1️⃣ Upload Your Pictures")
    uploaded_images = st.file_uploader(
        "Upload images (order below = order in the video)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
    )

    if uploaded_images:
        st.subheader("2️⃣ Write (or Paste) the Story for Each Picture")

        # Optional convenience: bulk-paste a whole script at once, split by blank lines,
        # and auto-fill each picture's text box in order.
        with st.expander("✏️ Optional: paste the whole story at once (split by blank lines)"):
            bulk_text = st.text_area(
                "Paste your full story here — separate each picture's sentence with a blank line",
                height=150,
                placeholder=(
                    "This is the first sentence of my story.\n\n"
                    "Then, an unexpected event took place."
                ),
                key="bulk_text",
            )
            apply_bulk = st.button("Apply to pictures below")
            if apply_bulk and bulk_text.strip():
                chunks = [c.strip() for c in bulk_text.split("\n\n") if c.strip()]
                for i, chunk in enumerate(chunks[: len(uploaded_images)]):
                    st.session_state[f"story_text_{i}"] = chunk

        # One row per image: thumbnail + its own text area, right next to each other.
        for index, uploaded_file in enumerate(uploaded_images):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            with col2:
                st.text_area(
                    f"Story text for picture {index + 1}",
                    key=f"story_text_{index}",
                    height=120,
                    placeholder="Write or paste the sentence that goes with this picture...",
                )
            st.divider()

    st.subheader("3️⃣ Choose a Voice")
    voice_label = st.selectbox("Select Voice", options=list(VOICE_OPTIONS.keys()))
    selected_voice = VOICE_OPTIONS[voice_label]

    st.subheader("4️⃣ Generate")
    generate_clicked = st.button("🚀 Generate Video", type="primary", use_container_width=True)

    if generate_clicked:
        # ---------------- Validation ----------------
        if not uploaded_images:
            st.error("❌ Please upload at least one picture before generating the video.")
            return

        missing_text_indexes = [
            i for i in range(len(uploaded_images))
            if not st.session_state.get(f"story_text_{i}", "").strip()
        ]
        if missing_text_indexes:
            missing_names = ", ".join(
                uploaded_images[i].name for i in missing_text_indexes
            )
            st.error(f"❌ Please add story text for: {missing_names}")
            return

        try:
            with st.spinner("Setting up workspace..."):
                setup_workspace()
                story_items = []
                for index, uploaded_file in enumerate(uploaded_images):
                    image_path = save_uploaded_image(uploaded_file)
                    text = st.session_state[f"story_text_{index}"].strip()
                    story_items.append((image_path, text))

            # ---------------- Audio Generation ----------------
            audio_progress = st.progress(0, text="Starting audio generation...")

            def audio_progress_callback(fraction, message):
                audio_progress.progress(fraction, text=message)

            audio_paths = asyncio.run(
                generate_all_audio(story_items, selected_voice, audio_progress_callback)
            )
            audio_progress.progress(1.0, text="Audio generation complete ✅")

            # ---------------- Video Compilation ----------------
            video_progress = st.progress(0, text="Starting video assembly...")

            def video_progress_callback(fraction, message):
                video_progress.progress(fraction, text=message)

            with st.spinner("Compiling final video... this may take a moment."):
                output_path = build_video(story_items, audio_paths, video_progress_callback)

            video_progress.progress(1.0, text="Video assembly complete ✅")

            # ---------------- Output ----------------
            st.success("✅ Video generated successfully!")
            st.video(output_path)

            with open(output_path, "rb") as f:
                st.download_button(
                    "⬇️ Download Video",
                    data=f,
                    file_name=OUTPUT_FILENAME,
                    mime="video/mp4",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"❌ Something went wrong while generating the video: {e}")
            with st.expander("Show detailed error traceback"):
                st.code(traceback.format_exc())


if __name__ == "__main__":
    main()