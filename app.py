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
import shutil
import random
import asyncio
import traceback
import requests

# fal_client is optional — only needed if the user turns on AI Full Motion.
# The rest of the app works fine without it installed.
try:
    import fal_client
    FAL_CLIENT_AVAILABLE = True
except ModuleNotFoundError:
    FAL_CLIENT_AVAILABLE = False

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
        VideoFileClip,
        CompositeVideoClip,
        concatenate_videoclips,
    )
except ModuleNotFoundError:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        VideoFileClip,
        CompositeVideoClip,
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


def _resized(clip, factor):
    """resized() on MoviePy 2.x, resize() on 1.x. `factor` can be a number or a function of t."""
    if hasattr(clip, "resized"):
        return clip.resized(factor)
    return clip.resize(factor)


def _positioned(clip, pos):
    """with_position() on MoviePy 2.x, set_position() on 1.x. `pos` can be a tuple or a function of t."""
    if hasattr(clip, "with_position"):
        return clip.with_position(pos)
    return clip.set_position(pos)


KEN_BURNS_EFFECTS = ["zoom_in", "zoom_out", "pan_left", "pan_right", "pan_up", "pan_down"]


def apply_ken_burns(image_clip, duration, effect=None, zoom_ratio=1.18):
    """
    Wrap a static ImageClip with a slow zoom and/or pan over its lifetime so it
    reads as motion instead of a frozen photo. Returns a same-sized clip
    (image_clip.size) ready to have duration/audio attached.
    """
    w, h = image_clip.size
    effect = effect or random.choice(KEN_BURNS_EFFECTS)

    def scale_at(t):
        progress = min(max(t / duration, 0), 1) if duration > 0 else 0
        if effect == "zoom_in":
            return 1 + (zoom_ratio - 1) * progress
        elif effect == "zoom_out":
            return zoom_ratio - (zoom_ratio - 1) * progress
        else:
            # Panning effects keep a constant zoom so there's room to move around in.
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
        else:  # zoom_in / zoom_out stay centered
            x, y = -max_x / 2, -max_y / 2
        return (x, y)

    moving_clip = _positioned(moving_clip, pos_at)
    framed_clip = CompositeVideoClip([moving_clip], size=(w, h))
    return framed_clip


# --------------------------------------------------------------------------
# AI Full Motion pipeline (fal.ai)
# --------------------------------------------------------------------------
# Two open-source models, chained together:
#   1. LTX-2.3 (image-to-video)  -> animates the photo: body/hand/background motion
#   2. LatentSync (video-to-video) -> re-syncs the mouth in that video to your narration audio
#
# Requires: pip install fal-client
#           a FAL_KEY (from https://fal.ai/dashboard/keys) set as an env var
#           or entered in the sidebar in the app.
#
# Cost is pay-per-use (no subscription) — roughly a few cents to ~$0.50 per
# picture depending on length/resolution. Check https://fal.ai/pricing for
# current rates before generating a large batch.

LTX_DURATION_BUCKETS = [6, 8, 10]  # seconds — the only durations LTX-2.3 accepts


def _pick_duration_bucket(target_seconds: float) -> str:
    """LTX-2.3 only generates 6s/8s/10s clips. Pick the closest bucket that's
    long enough to cover the narration (or the longest bucket if narration
    runs longer than 10s)."""
    for bucket in LTX_DURATION_BUCKETS:
        if target_seconds <= bucket:
            return str(bucket)
    return str(LTX_DURATION_BUCKETS[-1])


def generate_ai_motion_clip(image_path: str, audio_path: str, motion_prompt: str,
                             output_path: str, fal_key: str = None,
                             status_callback=None) -> str:
    """
    Runs the two-stage fal.ai pipeline for one (image, audio) pair and saves
    the final lip-synced, full-motion clip to output_path. Returns output_path.
    Raises on any failure — caller decides how to fall back.
    """
    if not FAL_CLIENT_AVAILABLE:
        raise RuntimeError("fal-client is not installed. Run: pip install fal-client")

    if fal_key:
        os.environ["FAL_KEY"] = fal_key
    if not os.environ.get("FAL_KEY"):
        raise RuntimeError("No FAL_KEY set. Get one at https://fal.ai/dashboard/keys")

    def report(msg):
        if status_callback:
            status_callback(msg)

    # Rough narration duration, just to pick a sensible LTX clip length.
    with AudioFileClip(audio_path) as probe:
        target_seconds = probe.duration
    duration_bucket = _pick_duration_bucket(target_seconds)

    # ---- Stage 1: animate the still photo (LTX-2.3 image-to-video) ----
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

    # ---- Stage 2: sync lips to the narration audio (LatentSync) ----
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

    # ---- Download the final clip locally so MoviePy can stitch it in ----
    report("Downloading generated clip...")
    response = requests.get(final_video_url, timeout=120)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)

    return output_path

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


def build_video(story_items, audio_paths, progress_callback=None, motion_effect="random",
                 motion_prompts=None, fal_key=None):
    """
    Build the final video by pairing each image with its corresponding audio clip.
    motion_effect:
      "none"      -> static image (original behavior)
      "random"/"zoom_in"/etc. -> Ken Burns pan/zoom (camera motion only)
      "ai_motion" -> full AI-generated motion + lip-sync via fal.ai (LTX-2.3 + LatentSync)
                     Falls back to Ken Burns for any picture where generation fails,
                     so one bad/slow API call doesn't kill the whole video.
    Returns the path to the exported video file.
    """
    video_segments = []
    audio_clips_to_close = []
    image_clips_to_close = []
    ai_generated_paths = []

    total = len(story_items)
    try:
        for index, (image_path, _) in enumerate(story_items):
            audio_path = audio_paths[index]
            audio_clip = AudioFileClip(audio_path)
            audio_clips_to_close.append(audio_clip)

            duration = audio_clip.duration
            segment_clip = None

            if motion_effect == "ai_motion":
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
                    # LatentSync already carries the narration audio in its output,
                    # so use the clip's own audio rather than re-attaching ours.
                    segment_clip = video_clip
                except Exception as ai_error:
                    if progress_callback:
                        progress_callback(
                            (index + 1) / total,
                            f"⚠️ AI motion failed for picture {index + 1} ({ai_error}); using Ken Burns instead...",
                        )
                    segment_clip = None  # fall through to Ken Burns below

            if segment_clip is None:
                image_clip = ImageClip(image_path)
                if motion_effect not in ("none", "ai_motion"):
                    effect = None if motion_effect == "random" else motion_effect
                    image_clip = apply_ken_burns(image_clip, duration, effect=effect)
                elif motion_effect == "ai_motion":
                    # AI motion failed for this one — still give it camera motion
                    # rather than a flat static frame.
                    image_clip = apply_ken_burns(image_clip, duration, effect=None)

                image_clip = _with_duration(image_clip, duration)
                image_clip = _with_audio(image_clip, audio_clip)
                image_clips_to_close.append(image_clip)
                segment_clip = image_clip

            video_segments.append(segment_clip)

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
# Password Gate
# --------------------------------------------------------------------------
def check_password() -> bool:
    """
    Simple shared-password gate for a publicly deployed app. Set APP_PASSWORD
    in Streamlit secrets (Settings -> Secrets):
        APP_PASSWORD = "your-password-here"
    Locally, you can instead set it as an environment variable of the same name.
    Returns True once the correct password has been entered for this session.
    """
    correct_password = None
    if hasattr(st, "secrets"):
        correct_password = st.secrets.get("APP_PASSWORD")
    if not correct_password:
        correct_password = os.environ.get("APP_PASSWORD")

    # If no password is configured anywhere, don't lock people out — just warn.
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

    st.subheader("4️⃣ Motion Effect")
    st.caption("Adds motion to each picture so the video feels alive instead of a static slideshow.")
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
            "AI Full Motion — lips, hands, body (requires fal.ai API key, paid per use)",
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
        "AI Full Motion — lips, hands, body (requires fal.ai API key, paid per use)": "ai_motion",
        "None (static, original behavior)": "none",
    }
    selected_motion_effect = motion_effect_map[motion_choice]

    fal_key_input = ""
    motion_prompts = {}
    if selected_motion_effect == "ai_motion":
        with st.expander("⚙️ AI Full Motion setup", expanded=True):
            st.markdown(
                "This mode sends each picture + narration to **fal.ai**, which runs two "
                "open-source models: **LTX-2.3** (animates hands/body/background) and "
                "**LatentSync** (syncs the mouth to your narration). It costs a small "
                "amount per picture (check current rates at fal.ai/pricing) — nothing "
                "runs on your laptop, no GPU needed on your end."
            )
            if not FAL_CLIENT_AVAILABLE:
                st.error("Missing dependency. Run: `pip install fal-client` and restart the app.")

            secret_key = st.secrets.get("FAL_KEY") if hasattr(st, "secrets") else None
            if secret_key:
                fal_key_input = secret_key
                st.success("Using fal.ai API key from app secrets ✅")
            else:
                fal_key_input = st.text_input(
                    "fal.ai API key (from fal.ai/dashboard/keys)",
                    type="password",
                    help=(
                        "For a deployed app, set this as FAL_KEY in Streamlit secrets instead "
                        "so visitors don't need to paste a key. This box is a fallback for "
                        "local runs or when no secret is configured."
                    ),
                )

            st.caption(
                "Optional: describe the motion you want for each picture below "
                "(e.g. 'she waves and smiles, leaves rustling in the background'). "
                "Leave blank for a sensible default."
            )
            if uploaded_images:
                for index, uploaded_file in enumerate(uploaded_images):
                    motion_prompts[index] = st.text_input(
                        f"Motion for picture {index + 1} ({uploaded_file.name})",
                        key=f"motion_prompt_{index}",
                        placeholder="e.g. gentle hand wave, background trees swaying",
                    )

    st.subheader("5️⃣ Generate")
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

        if selected_motion_effect == "ai_motion":
            if not FAL_CLIENT_AVAILABLE:
                st.error("❌ AI Full Motion needs the fal-client package. Run: pip install fal-client")
                return
            if not fal_key_input and not os.environ.get("FAL_KEY"):
                st.error("❌ AI Full Motion needs a fal.ai API key. Paste one in the setup box above.")
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
                output_path = build_video(
                    story_items,
                    audio_paths,
                    video_progress_callback,
                    motion_effect=selected_motion_effect,
                    motion_prompts=motion_prompts,
                    fal_key=fal_key_input,
                )

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
