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
import numpy as np

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
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
    )
except ModuleNotFoundError:
    from moviepy import (
        ImageClip,
        AudioFileClip,
        VideoFileClip,
        CompositeVideoClip,
        CompositeAudioClip,
        concatenate_videoclips,
        concatenate_audioclips,
    )

# MoviePy 2.x renamed several clip methods (set_duration -> with_duration,
# set_audio -> with_audio). These small helpers call whichever one exists,
# so the app works on both MoviePy 1.x and 2.x.
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


def _with_start(clip, start_time):
    """with_start() on MoviePy 2.x, set_start() on 1.x."""
    if hasattr(clip, "with_start"):
        return clip.with_start(start_time)
    return clip.set_start(start_time)


def _with_volume(clip, factor):
    """with_volume_scaled() on MoviePy 2.x, volumex() on 1.x."""
    if hasattr(clip, "with_volume_scaled"):
        return clip.with_volume_scaled(factor)
    return clip.volumex(factor)


def _with_duration(clip, duration):
    """with_duration() on MoviePy 2.x, set_duration() on 1.x."""
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _with_audio(clip, audio_clip):
    """with_audio() on MoviePy 2.x, set_audio() on 1.x."""
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)


def _subclip(clip, start, end):
    """subclipped() on MoviePy 2.x, subclip() on 1.x."""
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def _pad_audio_with_silence(audio_clip, target_duration):
    """Extend an audio clip to target_duration by concatenating real silence at the end.
    If audio is already longer than target_duration, trim it."""
    if audio_clip.duration >= target_duration:
        return _subclip(audio_clip, 0, target_duration)
    
    silence_duration = target_duration - audio_clip.duration
    
    try:
        # Try MoviePy 2.x method
        from moviepy.audio.AudioClip import AudioClip
        silence = AudioClip.make_silence(silence_duration, fps=44100, nchannels=2)
    except (TypeError, AttributeError):
        # Fall back to 1.x method with lambda
        from moviepy.audio.AudioClip import AudioClip
        silence = AudioClip(
            lambda t: np.zeros((2,)),
            duration=silence_duration,
            fps=44100
        )
    
    padded = concatenate_audioclips([audio_clip, silence])
    return padded


def _video_audio_with_dynamic_volume(video_audio, narration_duration, total_duration, ducked_volume):
    """Split video audio: ducked during narration (0 to narration_duration), 
    full volume after (narration_duration to total_duration)."""
    if narration_duration >= total_duration:
        # Narration covers whole video — keep ducked throughout
        return _with_volume(video_audio, ducked_volume)
    
    # Trim to total_duration first
    video_trimmed = _subclip(video_audio, 0, total_duration)
    
    # Split into two parts
    narration_part = _subclip(video_trimmed, 0, narration_duration)
    remaining_part = _subclip(video_trimmed, narration_duration, total_duration)
    
    # Apply volumes and set start times
    narration_part_ducked = _with_volume(narration_part, ducked_volume)
    narration_part_ducked = _with_start(narration_part_ducked, 0)
    
    remaining_part_full = _with_volume(remaining_part, 1.0)
    remaining_part_full = _with_start(remaining_part_full, narration_duration)
    
    # Composite and set final duration
    result = CompositeAudioClip([narration_part_ducked, remaining_part_full])
    result = _with_duration(result, total_duration)
    
    return result


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

# Fallback list used only if the live edge-tts voice catalog can't be fetched
# (e.g. no internet at import time). get_available_voices() below normally
# replaces this with the full, current set of Hindi + English-India voices.
FALLBACK_VOICE_OPTIONS = {
    "Male (English-India)": "en-IN-PrabhatNeural",
    "Female (English-India)": "en-IN-NeerjaNeural",
    "Male (Hindi)": "hi-IN-MadhurNeural",
    "Female (Hindi)": "hi-IN-SwaraNeural",
}

# --------------------------------------------------------------------------
# Sound library (SFX / BGM) — auto-detected from the story text
# --------------------------------------------------------------------------
# Drop your own audio files here (they are NOT wiped between runs, unlike
# temp_assets/):
#   sound_library/sfx/<name>.mp3   e.g. sound_library/sfx/laugh.mp3
#   sound_library/bgm/<name>.mp3   e.g. sound_library/bgm/love.mp3
# Supported extensions: .mp3 .wav .ogg .m4a
# The <name> must match the tag after the colon, e.g. "[sfx:laugh]" -> laugh.*
SOUND_LIBRARY_DIR = "sound_library"
SFX_DIR = os.path.join(SOUND_LIBRARY_DIR, "sfx")
BGM_DIR = os.path.join(SOUND_LIBRARY_DIR, "bgm")
SOUND_FILE_EXTENSIONS = (".mp3", ".wav", ".ogg", ".m4a")

# How SFX vs BGM cues behave once detected in the text:
SFX_MAX_SECONDS = 4          # one-shot stings are trimmed to this length
BGM_VOLUME = 0.22            # BGM plays quietly under the narration
SFX_VOLUME = 0.9             # SFX plays near-full volume as a short accent

# When a scene is an uploaded VIDEO clip that already has its own sound:
# duck the clip's original audio down and slightly boost the narration on
# top of it, so both are audible but the story text stays the clear focus.
ORIGINAL_VIDEO_AUDIO_DEFAULT_VOLUME = 0.3
NARRATION_VOLUME_WHEN_MIXED = 1.15

# Hindi/Hinglish keyword -> sound-cue tag map for Naagin series
SOUND_KEYWORD_MAP = {
    # ============ TEMPLE & SPIRITUAL ============
    r"(मंदिर|मंदिर|पूजा|प्रार्थना|भगवान|देवता|महादेव|शिव|आशीर्वाद|पवित्र|पूजनीय|भक्ति)": "[sfx:temple_bells]",
    r"(घंटी|घंटी|ध्वनि|शंख|शंखनाद|ॐ|ओम)": "[sfx:shankh_conch]",
    r"(आत्मा|आत्मा|आध्यात्मिक|दिव्य|पवित्र|पवित्रता|देवी)": "[sfx:sacred_humming]",
    r"(तांपूरा|तांपूरा|ध्रुव्य|सुर|संगीत)": "[sfx:tanpura_drone]",
    
    # ============ MYSTERY ============
    r"(गुप्त|रहस्य|छुपा|राज|रहस्य|गूढ़|अज्ञात|भेद|छिपा)": "[sfx:mystery_whoosh]",
    r"(गुफा|गुफा|गुप्त द्वार|दरवाज़ा|रहस्य द्वार|पुरातन)": "[sfx:stone_grinding]",
    r"(भविष्यवाणी|भविष्य|भविष्यद्वाणी|पूर्वाभास|इशारा)": "[sfx:magical_echo]",
    r"(प्राचीन|अति प्राचीन|पुरातत्व|ऐतिहासिक|पुरानी)": "[sfx:ancient_hum]",
    
    # ============ NAAG POWER ============
    r"(नाग|नागिन|साँप|सर्प|फुंकार|डसना|जहर|विष|काटना)": "[sfx:snake_hiss]",
    r"(शक्ति|नाग शक्ति|नागिन शक्ति|जागृत|जाग गई|शक्तिशाली|शक्तिमान)": "[sfx:energy_pulse]",
    r"(नागमणि|हीरा|रत्न|चमकना|दीप्ति|प्रकाश|गहरा रंग)": "[sfx:emerald_glow]",
    r"(ऊर्जा|शक्ति|चेतना|जीवन|प्रवाह|लहर|तरंग)": "[sfx:magical_energy]",
    
    # ============ NAAGLOK ============
    r"(नागलोक|नाग दुनिया|नाग राज्य|स्वर्ग|दिव्य लोक|दूसरी दुनिया|अन्य लोक|परलोक)": "[sfx:mystical_choir]",
    r"(क्रिस्टल|क्रिस्टल्स|पारदर्शी|चमकदार|दीप्तिमान|खनिज|रत्न)": "[sfx:crystal_ambience]",
    r"(जलप्रपात|झरना|पानी|जल|बहना|प्रवाहित|बहती)": "[sfx:waterfall]",
    r"(दिव्य|पवित्र|स्वर्गीय|ईश्वरीय|देवी|देव|परमात्मा)": "[sfx:divine_ambience]",
    
    # ============ KULGURU ============
    r"(कुलगुरु|गुरु|शिक्षक|बुजुर्ग|ज्ञानी|ऋषि|तपस्वी|ज्ञान|सलाह)": "[sfx:kulguru_chant]",
    r"(भारी|गंभीर|शक्तिशाली|प्रभावशाली|अधिकार|शक्ति)": "[sfx:heavy_bass_boom]",
    
    # ============ VILLAIN ============
    r"(खलनायक|दुष्ट|बुरा|दुश्मन|शत्रु|वैर|प्रतिद्वंद्वी|अन्य नाग|विरोधी)": "[sfx:dark_drone]",
    r"(फिसफिसाहट|गुप्त|छुपा|काला|अंधेरा|रात|भयानक)": "[sfx:evil_whisper]",
    r"(तूफान|गर्जना|बिजली|वज्र|आसमान|बादल|कहर)": "[sfx:villain_thunder]",
    r"(दिल की धड़कन|दिल|नाड़ी|स्पंद|तेज़)": "[sfx:heartbeat]",
    r"(धातु|लोहा|स्टील|तीव्र|कठोर|भारी)": "[sfx:metallic_hit]",
    
    # ============ GARUDA ============
    r"(गरुड़|उक्कब|चील|शिकारी पक्षी|पक्षी राज|आक्रमण|हमला|युद्ध)": "[sfx:eagle_scream]",
    r"(पंख|पंखों की आवाज़|पंख फड़फड़ाना|उड़ना|हवा|आकाश)": "[sfx:wings_flapping]",
    r"(तेज़ हवा|झोंका|गस्ट|हवा का झोंका|आंधी|तूफान)": "[sfx:wind_gust]",
    r"(गिरना|गिरा|धड़ाम|जोर की आवाज़|प्रभाव|टकराव|टक्कर)": "[sfx:heavy_impact]",
    r"(दहाड़|गर्जना|चीख|तीव्र आवाज़|शक्तिशाली|भयंकर)": "[sfx:roar]",
    
    # ============ ROMANCE (Myra & Veer) ============
    r"(प्यार|मोहब्बत|प्रेम|चाहत|ख्वाहिश|दिल|हृदय|प्रिय|प्रियतम|पति)": "[bgm:love]",
    r"(रोमांटिक|प्रेमपूर्ण|कोमल|नरम|मीठा|सुंदर|मनमोहक)": "[bgm:love]",
    r"(मिरा|वीर|कपल|जोड़ा|दोनों|साथ|संग|एक दूसरे)": "[bgm:myra_love]",
    
    # ============ ACTION ============
    r"(लड़ाई|संघर्ष|झगड़ा|मार|पिटाई|हमला|दंगा|युद्ध|विरोध)": "[sfx:punch]",
    r"(आग|आग लगना|जलना|अग्नि|दहकना|प्रज्वलित)": "[sfx:fire]",
    r"(विस्फोट|बम|फटना|धमाल|विस्फोटक|बिस्फोटन)": "[sfx:explosion]",
    r"(तलवार|तलवार की आवाज़|ख़ंजर|शस्त्र|हथियार|काटना|पार करना)": "[sfx:sword_clash]",
    r"(ऊर्जा किरण|शक्ति का विस्फोट|जादू|ताकत|शक्ति)": "[sfx:energy_blast]",
    r"(धरती|जमीन|दरार|फटना|कंपन|झनझनाहट)": "[sfx:ground_crack]",
    r"(मलबा|टुकड़े|उड़ना|भाग जाना|बिखरना)": "[sfx:flying_debris]",
    
    # ============ NAAGIN TRANSFORMATION (SIGNATURE SOUND) ============
    r"(रूपांतर|बदल|नागिन बन|शक्ति जागृत|परिवर्तन|बदलाव|रूप बदल|मेटामॉर्फोसिस)": "[sfx:naagin_transform]",
    r"(नागिन|नाग रूप|साँप का रूप|सर्प रूप|शक्तिशाली|जागृत)": "[sfx:naagin_transform]",
    
    # ============ NAAGMANI (ALWAYS SAME SOUND) ============
    r"(नागमणि|मणि|रत्न|हीरा|जादुई|शक्तिशाली|अनमोल|अमूल्य)": "[sfx:naagmani_signature]",
    
    # ============ EMOTIONAL/SAD ============
    r"(दुःख|गम|उदास|रुलाई|आँसू|दर्द|पीड़ा|तकलीफ|कष्ट|व्यथा)": "[bgm:emotional_sad]",
    r"(अकेला|अकेली|अकेलेपन|अलग|दूर|विछोह|वियोग|बिछड़ना)": "[bgm:emotional_sad]",
    r"(मृत्यु|मर|मरना|अंत|समाप्त|नष्ट|खत्म|जीवन)": "[bgm:emotional_sad]",
    
    # ============ FOREST ============
    r"(जंगल|वन|वनस्पति|पेड़|पत्तियां|घास|वनचर|कानन|वनस्पति)": "[bgm:forest]",
    r"(पक्षी|चिड़िया|कलरव|गीत|आवाज़|संगीत|चहचहाना)": "[sfx:forest_birds]",
    r"(हवा|हवा का झोंका|बयार|सुगंध|ठंडक)": "[sfx:forest_wind]",
    r"(नदी|जल|जलस्रोत|प्रवाह|गुड़गुड़ाहट)": "[sfx:river]",
    r"(टिड्डी|टिड्डियों की आवाज़|रात|रात की आवाज़|छोटी आवाज़|गिड़गिड़ाहट)": "[sfx:crickets]",
    r"(उल्लू|उल्लू की आवाज़|रात्रि|अंधेरा|शांति)": "[sfx:owl]",
    
    # ============ VILLAGE ============
    r"(गाँव|ग्रामीण|देहाती|कस्बा|गाँव के|घर)": "[bgm:village]",
    r"(गाय|गायों की आवाज़|पशु|गायब|डींक्ष|घंटी|मवेशी)": "[sfx:cow_bells]",
    r"(बच्चे|बच्चों|खेल|खिलवाड़|हँसी|शोर|चहचहाहट)": "[sfx:children_voices]",
    r"(बाज़ार|व्यापार|खरीद|बेच|दुकान|भीड़|लोग)": "[sfx:market_ambience]",
    r"(पैर|चलना|कदम|पदचाप|आना|जाना|चलना फिरना)": "[sfx:footsteps]",
}


def _parse_sound_tag(tag_str: str):
    """'[sfx:laugh]' -> ('sfx', 'laugh')"""
    m = re.match(r"\[(sfx|bgm):(\w+)\]", tag_str)
    return (m.group(1), m.group(2)) if m else (None, None)


def find_sound_cues(text: str):
    """Scan story text for every keyword match and return the cues found,
    sorted by where they occur in the text. Each cue is a dict with the
    matched phrase, its character position, and the sfx/bgm tag it maps to.
    Detection only — does not check whether an audio file exists yet."""
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
    """Look up an actual audio file in sound_library/ for a given cue. Returns
    None if the user hasn't dropped that file in yet (caller should skip it,
    not crash)."""
    folder = SFX_DIR if kind == "sfx" else BGM_DIR
    for ext in SOUND_FILE_EXTENSIONS:
        candidate = os.path.join(folder, name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def all_known_sound_tags():
    """Every (kind, name) pair the keyword map can trigger, in a stable order —
    used to populate the upload section's tag picker."""
    seen = []
    for tag in SOUND_KEYWORD_MAP.values():
        kind, name = _parse_sound_tag(tag)
        if kind and (kind, name) not in seen:
            seen.append((kind, name))
    return seen


def list_sound_library():
    """Every sound file currently saved, as {(kind, name): filepath}."""
    found = {}
    for kind, name in all_known_sound_tags():
        path = sound_file_path(kind, name)
        if path:
            found[(kind, name)] = path
    return found


def save_sound_file(uploaded_file, kind: str, name: str):
    """Save an uploaded SFX/BGM file into sound_library/ under the right tag
    name, replacing any existing file for that tag (including a different
    extension from a previous upload)."""
    folder = SFX_DIR if kind == "sfx" else BGM_DIR
    os.makedirs(folder, exist_ok=True)
    for ext in SOUND_FILE_EXTENSIONS:  # clear any previous version first
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
# Helper Functions
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_available_voices():
    """Fetch the live edge-tts voice catalog and return every Hindi and
    English-India voice as {friendly label: voice_id}. This picks up new
    Hindi voices Microsoft adds over time instead of relying on a hardcoded
    list. Falls back to a small hardcoded set if the catalog can't be
    fetched (e.g. no network)."""
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

    # Hindi-native voices first, then Multilingual (also speak Hindi text,
    # just not hi-IN native), then plain en-IN.
    def _sort_key(v):
        is_hindi_native = v["Locale"] != "hi-IN"
        is_multilingual = "Multilingual" not in v["ShortName"]
        return (is_hindi_native, is_multilingual, v.get("Gender", ""), v["ShortName"])

    filtered.sort(key=_sort_key)

    options = {}
    for v in filtered:
        short_name = v["ShortName"]                      # e.g. hi-IN-MadhurNeural
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
    """Create a fresh temp_assets directory, wiping any previous run."""
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR, exist_ok=True)


VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm", ".avi", ".mkv"}


def is_video_file(path_or_name: str) -> bool:
    return os.path.splitext(path_or_name)[1].lower() in VIDEO_EXTENSIONS


def save_uploaded_media(uploaded_file):
    """Persist a single uploaded image OR video to the temp workspace and
    return its path."""
    file_path = os.path.join(TEMP_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


async def generate_audio_file(text: str, voice: str, output_path: str,
                               rate: str = "+0%", pitch: str = "+0Hz"):
    """Generate a single TTS audio file asynchronously using edge-tts and
    return the word-boundary timing list edge-tts reports along the way
    (used to line up SFX/BGM cues with the moment the trigger word is
    actually spoken, instead of guessing by character position)."""
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, pitch=pitch)
    boundaries = []
    with open(output_path, "wb") as f:
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                f.write(chunk["data"])
            elif chunk["type"] == "WordBoundary":
                boundaries.append({
                    "audio_time": chunk["offset"] / 10_000_000,   # 100ns -> seconds
                    "duration": chunk["duration"] / 10_000_000,
                    "text": chunk["text"],
                })
    return boundaries


def _align_cues_to_audio(text: str, boundaries: list, cues: list):
    """Turn each cue's character position in the story text into an actual
    timestamp in the generated narration audio, using edge-tts's own
    word-boundary events. Falls back to a proportional estimate (character
    position / text length) for any word it can't confidently locate."""
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


def mix_scene_audio(narration_input, cues, output_path, scene_duration):
    """Layer any detected SFX/BGM cues on top of the narration for one scene.
    
    narration_input: either a file path (str) or an AudioFileClip object
    SFX play as a short accent right at the cue's timestamp; BGM plays quietly
    from the cue's timestamp to the end of the scene. Cues whose sound file
    hasn't been added to sound_library/ yet are silently skipped (detection
    still ran — the audio just isn't there yet). Returns the path to use for
    this scene: the mixed file if anything was layered in, otherwise the
    original narration path unchanged."""
    
    # Handle both clip and path inputs
    if isinstance(narration_input, str):
        narration_clip = AudioFileClip(narration_input)
        close_narration = True
    else:
        narration_clip = narration_input
        close_narration = False
    
    layers = [narration_clip]
    extra_clips = []  # sfx/bgm clips, closed separately from narration

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
        else:  # bgm — fill from the cue point to the end of the scene
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
        # No SFX/BGM to mix
        if isinstance(narration_input, str):
            # Input was a file path — just return it (already exists on disk)
            if close_narration:
                narration_clip.close()
            return narration_input
        else:
            # Input was a clip — write it to output_path then close
            narration_clip.write_audiofile(output_path, fps=44100, logger=None)
            narration_clip.close()  # Always close, whether close_narration flag or not
            return output_path

    composite = CompositeAudioClip(layers)
    # Use a small epsilon to avoid floating-point precision issues
    composite = _with_duration(composite, min(scene_duration, narration_clip.duration + 0.1))
    composite.write_audiofile(output_path, fps=44100, logger=None)
    composite.close()
    narration_clip.close()  # Always close — either we opened it from a path, or it's a clip we're done with
    for c in extra_clips:
        try:
            c.close()
        except Exception:
            pass
    return output_path


async def generate_all_audio(story_items, scene_voices, scene_styles, progress_callback=None):
    """
    Generate TTS audio for every scene. Each scene can have multiple dialogue slots
    (speaker 1 voice + text, speaker 2 voice + text, etc.), and they're concatenated
    in sequence. Auto-detect + mix in SFX/BGM across the entire concatenated audio
    for that scene.

    story_items: list of (image_path, [slot1, slot2, ...])
      where each slot is {"text": "...", "voice": "...", "rate": "...", "pitch": "..."}
    scene_voices: NOT USED in this version (each slot has its own voice)
    scene_styles: NOT USED in this version (each slot has its own rate/pitch)

    Returns (audio_paths, scene_cues) — audio_paths aligned with story_items order
    (ready to hand straight to build_video), and scene_cues (the detected cues
    per scene, for showing the user what was auto-added).
    """
    audio_paths = []
    scene_cues = []
    total = len(story_items)

    for index, (_, slots) in enumerate(story_items):
        # Generate a separate TTS file for each dialogue slot
        slot_audios = []
        slot_boundaries = []
        combined_text = ""  # concatenate all slot texts for cue detection

        for slot_index, slot in enumerate(slots):
            text = slot.get("text", "").strip()
            if not text:
                continue

            voice = slot.get("voice", "")
            rate = slot.get("rate", "+0%")
            pitch = slot.get("pitch", "+0Hz")

            raw_audio_path = os.path.join(TEMP_DIR, f"audio_raw_{index}_slot{slot_index}.mp3")
            boundaries = await generate_audio_file(text, voice, raw_audio_path, rate=rate, pitch=pitch)
            slot_audios.append(raw_audio_path)
            slot_boundaries.append((text, boundaries))
            combined_text += text + " "

        # Concatenate all slot audios into one scene audio
        if slot_audios:
            mixed_path = os.path.join(TEMP_DIR, f"audio_{index}.mp3")
            slot_clips = [AudioFileClip(p) for p in slot_audios]
            concatenated = concatenate_audioclips(slot_clips)
            scene_duration = concatenated.duration

            # Detect cues in the COMBINED text and align them to the concatenated audio
            cues = find_sound_cues(combined_text)
            # Simple alignment: assume cues appear proportionally across the combined duration
            if combined_text and cues:
                text_len = len(combined_text)
                for cue in cues:
                    ratio = cue["start"] / text_len
                    cue["audio_time"] = ratio * scene_duration
            scene_cues.append(cues)

            # Mix in SFX/BGM (pass clip directly to avoid temp file precision issues)
            mixed_path = os.path.join(TEMP_DIR, f"audio_{index}.mp3")
            final_path = mix_scene_audio(concatenated, cues, mixed_path, scene_duration)
            # concatenated clip is now closed by mix_scene_audio, so don't close it again
            for clip in slot_clips:
                clip.close()
            audio_paths.append(final_path)
        else:
            # No non-empty slots — create a silent 2-second audio placeholder
            # (build_video still needs something to work with)
            silent_path = os.path.join(TEMP_DIR, f"audio_silent_{index}.mp3")
            try:
                # Try MoviePy 2.x method
                from moviepy.audio.AudioClip import AudioClip
                silent_clip = AudioClip.make_silence(2.0, fps=44100, nchannels=2)
            except (TypeError, AttributeError):
                # Fall back to 1.x method
                from moviepy.audio.AudioClip import AudioClip
                silent_clip = AudioClip(lambda t: np.zeros((2,)), duration=2.0, fps=44100)
            silent_clip.write_audiofile(silent_path, fps=44100, logger=None)
            silent_clip.close()
            audio_paths.append(silent_path)
            scene_cues.append([])

        if progress_callback:
            progress_callback((index + 1) / total, f"Generating audio {index + 1}/{total}...")

    return audio_paths, scene_cues


def build_video(story_items, audio_paths, progress_callback=None, motion_effect="random",
                 motion_prompts=None, fal_key=None, video_audio_volumes=None, video_speeds=None):
    """
    Build the final video by pairing each image with its corresponding audio clip.
    motion_effect:
      "none"      -> static image (original behavior)
      "random"/"zoom_in"/etc. -> Ken Burns pan/zoom (camera motion only)
      "ai_motion" -> full AI-generated motion + lip-sync via fal.ai (LTX-2.3 + LatentSync)
                     Falls back to Ken Burns for any picture where generation fails,
                     so one bad/slow API call doesn't kill the whole video.
    video_audio_volumes: list aligned with story_items — for video-clip scenes,
      how loud the clip's OWN original audio plays under the narration
      (0.0 = mute it, 1.0 = full volume). Narration always stays prioritized.
    video_speeds: list aligned with story_items — for video-clip scenes,
      the playback speed (0.25 = very slow, 1.0 = normal, 2.0 = very fast).
    Returns the path to the exported video file.
    """
    # Initialize empty lists if not provided
    if video_audio_volumes is None:
        video_audio_volumes = []
    if video_speeds is None:
        video_speeds = []
    
    print(f"DEBUG: build_video called with {len(story_items)} scenes")
    print(f"DEBUG: video_speeds = {video_speeds}")
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

            if is_video_file(image_path):
                # User uploaded an actual video clip for this scene.
                raw_clip = VideoFileClip(image_path)
                image_clips_to_close.append(raw_clip)
                
                original_duration = raw_clip.duration
                original_fps = raw_clip.fps
                
                # Get speed from slider
                video_speed = 1.0
                if video_speeds and index < len(video_speeds):
                    video_speed = video_speeds[index]
                
                print(f"DEBUG: Video {index} - Original: {original_duration:.2f}s @ {original_fps}fps, Speed: {video_speed}x")
                
                # Calculate expected duration after speed change
                new_duration_after_speed = original_duration / video_speed
                narration_duration = audio_clip.duration
                
                source_clip = raw_clip
                
                # Apply speed by changing FPS directly (more reliable than speedx)
                if video_speed != 1.0:
                    try:
                        # Method 1: Try speedx() first
                        source_clip = raw_clip.speedx(video_speed)
                        actual_duration = source_clip.duration
                        print(f"✓ speedx({video_speed}x) worked: {original_duration:.2f}s → {actual_duration:.2f}s")
                    except:
                        try:
                            # Method 2: Change FPS directly
                            # Lower FPS = slower playback
                            # If speed is 0.5x, use half the FPS
                            new_fps = original_fps * video_speed
                            
                            # Create a new clip with modified FPS
                            source_clip = raw_clip.set_fps(new_fps)
                            actual_duration = source_clip.duration
                            print(f"✓ FPS method: {original_fps}fps → {new_fps}fps, duration: {original_duration:.2f}s → {actual_duration:.2f}s")
                        except:
                            try:
                                # Method 3: Use speedx with inverted speed
                                source_clip = raw_clip.speedx(1.0 / video_speed)
                                actual_duration = source_clip.duration
                                print(f"✓ Inverted speedx(1/{video_speed}x) worked: {original_duration:.2f}s → {actual_duration:.2f}s")
                            except Exception as e:
                                print(f"⚠ All speed methods failed: {e}")
                                source_clip = raw_clip
                                video_speed = 1.0
                                actual_duration = original_duration
                else:
                    actual_duration = original_duration
                    print(f"ℹ Video {index}: normal speed (1.0x)")

                # Use actual duration from speedx, or calculated if it failed
                video_duration = actual_duration if 'actual_duration' in locals() else new_duration_after_speed
                
                print(f"Final check: Video {video_duration:.2f}s vs Narration {narration_duration:.2f}s")

                if video_duration >= narration_duration:
                    # Video is long enough - NO LOOP
                    duration = video_duration
                    print(f"✓ NO LOOP: {video_duration:.2f}s ≥ {narration_duration:.2f}s")
                else:
                    # Video still short
                    if video_speed == 1.0:
                        # Normal speed - must loop
                        loops_needed = int(narration_duration // original_duration) + 1
                        print(f"ℹ LOOPING: speed=1.0x, loops needed={loops_needed}")
                        looped = concatenate_videoclips([source_clip] * loops_needed, method="compose")
                        image_clips_to_close.append(looped)
                        source_clip = looped
                        duration = narration_duration
                    else:
                        # Speed was applied but still short - don't loop
                        duration = video_duration
                        print(f"✓ NO LOOP (speed applied): {video_duration:.2f}s (will end before narration)")

                # Use source_clip directly - if speedx() was applied, it has the correct duration
                # Don't subclip because it will undo the speedx() effect
                segment_clip = source_clip
                segment_duration = segment_clip.duration  # Get the ACTUAL duration after speed adjustment
                print(f"✓ Segment clip duration: {segment_duration:.2f}s (video_speed={video_speed}x)")

                # Handle audio: narration + video's original audio
                original_audio = segment_clip.audio  # None if the clip is silent
                video_vol = (
                    video_audio_volumes[index]
                    if video_audio_volumes and index < len(video_audio_volumes)
                    else ORIGINAL_VIDEO_AUDIO_DEFAULT_VOLUME
                )

                # Pad or trim narration to match ACTUAL segment duration (after speed adjustment)
                if narration_duration < segment_duration:
                    # Video is longer → pad narration with silence at the end
                    narration_to_mix = _pad_audio_with_silence(audio_clip, segment_duration)
                    print(f"✓ Padded audio from {narration_duration:.2f}s to {segment_duration:.2f}s")
                else:
                    # Narration is same or longer → trim to segment duration
                    narration_to_mix = _subclip(audio_clip, 0, segment_duration)
                    print(f"ℹ Trimmed audio from {narration_duration:.2f}s to {segment_duration:.2f}s")

                if original_audio is not None and video_vol > 0:
                    # Mix video's original audio + narration
                    dynamic_video_audio = _video_audio_with_dynamic_volume(
                        original_audio, narration_duration, segment_duration, video_vol
                    )
                    narration_boosted = _with_volume(narration_to_mix, NARRATION_VOLUME_WHEN_MIXED)
                    combined_audio = CompositeAudioClip([dynamic_video_audio, narration_boosted])
                    combined_audio = _pad_audio_with_silence(combined_audio, segment_duration)
                    segment_clip = _with_audio(segment_clip, combined_audio)
                    audio_clips_to_close.append(combined_audio)
                else:
                    # No video audio, just attach narration (already padded if necessary)
                    segment_clip = _with_audio(segment_clip, narration_to_mix)

                image_clips_to_close.append(segment_clip)
                video_segments.append(segment_clip)
                if progress_callback:
                    progress_callback(
                        (index + 1) / total, f"Assembling video segment {index + 1}/{total}..."
                    )
                continue

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

    # ---------------- Step 1: Upload Images or Video Clips ----------------
    st.subheader("1️⃣ Upload Your Pictures or Video Clips")
    st.caption("Mix and match — each scene can be a still photo OR a short video clip.")
    uploaded_images = st.file_uploader(
        "Upload images/videos (order below = order in the video)",
        type=["jpg", "jpeg", "png", "mp4", "mov", "m4v", "webm"],
        accept_multiple_files=True,
    )

    voice_options = get_available_voices()
    voice_labels = list(voice_options.keys())

    # ---------------- Step 2: Sound Library (SFX / BGM uploads) ----------------
    st.subheader("2️⃣ Sound Library — Upload SFX & Music (optional)")
    st.caption(
        "Upload audio here for the tags the story auto-detects (laugh, thunder, love, "
        "devotional, suspense, etc). Anything you don't upload is simply skipped — "
        "detection still runs, the sound just won't play until you add it."
    )
    with st.expander("🎵 Upload / manage sound files", expanded=False):
        library_now = list_sound_library()
        known_tags = all_known_sound_tags()

        if library_now:
            saved_labels = ", ".join(f"{k}:{n}" for (k, n) in library_now.keys())
            st.success(f"✅ Currently saved: {saved_labels}")
        else:
            st.info("No sound files saved yet — everything below will be auto-detected but silent until you add some.")

        sound_uploads = st.file_uploader(
            "Choose one or more audio files (mp3/wav/ogg/m4a)",
            type=["mp3", "wav", "ogg", "m4a"],
            accept_multiple_files=True,
            key="sound_uploads",
        )

        if sound_uploads:
            st.caption("Match each uploaded file to the sound it represents, then save.")
            tag_display = [f"{'🔊' if k == 'sfx' else '🎵'} {k}:{n}" for k, n in known_tags]
            for u_index, up_file in enumerate(sound_uploads):
                stem = os.path.splitext(up_file.name)[0].lower()
                guessed_index = next(
                    (i for i, (k, n) in enumerate(known_tags) if n == stem), 0
                )
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
                    st.success("Saved: " + ", ".join(saved) + " — reload the page to refresh the list above.")

    st.subheader("3️⃣ Default Voice")
    st.caption("Used to pre-fill every scene below — you can still override the voice per picture.")
    default_voice_label = st.selectbox("Default voice", options=voice_labels, key="default_voice_label")
    selected_voice = voice_options[default_voice_label]
    default_voice_index = voice_labels.index(default_voice_label)

    if uploaded_images:
        st.subheader("4️⃣ Write the Story — Voice & Sounds Auto-Detect Per Scene")
        st.caption(
            "Hindi keywords in your text (हंसना, बिजली, प्यार, शेर, डर...) automatically "
            "trigger a matching sound effect or background music cue — no manual tagging needed. "
            "Drop the matching files into `sound_library/sfx/` or `sound_library/bgm/` on the server "
            "(see the top of app.py for the naming convention); cues detected here with no file yet "
            "are flagged so you know what to add."
        )

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

        VOICE_STYLE_PRESETS = {
            "Natural (no change)": (0, 0),
            "Deep & Slow (serious/dramatic)": (-20, -15),
            "Young & Energetic": (15, 10),
            "Warm & Soft": (-10, 5),
            "Bright & Fast (excited)": (15, 15),
        }

        def _apply_voice_preset(scene_index):
            preset_name = st.session_state.get(f"scene_style_preset_{scene_index}")
            if preset_name in VOICE_STYLE_PRESETS:
                rate, pitch = VOICE_STYLE_PRESETS[preset_name]
                st.session_state[f"scene_rate_{scene_index}"] = rate
                st.session_state[f"scene_pitch_{scene_index}"] = pitch

        def _apply_slot_voice_preset(scene_index, slot_idx):
            preset_name = st.session_state.get(f"slot_style_preset_{scene_index}_{slot_idx}")
            if preset_name in VOICE_STYLE_PRESETS:
                rate, pitch = VOICE_STYLE_PRESETS[preset_name]
                st.session_state[f"slot_rate_{scene_index}_{slot_idx}"] = rate
                st.session_state[f"slot_pitch_{scene_index}_{slot_idx}"] = pitch

        # One scene per row: thumbnail/video + 2-3 dialogue slots
        # Each slot = one speaker with text, voice, optional rate/pitch
        NUM_DIALOGUE_SLOTS = 3
        for index, uploaded_file in enumerate(uploaded_images):
            st.subheader(f"Scene {index + 1}: {uploaded_file.name}")
            col_media, col_dialogue = st.columns([1, 3])

            with col_media:
                if is_video_file(uploaded_file.name):
                    st.video(uploaded_file)
                    st.caption(f"🎬 Video clip")
                else:
                    st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
                if is_video_file(uploaded_file.name):
                    st.slider(
                        "🔊 Original video sound level", 0.0, 1.0,
                        ORIGINAL_VIDEO_AUDIO_DEFAULT_VOLUME, step=0.05,
                        key=f"scene_video_audio_vol_{index}",
                        help="How loud this clip's own audio plays under your narration. "
                             "0 = mute the clip's audio, 1 = full volume. Narration stays prioritized.",
                    )
                    st.slider(
                        "▶️ Video speed", 0.25, 2.0, 1.0, step=0.25,
                        key=f"scene_video_speed_{index}",
                        help="Slow down (0.25–0.75) or speed up (1.25–2.0) this video clip. "
                             "1.0 = normal speed.",
                    )

            with col_dialogue:
                st.caption("Add up to 3 speakers for this scene — each with their own voice and text.")
                dialogue_slots = []
                for slot_idx in range(NUM_DIALOGUE_SLOTS):
                    with st.expander(f"Speaker {slot_idx + 1}", expanded=(slot_idx == 0)):
                        slot_text = st.text_area(
                            "Dialogue text",
                            key=f"slot_text_{index}_{slot_idx}",
                            height=80,
                            placeholder="Leave empty to skip this speaker...",
                        )

                        slot_voice_label = st.selectbox(
                            "Voice",
                            options=voice_labels,
                            index=default_voice_index,
                            key=f"slot_voice_label_{index}_{slot_idx}",
                        )
                        slot_voice = voice_options.get(slot_voice_label, selected_voice)

                        with st.expander("⚙️ Style (optional)", expanded=False):
                            st.selectbox(
                                "Style preset",
                                options=list(VOICE_STYLE_PRESETS.keys()),
                                key=f"slot_style_preset_{index}_{slot_idx}",
                                on_change=_apply_slot_voice_preset,
                                args=(index, slot_idx),
                            )
                            st.slider("Speaking rate", -50, 50, 0, step=5,
                                     key=f"slot_rate_{index}_{slot_idx}",
                                     help="Negative = slower, positive = faster")
                            st.slider("Pitch", -50, 50, 0, step=5,
                                     key=f"slot_pitch_{index}_{slot_idx}",
                                     help="Negative = deeper, positive = higher")

                        if slot_text.strip():  # Only include non-empty slots
                            rate_pct = st.session_state.get(f"slot_rate_{index}_{slot_idx}", 0)
                            pitch_hz = st.session_state.get(f"slot_pitch_{index}_{slot_idx}", 0)
                            dialogue_slots.append({
                                "text": slot_text.strip(),
                                "voice": slot_voice,
                                "rate": f"{rate_pct:+d}%",
                                "pitch": f"{pitch_hz:+d}Hz",
                            })

                        # Live preview of detected sounds for this slot
                        scene_cues_preview = find_sound_cues(slot_text) if slot_text.strip() else []
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
                                bits.append(f"{icon} {cue['name']}" if has_file else f"{icon} {cue['name']} ⚠️")
                            st.caption("Auto-detected sounds: " + " · ".join(bits))

                # Collect dialogue slots for this scene EXPLICITLY
                dialogue_slots = []
                for slot_idx in range(NUM_DIALOGUE_SLOTS):
                    slot_text_key = f"slot_text_{index}_{slot_idx}"
                    slot_text = st.session_state.get(slot_text_key, "").strip()
                    
                    if slot_text:  # Only include non-empty slots
                        slot_voice_key = f"slot_voice_label_{index}_{slot_idx}"
                        slot_voice_label = st.session_state.get(slot_voice_key, default_voice_label)
                        slot_voice = voice_options.get(slot_voice_label, selected_voice)
                        
                        rate_pct = st.session_state.get(f"slot_rate_{index}_{slot_idx}", 0)
                        pitch_hz = st.session_state.get(f"slot_pitch_{index}_{slot_idx}", 0)
                        
                        dialogue_slots.append({
                            "text": slot_text,
                            "voice": slot_voice,
                            "rate": f"{rate_pct:+d}%",
                            "pitch": f"{pitch_hz:+d}Hz",
                        })

                # Store the dialogue slots for this scene
                st.session_state[f"scene_dialogue_{index}"] = dialogue_slots if dialogue_slots else [
                    {"text": "", "voice": selected_voice, "rate": "+0%", "pitch": "+0Hz"}
                ]

            st.divider()

    st.subheader("5️⃣ Motion Effect")
    st.caption(
        "Applies to picture scenes only — video-clip scenes already have their own motion "
        "and are used as-is."
    )
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
                    if is_video_file(uploaded_file.name):
                        continue  # already a video clip — AI motion doesn't apply
                    motion_prompts[index] = st.text_input(
                        f"Motion for picture {index + 1} ({uploaded_file.name})",
                        key=f"motion_prompt_{index}",
                        placeholder="e.g. gentle hand wave, background trees swaying",
                    )

    st.subheader("6️⃣ Generate")
    generate_clicked = st.button("🚀 Generate Video", type="primary", use_container_width=True)

    if generate_clicked:
        # ---------------- Validation ----------------
        if not uploaded_images:
            st.error("❌ Please upload at least one picture before generating the video.")
            return

        # Check if each scene has at least ONE speaker with text (multi-dialogue slots)
        missing_text_indexes = []
        for i in range(len(uploaded_images)):
            scene_dialogue = st.session_state.get(f"scene_dialogue_{i}", [])
            # Filter out empty slots
            has_text = any(slot.get("text", "").strip() for slot in scene_dialogue)
            if not has_text:
                missing_text_indexes.append(i)

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
                video_audio_volumes = []
                video_speeds = []
                for index, uploaded_file in enumerate(uploaded_images):
                    image_path = save_uploaded_media(uploaded_file)
                    # Get the multi-slot dialogue for this scene
                    dialogue_slots = st.session_state.get(f"scene_dialogue_{index}", [])
                    story_items.append((image_path, dialogue_slots))

                    if is_video_file(uploaded_file.name):
                        vol = st.session_state.get(f"scene_video_audio_vol_{index}", ORIGINAL_VIDEO_AUDIO_DEFAULT_VOLUME)
                        video_audio_volumes.append(vol)
                        speed = st.session_state.get(f"scene_video_speed_{index}", 1.0)
                        video_speeds.append(speed)
                    else:
                        video_audio_volumes.append(0)  # not a video, doesn't matter
                        video_speeds.append(1.0)  # no speed change for images

            # Audio generation now handles multiple dialogue slots per scene
            audio_progress = st.progress(0, text="Starting audio generation...")

            def audio_progress_callback(fraction, message):
                audio_progress.progress(fraction, text=message)

            audio_paths, scene_cues = asyncio.run(
                generate_all_audio(story_items, None, None, audio_progress_callback)
            )
            audio_progress.progress(1.0, text="Audio generation complete ✅")

            # Let the user know which auto-detected sounds actually got mixed in
            # vs. which ones are still waiting on a file in sound_library/.
            all_cues = [c for cues in scene_cues for c in cues]
            if all_cues:
                missing = sorted({
                    f"{c['kind']}:{c['name']}" for c in all_cues
                    if sound_file_path(c["kind"], c["name"]) is None
                })
                found = sorted({
                    f"{c['kind']}:{c['name']}" for c in all_cues
                    if sound_file_path(c["kind"], c["name"]) is not None
                })
                if found:
                    st.info("🔊 Sounds mixed in: " + ", ".join(found))
                if missing:
                    st.warning(
                        "⚠️ Detected but not mixed in (no file in sound_library/ yet): "
                        + ", ".join(missing)
                    )

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
                    video_audio_volumes=video_audio_volumes,
                    video_speeds=video_speeds,
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
