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

# --------------------------------------------------------------------------
# MoviePy 1.x / 2.x Compatibility Helpers
# --------------------------------------------------------------------------
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


def _with_duration(clip, duration):
    if hasattr(clip, "with_duration"):
        return clip.with_duration(duration)
    return clip.set_duration(duration)


def _with_audio(clip, audio_clip):
    if hasattr(clip, "with_audio"):
        return clip.with_audio(audio_clip)
    return clip.set_audio(audio_clip)


def _subclip(clip, start, end):
    if hasattr(clip, "subclipped"):
        return clip.subclipped(start, end)
    return clip.subclip(start, end)


def _multiply_speed(clip, factor):
    """
    Safely adjusts the playback speed of a video clip across MoviePy versions.
    """
    if factor == 1.0:
        return clip
        
    # MoviePy 2.x approach
    if hasattr(clip, "with_effects"):
        from moviepy.video.fx import MultiplySpeed
        return clip.with_effects([MultiplySpeed(factor)])
    
    # MoviePy 1.x approach
    if hasattr(clip, "fx"):
        import moviepy.video.fx.all as vfx
        return clip.fx(vfx.speedx, factor)
        
    # Fallback raw length recalculation if fx packages mismatch
    new_duration = clip.duration / factor
    return _with_duration(clip, new_duration)


def _pad_audio_with_silence(audio_clip, target_duration):
    if audio_clip.duration >= target_duration:
        return _subclip(audio_clip, 0, target_duration)
    
    silence_duration = target_duration - audio_clip.duration
    
    try:
        from moviepy.audio.AudioClip import AudioClip
        silence = AudioClip.make_silence(silence_duration, fps=44100, nchannels=2)
    except (TypeError, AttributeError):
        from moviepy.audio.AudioClip import AudioClip
        silence = AudioClip(
            lambda t: np.zeros((2,)),
            duration=silence_duration,
            fps=44100
        )
    
    padded = concatenate_audioclips([audio_clip, silence])
    return padded


def _video_audio_with_dynamic_volume(video_audio, narration_duration, total_duration, ducked_volume):
    if narration_duration >= total_duration:
        return _with_volume(video_audio, ducked_volume)
    
    video_trimmed = _subclip(video_audio, 0, total_duration)
    
    narration_part = _subclip(video_trimmed, 0, narration_duration)
    remaining_part = _subclip(video_trimmed, narration_duration, total_duration)
    
    narration_part_ducked = _with_volume(narration_part, ducked_volume)
    narration_part_ducked = _with_start(narration_part_ducked, 0)
    
    remaining_part_full = _with_volume(remaining_part, 1.0)
    remaining_part_full = _with_start(remaining_part_full, narration_duration)
    
    result = CompositeAudioClip([narration_part_ducked, remaining_part_full])
    result = _with_duration(result, total_duration)
    
    return result


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
        raise RuntimeError("No FAL_KEY set.")

    def report(msg):
        if status_callback:
            status_callback(msg)

    with AudioFileClip(audio_path) as probe:
        target_seconds = probe.duration
    duration_bucket = _pick_duration_bucket(target_seconds)

    report("Uploading photo...")
    image_url = fal_client.upload_file(image_path)

    report("Generating motion...")
    motion_result = fal_client.subscribe(
        "fal-ai/ltx-2.3/image-to-video",
        arguments={
            "image_url": image_url,
            "prompt": motion_prompt or "subtle natural motion",
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
# Audio Logic & Config Constants
# --------------------------------------------------------------------------
TEMP_DIR = "temp_assets"
OUTPUT_FILENAME = "final_output.mp4"

FALLBACK_VOICE_OPTIONS = {
    "Male (English-India)": "en-IN-PrabhatNeural",
    "Female (English-India)": "en-IN-NeerjaNeural",
    "Male (Hindi)": "hi-IN-MadhurNeural",
    "Female (Hindi)": "hi-IN-SwaraNeural",
}

SOUND_LIBRARY_DIR = "sound_library"
SFX_DIR = os.path.join(SOUND_LIBRARY_DIR, "sfx")
BGM_DIR = os.path.join(SOUND_LIBRARY_DIR, "bgm")
SOUND_FILE_EXTENSIONS = (".mp3", ".wav", ".ogg", ".m4a")

SFX_MAX_SECONDS = 4
BGM_VOLUME = 0.22
SFX_VOLUME = 0.9

ORIGINAL_VIDEO_AUDIO_DEFAULT_VOLUME = 0.3
NARRATION_VOLUME_WHEN_MIXED = 1.15

SOUND_KEYWORD_MAP = {
    r"(मंदिर|पूजा|प्रार्थना|भगवान|देवता|महादेव|शिव|आशीर्वाद|पवित्र|पूजनीय|भक्ति)": "[sfx:temple_bells]",
    r"(घंटी|ध्वनि|शंख|शंखनाद|ॐ|ओम)": "[sfx:shankh_conch]",
    r"(आत्मा|आध्यात्मिक|दिव्य|पवित्रता|देवी)": "[sfx:sacred_humming]",
    r"(तांपूरा|ध्रुव्य|सुर|संगीत)": "[sfx:tanpura_drone]",
    r"(गुप्त|रहस्य|छुपा|राज|गूढ़|अज्ञात|भेद|छिपा)": "[sfx:mystery_whoosh]",
    r"(गुफा|गुप्त द्वार|दरवाज़ा|रहस्य द्वार|पुरातन)": "[sfx:stone_grinding]",
    r"(भविष्यवाणी|भविष्य|भविष्यद्वाणी|पूर्वाभास|इशारा)": "[sfx:magical_echo]",
    r"(प्राचीन|अति प्राचीन|पुरातत्व|ऐतिहासिक|पुरानी)": "[sfx:ancient_hum]",
    r"(नाग|नागिन|साँप|सर्प|फुंकार|डसना|जहर|विष|काटना)": "[sfx:snake_hiss]",
    r"(शक्ति|नाग शक्ति|नागिन शक्ति|जागृत|जाग गई|शक्तिशाली|शक्तिमान)": "[sfx:energy_pulse]",
    r"(नागमणि|हीरा|रत्न|चमकना|दीप्ति|प्रकाश|गहरा रंग)": "[sfx:emerald_glow]",
    r"(ऊर्जा|चेतना|जीवन|प्रवाह|लहर|तरंग)": "[sfx:magical_energy]",
    r"(नागलोक|नाग दुनिया|नाग राज्य|स्वर्ग|दिव्य लोक|दूसरी दुनिया|अन्य लोक|परलोक)": "[sfx:mystical_choir]",
    r"(क्रिस्टल|क्रिस्टल्स|पारदर्शी|चमकदार|दीप्तिमान|खनिज|रत्न)": "[sfx:crystal_ambience]",
    r"(जलप्रपात|झरना|पानी|जल|बहना|प्रवाहित|बहती)": "[sfx:waterfall]",
    r"(ईश्वरीय|देव|परमात्मा)": "[sfx:divine_ambience]",
    r"(कुलगुरु|गुरु|शिक्षक|बुजुर्ग|ज्ञानी|ऋषि|तपस्वी|ज्ञान|सलाह)": "[sfx:kulguru_chant]",
    r"(भारी|गंभीर|प्रभावशाली|अधिकार)": "[sfx:heavy_bass_boom]",
    r"(खलनायक|दुष्ट|बुरा|दुश्मन|शत्रु|वैर|प्रतिद्वंद्वी|अन्य नाग|विरोधी)": "[sfx:dark_drone]",
    r"(फिसफिसाहट|काला|अंधेरा|रात|भयानक)": "[sfx:evil_whisper]",
    r"(तूफान|गर्जना|बिजली|वज्र|आसमान|बादल|कहर)": "[sfx:villain_thunder]",
    r"(दिल की धड़कन|दिल|नाड़ी|स्पंद|तेज़)": "[sfx:heartbeat]",
    r"(धातु|लोहा|स्टील|तीव्र|कठोर)": "[sfx:metallic_hit]",
    r"(गरुड़|उक्कब|चील|शिकारी पक्षी|पक्षी राज|आक्रमण|हमला|युद्ध)": "[sfx:eagle_scream]",
    r"(पंख|पंखों की आवाज़|पंख फड़फड़ाना|उड़ना|हवा|आकाश)": "[sfx:wings_flapping]",
    r"(तेज़ हवा|झोंका|गस्ट|हवा का झोंका|आंधी)": "[sfx:wind_gust]",
    r"(गिरना|गिरा|धड़ाम|जोर की आवाज़|प्रभाव|टकराव|टक्कर)": "[sfx:heavy_impact]",
    r"(दहाड़|चीख|तीव्र आवाज़|भयंकर)": "[sfx:roar]",
    r"(प्यार|मोहब्बत|प्रेम|चाहत|ख्वाहिश|हृदय|प्रिय|प्रियतम|पति)": "[bgm:love]",
    r"(रोमांटिक|प्रेमपूर्ण|कोमल|नरम|मीठा|सुंदर|मनमोहक)": "[bgm:love]",
    r"(मिरा|वीर|कपल|जोड़ा|दोनों|साथ|संग|एक दूसरे)": "[bgm:myra_love]",
    r"(लड़ाई|संघर्ष|झगड़ा|मार|पिटाई|दंगा|विरोधी)": "[sfx:punch]",
    r"(आग|आग लगना|जलना|अग्नि|दहकना|प्रज्वलित)": "[sfx:fire]",
    r"(विस्फोट|बम|फटना|धमाल|विस्फोटक|बिस्फोटन)": "[sfx:explosion]",
    r"(तलवार|तलवार की आवाज़|ख़ंजर|शस्त्र|हथियार|काटना|पार करना)": "[sfx:sword_clash]",
    r"(ऊर्जा किरण|शक्ति का विस्फोट|जादू|ताकत)": "[sfx:energy_blast]",
    r"(धरती|जमीन|दरार|कंपन|झनझनाहट)": "[sfx:ground_crack]",
    r"(मलबा|टुकड़े|उड़ना|भाग जाना|बिखरना)": "[sfx:flying_debris]",
    r"(रूपांतर|बदल|नागिन बन|शक्ति जागृत|परिवर्तन|बदलाव|रूप बदल|मेटामॉर्फोसिस)": "[sfx:naagin_transform]",
    r"(नागमणि|मणि|जादुई|अमूल्य)": "[sfx:naagmani_signature]",
    r"(दुःख|गम|उदास|रुलाई|आँसू|दर्द|पीड़ा|तकलीफ|कष्ट|व्यथा)": "[bgm:emotional_sad]",
    r"(अकेला|अकेली|अकेलेपन|अलग|दूर|विछोह|वियोग|बिछड़ना)": "[bgm:emotional_sad]",
    r"(मृत्यु|मर|मरना|अंत|समाप्त|नष्ट|खत्म)": "[bgm:emotional_sad]",
    r"(जंगल|वन|वनस्पति|पेड़|पत्तियां|घास|वनचर|कानन)": "[bgm:forest]",
    r"(पक्षी|चिड़िया|कलरव|गीत|चहचहाना)": "[sfx:forest_birds]",
    r"(बयार|सुगंध|ठंडक)": "[sfx:forest_wind]",
    r"(नदी|जलस्रोत|गुड़गुड़ाहट)": "[sfx:river]",
    r"(टिड्डी|टिड्डियों की आवाज़|रात की आवाज़|गिड़गिड़ाहट)": "[sfx:crickets]",
    r"(उल्लू|उल्लू की आवाज़|रात्रि|शांति)": "[sfx:owl]",
    r"(गाँव|ग्रामीण|देहाती|कस्बा|घर)": "[bgm:village]",
    r"(गाय|गायों की आवाज़|पशु|गायब|घंटी|मवेशी)": "[sfx:cow_bells]",
    r"(बच्चे|बच्चों|खेल|खिलवाड़|हँसी|शोर|चहचहाहट)": "[sfx:children_voices]",
    r"(बाज़ार|व्यापार|खरीद|बेच|दुकान|भीड़|लोग)": "[sfx:market_ambience]",
    r"(पैर|चलना|कदम|पदचाप|आना|जाना)": "[sfx:footsteps]",
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
    dest_path = os.path.join(folder, name + ext)
    with open(dest_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return dest_path


@st.cache_data(show_spinner=False)
def get_available_voices():
    try:
        all_voices = asyncio.run(edge_tts.list_voices())
    except Exception:
        return dict(FALLBACK_VOICE_OPTIONS)

    wanted_locales = {"hi-IN", "en-IN"}
    filtered = [v for v in all_voices if v.get("Locale") in wanted_locales]
    if not filtered:
        return dict(FALLBACK_VOICE_OPTIONS)

    options = {}
    for v in filtered:
        short_name = v["ShortName"]
        persona = short_name.split("-")[-1].replace("Neural", "")
        lang = "Hindi" if v["Locale"] == "hi-IN" else "English-IN"
        label = f"{v.get('Gender', '')} ({lang}) — {persona}"
        options[label] = short_name
    return options


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


async def generate_audio_file(text: str, voice: str, output_path: str):
    # Lock generation speed strictly at baseline 1X
    communicate = edge_tts.Communicate(text=text, voice=voice, rate="+0%", pitch="+0Hz")
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


def mix_scene_audio(narration_input, cues, output_path, scene_duration):
    if isinstance(narration_input, str):
        narration_clip = AudioFileClip(narration_input)
        close_narration = True
    else:
        narration_clip = narration_input
        close_narration = False
        
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
        if isinstance(narration_input, str):
            if close_narration:
                narration_clip.close()
            return narration_input
        else:
            narration_clip.write_audiofile(output_path, fps=44100, logger=None)
            narration_clip.close()
            return output_path

    composite = CompositeAudioClip(layers)
    composite = _with_duration(composite, min(scene_duration, narration_clip.duration + 0.1))
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

    for index, (_, slots, *_) in enumerate(story_items):
        slot_audios = []
        combined_text = ""

        for slot_index, slot in enumerate(slots):
            text = slot.get("text", "").strip()
            if not text:
                continue

            voice = slot.get("voice", "")
            raw_audio_path = os.path.join(TEMP_DIR, f"audio_raw_{index}_slot{slot_index}.mp3")
            await generate_audio_file(text, voice, raw_audio_path)
            slot_audios.append(raw_audio_path)
            combined_text += text + " "

        if slot_audios:
            mixed_path = os.path.join(TEMP_DIR, f"audio_{index}.mp3")
            slot_clips = [AudioFileClip(p) for p in slot_audios]
            concatenated = concatenate_audioclips(slot_clips)
            scene_duration = concatenated.duration

            cues = find_sound_cues(combined_text)
            if combined_text and cues:
                text_len = len(combined_text)
                for cue in cues:
                    ratio = cue["start"] / text_len
                    cue["audio_time"] = ratio * scene_duration
            scene_cues.append(cues)

            final_path = mix_scene_audio(concatenated, cues, mixed_path, scene_duration)
            audio_paths.append(final_path)
            
            for p in slot_audios:
                try:
                    os.remove(p)
                except Exception:
                    pass
        else:
            audio_paths.append(None)
            scene_cues.append([])

        if progress_callback:
            progress_callback(index + 1, total)

    return audio_paths, scene_cues


def build_video(story_items, audio_paths, motion_prompts, video_speeds, use_ai_motion=False, fal_key=None, status_container=None):
    clips = []
    
    for index, (media_path, slots, _) in enumerate(story_items):
        audio_path = audio_paths[index]
        motion_prompt = motion_prompts[index] if index < len(motion_prompts) else ""
        video_speed = video_speeds[index] if index < len(video_speeds) else 1.0
        
        if audio_path and os.path.exists(audio_path):
            with AudioFileClip(audio_path) as probe:
                narration_duration = probe.duration
        else:
            narration_duration = 5.0

        # Case A: Input media is a VIDEO clip
        if is_video_file(media_path):
            if status_container:
                status_container.write(f"🎬 Processing Video Scene {index + 1} (Speed: {video_speed}x)...")
            
            raw_video = VideoFileClip(media_path)
            video_clip = _multiply_speed(raw_video, video_speed)
            
            total_duration = max(narration_duration, video_clip.duration)
            
            if video_clip.duration < total_duration:
                loops = int(total_duration // video_clip.duration) + 1
                video_clip = concatenate_videoclips([video_clip] * loops)
                
            video_clip = _subclip(video_clip, 0, total_duration)
            
            if video_clip.audio:
                ducked_bg_audio = _video_audio_with_dynamic_volume(
                    video_clip.audio, narration_duration, total_duration, ORIGINAL_VIDEO_AUDIO_DEFAULT_VOLUME
                )
                if audio_path and os.path.exists(audio_path):
                    narration_clip = AudioFileClip(audio_path)
                    narration_clip = _with_volume(narration_clip, NARRATION_VOLUME_WHEN_MIXED)
                    combined_audio = CompositeAudioClip([ducked_bg_audio, narration_clip])
                    video_clip = _with_audio(video_clip, combined_audio)
            else:
                if audio_path and os.path.exists(audio_path):
                    video_clip = _with_audio(video_clip, AudioFileClip(audio_path))
            clips.append(video_clip)

        # Case B: Input media is an IMAGE (AI Full Motion Track)
        elif use_ai_motion and audio_path and os.path.exists(audio_path):
            if status_container:
                status_container.write(f"🤖 Running AI Full Motion Pipeline for Scene {index + 1}...")
            ai_output_path = os.path.join(TEMP_DIR, f"ai_motion_{index}.mp4")
            try:
                generate_ai_motion_clip(
                    image_path=media_path, audio_path=audio_path, motion_prompt=motion_prompt,
                    output_path=ai_output_path, fal_key=fal_key,
                    status_callback=lambda msg: status_container.write(f"  ↳ Scene {index + 1}: {msg}") if status_container else None
                )
                raw_ai = VideoFileClip(ai_output_path)
                video_clip = _multiply_speed(raw_ai, video_speed)
                video_clip = _with_duration(video_clip, narration_duration)
                clips.append(video_clip)
            except Exception as e:
                if status_container:
                    status_container.warning(f"⚠️ AI Motion failed. Falling back to Ken Burns.")
                img_clip = ImageClip(media_path)
                base_duration = narration_duration / video_speed
                img_clip = _with_duration(img_clip, base_duration)
                kb_clip = apply_ken_burns(img_clip, base_duration)
                kb_clip = _multiply_speed(kb_clip, video_speed)
                kb_clip = _with_duration(kb_clip, narration_duration)
                if audio_path and os.path.exists(audio_path):
                    kb_clip = _with_audio(kb_clip, AudioFileClip(audio_path))
                clips.append(kb_clip)

        # Case C: Input media is an IMAGE (Standard Ken Burns Mode)
        else:
            if status_container:
                status_container.write(f"🖼️ Framing Image Scene {index + 1} (Ken Burns Mode)...")
            img_clip = ImageClip(media_path)
            base_duration = narration_duration / video_speed
            img_clip = _with_duration(img_clip, base_duration)
            kb_clip = apply_ken_burns(img_clip, base_duration)
            
            kb_clip = _multiply_speed(kb_clip, video_speed)
            kb_clip = _with_duration(kb_clip, narration_duration)
            if audio_path and os.path.exists(audio_path):
                kb_clip = _with_audio(kb_clip, AudioFileClip(audio_path))
            clips.append(kb_clip)

    if not clips:
        raise RuntimeError("No media elements generated.")
        
    final_video = concatenate_videoclips(clips, method="compose")
    final_video.write_videofile(OUTPUT_FILENAME, fps=24, codec="libx264", audio_codec="aac", remove_temp=True, logger=None)
    final_video.close()
    for c in clips:
        c.close()
    return OUTPUT_FILENAME


# --------------------------------------------------------------------------
# Streamlit Interface
# --------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Automated Story-to-Video Generator", layout="wide")
    
    # === PASSWORD GATEWAY LAYER ===
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.title("🔒 Access Restricted")
        password_input = st.text_input("Enter Password to Unlock App", type="password")
        if st.button("Unlock Interface"):
            if password_input == "your_password_here":   # Change your application password target here
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Incorrect Password. Access Denied.")
        return
    # ===============================

    st.title("🎬 Automated Story-to-Video Generator")
    st.caption("Construct immersive, audio-synced cinematic clips completely out of single-page structural inputs.")

    os.makedirs(SOUND_LIBRARY_DIR, exist_ok=True)
    os.makedirs(SFX_DIR, exist_ok=True)
    os.makedirs(BGM_DIR, exist_ok=True)

    voice_options = get_available_voices()

    with st.sidebar:
        st.header("⚙️ Pipeline Configuration")
        use_ai_motion = st.checkbox("Enable AI Full Motion (Chained LTX + LatentSync)", value=False)
        fal_key_input = st.text_input("FAL_KEY API Token", type="password")
        
        st.markdown("---")
        st.subheader("🎵 Sound Library Manager")
        all_tags = all_known_sound_tags()
        tag_labels = [f"[{kind}:{name}]" for kind, name in all_tags]
        selected_tag_str = st.selectbox("Assign Sound Node Tag", options=tag_labels)
        uploaded_sound = st.file_uploader("Upload Audio Sample", type=["mp3", "wav", "m4a", "ogg"])
        
        if uploaded_sound and selected_tag_str:
            kind, name = _parse_sound_tag(selected_tag_str)
            save_sound_file(uploaded_sound, kind, name)
            st.success("Registered audio profile successfully!")

    st.subheader("🎞️ Storyboard Canvas")
    num_scenes = st.number_input("Total Timeline Scenes", min_value=1, max_value=25, value=2)
    story_inputs = []
    
    for idx in range(num_scenes):
        st.markdown(f"### 🎬 Scene Setup {idx + 1}")
        col1, col2 = st.columns([1, 2])
        
        with col1:
            uploaded_media = st.file_uploader(f"Upload media for Scene {idx + 1}", type=["png", "jpg", "jpeg", "mp4", "mov", "avi"], key=f"media_{idx}")
            motion_prompt = ""
            if use_ai_motion:
                motion_prompt = st.text_input(f"AI Prompt modifiers (Scene {idx + 1})", value="subtle movements", key=f"prompt_{idx}")
            
            # Speed configuration slider
            v_speed = st.slider(f"🏃 Video Playback Speed (Scene {idx+1})", min_value=0.25, max_value=4.0, value=1.0, step=0.25, key=f"vspeed_{idx}")
        
        with col2:
            st.markdown("**Dialogue & Narration Tracks**")
            num_slots = st.number_input("Dialogue instances", min_value=1, max_value=5, value=1, key=f"slots_count_{idx}")
            slots = []
            
            for s_idx in range(int(num_slots)):
                s_col1, s_col2 = st.columns([3, 1])
                with s_col1:
                    txt = st.text_area(f"Spoken text line {s_idx + 1}", height=68, key=f"txt_{idx}_{s_idx}")
                with s_col2:
                    vc = st.selectbox("Voice", options=list(voice_options.keys()), index=0, key=f"vc_{idx}_{s_idx}")
                    
                if txt.strip():
                    slots.append({"text": txt, "voice": voice_options[vc]})
            
            if uploaded_media and slots:
                setup_workspace()
                local_media_path = save_uploaded_media(uploaded_media)
                story_inputs.append((local_media_path, slots, {"motion_prompt": motion_prompt, "video_speed": v_speed}))

    st.markdown("---")
    
    if st.button("🚀 Render Master Video Composition", use_container_width=True):
        if not story_inputs:
            st.error("Please add uploaded media assets and narrative text scripts first.")
            return
            
        status_box = st.container()
        try:
            status_box.info("🎙️ Synthesizing Voice Narration tracks (Locked at steady 1x speed)...")
            audio_paths, _ = asyncio.run(generate_all_audio(
                story_inputs, progress_callback=lambda c, t: status_box.write(f"  ↳ Narrative Generation: Scene {c}/{t} processed.")
            ))
            
            status_box.info("🎬 Rendering timeline layers applying target visual speeds...")
            motion_prompts = [item[2]["motion_prompt"] for item in story_inputs]
            video_speeds = [item[2]["video_speed"] for item in story_inputs]
            
            output_mp4 = build_video(
                story_items=story_inputs, audio_paths=audio_paths,
                motion_prompts=motion_prompts, video_speeds=video_speeds,
                use_ai_motion=use_ai_motion, fal_key=fal_key_input, status_container=status_box
            )
            
            status_box.success("🎉 Video Composition completed successfully!")
            with open(output_mp4, "rb") as video_file:
                st.video(video_file.read())
                
        except Exception:
            st.error("🚨 Compilation failed during background generation workflow.")
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
