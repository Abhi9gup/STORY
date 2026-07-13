# 🎬 Video Scene Duration & Narration Handling

## The Change

When you upload a **video clip** as a scene and write narration for it:

### Before (Old Behavior)
- Narration length determined the scene length
- If video was longer than narration, the narration would **loop/repeat** to fill the video

### Now (New Behavior)
- **Video's original length** determines the scene length
- Narration plays **once only** — never repeats, even if video is longer
- If video is longer than narration: narration plays once, then video continues with silence or its original audio
- If video is shorter than narration: video is looped to match narration length (so you hear all your dialogue)

---

## Examples

### Scenario 1: Short Narration, Long Video
**Video duration:** 10 seconds  
**Narration text:** "नमस्ते! यह एक कहानी है।" (3 seconds)  
**Result:** Narration plays (0–3s), then video continues in silence or with its original audio (3–10s)

### Scenario 2: Long Narration, Short Video
**Video duration:** 3 seconds  
**Narration text:** Whole dialogue scene, 10 seconds of speech  
**Result:** Video loops to match 10-second narration, so you hear all your dialogue

### Scenario 3: Perfect Match
**Video duration:** 5 seconds  
**Narration text:** Dialogue that takes 5 seconds to speak  
**Result:** Everything lines up perfectly, no padding or looping needed

---

## Why This Change?

This mimics real video editing behavior:
- **Short clips** (music videos, b-roll): Loop them to fill your narration
- **Long clips** (pre-recorded dialogue, action scenes): Use their full length, don't force narration to repeat

It gives you more natural video flow, especially when mixing:
- Pre-recorded video with live narration
- Action footage where the narration is a voiceover (doesn't need to loop)
- Audio ambient content from the video itself (dialogue, music)

---

## What About SFX/BGM?

Sound effects and background music still work the same way:
- Detected across all speakers' narration text
- Triggered at their keyword moments
- Play within the narration audio only (not extended to fill the video gap)

---

## Video's Original Audio

If your video clip has its own audio (background music, dialogue, ambient sound):
- It still plays **underneath** your narration
- Volume controlled by the "🔊 Original video sound level" slider per scene
- Plays for the **full video duration** (even if narration is shorter)

---

## No Change for Image Scenes

This only affects video-clip scenes. Still-images continue to work as before:
- Single narration per scene
- Ken Burns motion effect applies
- No looping or padding concerns (video content doesn't apply)

---

## TL;DR

✅ Video clips use their **full original length**  
✅ Narration plays **once, never repeats**  
✅ Long narration? Video loops to fit it  
✅ Short narration? Video continues with its own audio or silence  
✅ More natural, realistic video editing behavior  

Done! 🎬
