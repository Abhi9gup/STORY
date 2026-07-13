# 🎭 Multi-Voice Dialogue per Scene — Usage Guide

Your app now supports **2-3 different speakers per scene**. Each speaker has their own voice, text, and optional rate/pitch styling.

---

## How It Works

### UI Structure
For each scene, you'll see **3 expandable speaker slots**:

```
Scene 1: photo.jpg
├─ Speaker 1 [expanded by default]
│  ├─ Dialogue text
│  ├─ Voice selector
│  └─ Style (rate/pitch/preset)
├─ Speaker 2 [collapsed]
│  ├─ Dialogue text
│  ├─ Voice selector
│  └─ Style (rate/pitch/preset)
└─ Speaker 3 [collapsed]
   ├─ Dialogue text
   ├─ Voice selector
   └─ Style (rate/pitch/preset)
```

### Playback Order
When the video plays:
1. **Speaker 1 text** plays from the start (voice A)
2. **Speaker 2 text** plays right after (voice B)
3. **Speaker 3 text** plays right after (voice C)
4. All audio plays **under the same video/image** for that scene
5. **SFX/BGM** are detected across all speakers' combined text and triggered at matching keywords

---

## Example: Dialogue Scene

**Setup:**
- Scene 1: Image of two people talking

**Speakers:**
- **Speaker 1** (Madhur — Male Hindi):  
  *Dialogue:* "नमस्ते! मेरा नाम राज है।"  
  *Style:* Natural

- **Speaker 2** (Swara — Female Hindi):  
  *Dialogue:* "मुझे खुशी है तुमसे मिलकर।"  
  *Style:* Warm & Soft

- **Speaker 3** (Leave empty):  
  *(skipped)*

**Playback:**
1. Madhur's voice: "नमस्ते!..." (2-3 sec)
2. Swara's warm voice: "मुझे खुशी है..." (2-3 sec)
3. Video image plays for total ~5-6 seconds

**Auto-detected sounds:**
- "खुशी" (happy) might trigger "love" BGM if you have that keyword mapped

---

## Tips

### Empty Slots Are OK
Leave Speaker 2 and Speaker 3 empty if you only need one narrator for a scene. The app skips empty slots automatically.

### Voice Variety
Use different voices to show **dialogue between characters**:
- Speaker 1: Madhur (deep male for "hero")
- Speaker 2: Swara (warm female for "heroine")
- Speaker 3: Ava or Andrew (multilingual for narrator or other character)

### Mixing Different Languages
Each speaker can speak Hindi, English, or multilingual text:
- Speaker 1: "यह एक कहानी है" (Hindi)
- Speaker 2: "This is a story" (English)
- Speaker 3: "C'est une histoire" (French, via multilingual voice)

### Pauses Between Speakers
Each speaker's audio plays immediately after the previous one — no gaps. If you want a pause, add silence to the end of one speaker's text (just type spaces or very short words).

### Sound Cue Detection
SFX/BGM keywords are detected **across all speakers' text combined** for that scene:
- If Speaker 1 says "... हंसने लगा" (laugh) → laugh.mp3 triggers
- If Speaker 2 says "... बिजली गिरी" (thunder) → thunder.mp3 triggers
- Timing is proportional — earlier speakers' keywords trigger earlier in the concatenated audio

---

## Common Scenarios

### Narrated Story (1 speaker)
- **Speaker 1**: Full narration with all text
- **Speaker 2 & 3**: Empty
- **Result**: Single voice narrating the scene

### Dialogue Scene (2 speakers)
- **Speaker 1**: Character A's dialogue
- **Speaker 2**: Character B's dialogue
- **Speaker 3**: Empty (or use for a third character)
- **Result**: Conversation under one image

### Narrator + Character Dialogue (3 speakers)
- **Speaker 1**: Narrator/background voice (deeper, slower)
- **Speaker 2**: Character A (energetic, higher)
- **Speaker 3**: Character B (warm, gentle)
- **Result**: Rich layered scene with multiple voices

### Multilingual Combo
- **Speaker 1**: Hindi narration (Madhur)
- **Speaker 2**: English commentary (Ava)
- **Speaker 3**: Local dialect (Andrew)

---

## Styling Each Speaker Independently

Each speaker's "⚙️ Style (optional)" section lets you set:
- **Speaking rate**: Slow & dramatic or fast & excited
- **Pitch**: Deep or high
- **Style preset**: Quick shortcuts (Deep & Slow, Young & Energetic, etc.)

**Example:**
- Speaker 1 (hero): Deep & Slow (serious)
- Speaker 2 (heroine): Bright & Fast (excited)
- Speaker 3 (narrator): Natural (neutral)

---

## Behind the Scenes

**What the app does:**
1. Generates TTS audio for each non-empty speaker **separately** (preserves voice distinction)
2. **Concatenates** them in order (Speaker 1 → Speaker 2 → Speaker 3)
3. Detects **all SFX/BGM keywords** in the combined text
4. **Aligns keyword timing** proportionally across the concatenated audio
5. **Mixes in SFX/BGM** at the right moments
6. Attaches the final audio to the scene's image/video

---

## Troubleshooting

**Q: Why don't my speakers sound like they're interrupting each other?**  
A: They play sequentially, not overlapping. If you want overlapping dialogue, that requires a different approach (multiple audio tracks, mixing them with start offsets) — for now, speakers take turns.

**Q: Can I reorder speakers?**  
A: Not in the UI. The order is always Speaker 1 → 2 → 3. If you want a different order, put the text in the slots in your desired order.

**Q: One speaker's voice is too quiet/loud compared to others.**  
A: Edge-tts TTS volumes can vary. You can manually adjust after generation, or try different rate/pitch values (sometimes a slower rate sounds louder).

**Q: How long can each speaker's text be?**  
A: No hard limit, but TTS audio quality can degrade on very long blocks (100+ words). Keep it natural-sounding — break long monologues into 2-3 slots or use different speakers.

---

Enjoy creating multi-voice stories! 🎬
