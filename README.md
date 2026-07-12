# Sound Library

Drop your own short audio files here — the app auto-detects Hindi keywords in
each scene's story text and mixes in the matching file. Nothing plays until
you add the file; detection still runs and the app will tell you what's
missing.

## Naming convention

    sound_library/sfx/<name>.mp3   -> one-shot accent, plays right at the keyword moment (trimmed to ~4s)
    sound_library/bgm/<name>.mp3   -> mood music, plays quietly from the keyword moment to the end of that scene

Supported extensions: .mp3 .wav .ogg .m4a

## Files this app currently looks for

Based on the keyword map in app.py (SOUND_KEYWORD_MAP):

sfx/  laugh.mp3  hiss.mp3  thunder.mp3  whoosh.mp3  wind.mp3  ding.mp3
      crying.mp3  fear.mp3  panting.mp3  lion_roar.mp3  dog_bark.mp3
      cat_meow.mp3  wolf_howl.mp3  pain_groan.mp3

bgm/  love.mp3  devotional.mp3  suspense.mp3

Add more keyword -> tag rules in SOUND_KEYWORD_MAP (app.py) any time — just
follow the same "[sfx:name]" / "[bgm:name]" pattern and drop a matching file
here with that name.
