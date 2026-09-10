---
name: music-must-sound-live
description: "Feedback: generated music must sound like a live band recording (real samples, groove, post-production), never synth/8-bit tones"
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 0bbbf8df-472c-4eeb-bc39-0b004662565a
  modified: 2026-09-10T16:15:47.365Z
---

When producing music for this user, synthesized oscillator/noise instruments are not acceptable. The first doom cover (numpy synth)
was rejected as "8-bit / Sega Mega Drive"; the parts were fine, the sound was not.

**Why:** the user judges results by how much they feel like "absolute live recordings" with groove and real post-production.
**How to apply:** start from real sampled instruments (multi-velocity drum kits, DI guitar libraries through amp captures + cab IRs),
humanize timing/velocity/tempo, and mix with room mics, bus compression and saturation. See [[zmusic-doom-cover-pipeline]] for the working chain.
