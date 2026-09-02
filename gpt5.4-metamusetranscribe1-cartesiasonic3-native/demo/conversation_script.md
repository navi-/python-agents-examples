# Babel demo call: one phone, three people, three languages

Rendered conversation: `demo_conversation.mp3` (88s). Voices: Babel = Kokoro `af_heart` (one voice, three languages, heard through the phone band-pass), Sam = `am_michael`, Carlos = `em_alex`, Priya = `hf_alpha`.

Scenario: three friends call Bistro Mundo, put the phone in the middle of the table and order in English, Spanish and Hindi. Meta Muse Voice Transcribe tags every utterance with a stable speaker label and endpoints each turn; GPT-5.4 mini answers each person in their own language; Cartesia Sonic-3 speaks it with one voice.

| Time | Speaker | Line |
|---|---|---|
| 00:00.45 | **Babel** | `<en>` Hi, welcome to Bistro Mundo, I'm Babel. Put me on speaker. Everyone can order in any language you like, and I'll remember who said what. |
| 00:09.34 | Sam | Hey Babel, it's Sam. There are three of us. I'll start with two tacos al pastor and a sparkling water. |
| 00:16.62 | **Babel** | `<en>` Got it, Sam. Two tacos al pastor and a sparkling water for you. Who's next? |
| 00:21.88 | Carlos | Hola Babel, soy Carlos. Para mí la paella de mariscos, por favor. ¿Es suficiente para dos personas? |
| 00:27.92 | **Babel** | `<es>` Claro, Carlos. La paella de mariscos rinde para dos personas, y ya está anotada a tu nombre. |
| 00:34.62 | Priya | Hi, मैं Priya हूँ। मेरे लिए एक paneer tikka, extra spicy, और एक mango lassi please. |
| 00:43.66 | **Babel** | `<hi>` ज़रूर Priya। एक पनीर टिक्का, एक्स्ट्रा स्पाइसी, और एक मैंगो लस्सी आपके नाम पर लिख दिया। |
| 00:51.45 | **Babel** | `<en>` By the way, tonight we also have a chef's special, a saffron seafood stew with grilled bread and a little bit of *(interrupted)* |
| 00:54.05 | Sam | Sorry Babel, quick change. Make mine three tacos, not two. |
| 00:58.79 | **Babel** | `<en>` No problem, Sam. Three tacos al pastor for you now. |
| 01:02.75 | Carlos | Babel, ¿me puedes repetir quién pidió qué? |
| 01:05.61 | **Babel** | `<es>` Por supuesto. Sam: tres tacos al pastor y un agua con gas. Carlos: la paella de mariscos. Priya: un paneer tikka extra picante y un mango lassi. |
| 01:17.17 | Priya | Perfect. Babel, ticket text कर दो, and we're done. |
| 01:21.83 | **Babel** | `<hi>` हो गया Priya, टिकट आपके फ़ोन पर भेज दिया। Sam, Carlos, Priya, thank you, enjoy your dinner! |

## What the pipeline does, line by line

- **00:00.45 Babel**: Turn 1 greeting. Plivo `start` -> Muse session opened (g711_ulaw, diarization on, keyword biasing: menu names) -> GPT-5.4 mini streams `<en> ...` -> Cartesia Sonic-3 context, sentence by sentence -> Plivo playAudio -> checkpoint -> playedStream.
- **00:09.34 Sam**: Muse final transcript tagged `[Speaker 1]`, endpointed 0.4s after Sam stops. Tools: identify_speaker(1, Sam) -> Muse keyword biasing updated with 'Sam'; add_order_item(1, tacos al pastor, 2); add_order_item(1, sparkling water, 1).
- **00:16.62 Babel**: Reply tagged `<en>`; Cartesia language = en.
- **00:21.88 Carlos**: New voice: Muse diarization assigns `[Speaker 2]` (event speaker_detected). Spanish, no language hint needed. Tools: identify_speaker(2, Carlos, es); add_order_item(2, seafood paella, 1).
- **00:27.92 Babel**: Reply tagged `<es>`; same Cartesia voice, Spanish synthesis.
- **00:34.62 Priya**: `[Speaker 3]`. Mid-sentence code-switching Hindi/English in ONE utterance, transcribed as one segment. Tools: identify_speaker(3, Priya, hi); add_order_item(3, paneer tikka, 1, 'extra spicy'); add_order_item(3, mango lassi, 1).
- **00:43.66 Babel**: Reply tagged `<hi>`.
- **00:51.45 Babel**: Babel keeps talking...
- **00:54.05 Sam**: BARGE-IN. Sam talks over Babel: Silero VAD speech START (prob > 0.85, echo of Babel's own audio stays < 0.75) -> in-flight turn task cancelled, Cartesia context cancelled, send queue drained, `clearAudio` sent to Plivo, turn_complete(barge_in=true). Muse still endpoints Sam's sentence as `[Speaker 1 (Sam)]`.
- **00:58.79 Babel**: Tools: add_order_item(1, tacos al pastor, 3, 'replaces the earlier two').
- **01:02.75 Carlos**: `[Speaker 2 (Carlos)]`. Tool: get_order_summary() -> grouped by speaker label.
- **01:05.61 Babel**: Read-back grouped by diarized speaker, in Carlos's language.
- **01:17.17 Priya**: `[Speaker 3 (Priya)]`, code-switched again. Tools: send_sms(caller number, ticket); end_call('order confirmed').
- **01:21.83 Babel**: Reply tagged `<hi>`. end_call waits for Plivo `playedStream` so the goodbye finishes before the session ends. call_summary: 8 turns, 3 speakers, 1 barge-in.

## Structured events emitted during this call

| Event | Count | Notes |
|---|---|---|
| `call_answered` | 1 | session start, SIP headers |
| `speaker_detected` | 3 | first time Muse labels Speaker 1, 2, 3 |
| `speaker_identified` | 3 | names learned, Muse keyword biasing updated |
| `user_text` | 7 | one per committed turn, speaker-tagged text |
| `agent_text` | 8 | fired as soon as the LLM reply is known, before TTS |
| `turn_complete` | 8 | per-turn llm_ms / tts_ttfb_ms / playback_ms, one with barge_in=true |
| `call_summary` | 1 | turns, speakers, barge-ins, TTFS avg, errors |
