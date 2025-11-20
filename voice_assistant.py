"""
Voice Assistant using Google Gemini API (fixed message format)
Features:
 - Voice input (SpeechRecognition + PyAudio)
 - Speech-to-text using Gemini Audio model
 - Text-to-speech using Gemini Audio output
 - Multi-turn context (Gemini 'parts' format)
 - Notes, reminders, calendar events
 - Interrupt handling
"""

import os
import time
import queue
import json
import threading
from datetime import datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv

import speech_recognition as sr
import pyttsx3
import pygame
import google.generativeai as genai
import dateparser

# =============================
# LOAD ENVIRONMENT VARIABLES
# =============================
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise ValueError("Set GOOGLE_API_KEY in the .env file!")

genai.configure(api_key=API_KEY)

# =============================
# SETTINGS
# =============================
MODEL_TEXT = "gemini-2.0-flash"
MODEL_AUDIO = "gemini-2.0-flash-lite"   # Gemini Audio input model
TTS_MODEL = "gemini-2.0-flash-lite"

NOTES_FILE = Path("assistant_notes.json")
CALENDAR_DIR = Path("calendar")
CALENDAR_DIR.mkdir(exist_ok=True)

stop_speaking = threading.Event()

# pyttsx3 fallback TTS
local_tts = pyttsx3.init()
local_tts.setProperty("rate", 180)

# conversation history in Gemini-compatible format:
# each item is a dict with keys like: {"role":"user","parts":[{"text":"..."}]}
conversation = [
    {
        "role": "user",
        "parts": [
            {"text": "You are a helpful, polite voice assistant. Follow user commands and maintain conversation context."}
        ]
    }
]


# =============================
# SAVE NOTES
# =============================
def save_note(text):
    notes = []
    if NOTES_FILE.exists():
        try:
            notes = json.loads(NOTES_FILE.read_text())
        except Exception:
            notes = []
    notes.append({"timestamp": datetime.now().isoformat(), "note": text})
    NOTES_FILE.write_text(json.dumps(notes, indent=2))


# =============================
# CALENDAR EVENT
# =============================
def add_event(summary, start_dt):
    uid = str(int(time.time()))
    dtstart = start_dt.strftime("%Y%m%dT%H%M%S")
    dtend = (start_dt + timedelta(minutes=30)).strftime("%Y%m%dT%H%M%S")
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
UID:{uid}
DTSTART:{dtstart}
DTEND:{dtend}
SUMMARY:{summary}
END:VEVENT
END:VCALENDAR
"""
    filename = CALENDAR_DIR / f"event_{uid}.ics"
    filename.write_text(ics)
    return str(filename)


# =============================
# GEMINI SPEECH-TO-TEXT
# =============================
def transcribe_audio_google(audio_path):
    """
    Send audio bytes to Gemini audio model for transcription.
    Returns the transcribed text (or raises).
    """
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    model = genai.GenerativeModel(MODEL_AUDIO)

    # Gemini expects Blob style content for audio: provide mime_type + data keys.
    # We pass contents as a list with a single blob content entry.
    try:
        response = model.generate_content(
            contents=[
                {
                    "mime_type": "audio/wav",
                    "data": audio_bytes
                }
            ],
            generation_config={"temperature": 0.0},
        )
    except Exception as e:
        # Some SDK versions may expect a different wrapper — re-raise with context
        raise RuntimeError(f"Gemini STT failure: {e}") from e

    # Extract text robustly
    text = getattr(response, "text", None)
    if not text:
        # try candidate shapes
        try:
            candidates = getattr(response, "candidates", None)
            if candidates and len(candidates) > 0:
                text = getattr(candidates[0], "content", None) or getattr(candidates[0], "output", None)
        except Exception:
            text = None

    if not text:
        # Last fallback: convert response to str
        text = str(response)

    return text


# =============================
# GEMINI TTS
# =============================
def gemini_tts(text, outfile="tts_output.wav"):
    model = genai.GenerativeModel(TTS_MODEL)

    contents = [
        {
            "role": "user",
            "parts": [{"text": text}]
        }
    ]

    try:
        response = model.generate_content(
            contents=contents,
            generation_config={
                "response_mime_type": "audio/wav"
            }
        )
    except Exception as e:
        raise RuntimeError(f"Gemini TTS failed: {e}") from e

    # Gemini Flash returns audio under response.candidates[0].content[0].binary
    audio_data = None

    try:
        audio_data = response.candidates[0].content[0].raw_audio
    except:
        pass

    try:
        audio_data = response.candidates[0].content[0].binary
    except:
        pass

    try:
        audio_data = response.candidates[0].content[0].audio
    except:
        pass

    if not audio_data:
        raise RuntimeError("Gemini did not return audio in TTS response")

    with open(outfile, "wb") as f:
        f.write(audio_data)

    return outfile


def play_audio(path):
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(path)
        pygame.mixer.music.play()

        while pygame.mixer.music.get_busy():
            if stop_speaking.is_set():
                pygame.mixer.music.stop()
                break
            time.sleep(0.1)

        pygame.mixer.quit()
    except Exception as e:
        print("Audio play error:", e)


def speak(text):
    """
    Speak text using Gemini TTS if possible, otherwise fallback to local pyttsx3.
    Honors stop_speaking event.
    """
    stop_speaking.clear()
    try:
        audio_file = gemini_tts(text)
        play_audio(audio_file)
    except Exception as e:
        # Fallback to local TTS
        print("Falling back to local TTS... (reason: {})".format(e))
        try:
            local_tts.say(text)
            local_tts.runAndWait()
        except Exception as ee:
            print("Local TTS failed:", ee)


# =============================
# GEMINI TEXT CHAT (fixed format)
# =============================
def gemini_chat(user_text):
    conversation.append({
        "role": "user",
        "parts": [{"text": user_text}]
    })

    model = genai.GenerativeModel(MODEL_TEXT)

    try:
        response = model.generate_content(
            contents=conversation,
            generation_config={"temperature": 0.2}
        )
    except Exception as e:
        conversation.pop()
        raise RuntimeError(f"Gemini chat API error: {e}") from e

    assistant_text = None

    try:
        assistant_text = response.text
    except:
        pass

    if not assistant_text:
        try:
            assistant_text = response.candidates[0].content[0].text
        except:
            pass

    if not assistant_text:
        assistant_text = "I'm here."

    conversation.append({
        "role": "model",
        "parts": [{"text": assistant_text}]
    })

    return assistant_text


# =============================
# HANDLE COMMANDS
# =============================
def handle_command(text):
    lower = text.lower()

    # TIMER / REMINDER
    if "timer" in lower or "remind" in lower:
        import re
        m = re.search(r"(\d+)\s*(seconds|second|minutes|minute|hours|hour)", lower)
        if m:
            num = int(m.group(1))
            unit = m.group(2)
            seconds = num * (60 if "minute" in unit else 3600 if "hour" in unit else 1)

            msg_match = re.search(r"to (.+)$", lower)
            msg = msg_match.group(1) if msg_match else "your reminder"

            def worker():
                time.sleep(seconds)
                speak(f"Reminder: {msg}")

            threading.Thread(target=worker, daemon=True).start()
            return f"Okay, I will remind you in {num} {unit}."

        return "Tell me like: remind me in 5 minutes to check the oven."

    # NOTES
    if "note" in lower or "remember" in lower:
        save_note(text)
        return "Note saved."

    # CALENDAR
    if "event" in lower or "schedule" in lower or "calendar" in lower:
        dt = dateparser.parse(text)
        if not dt:
            return "I couldn't understand the time. Try: schedule meeting tomorrow at 3 pm."
        filepath = add_event(text, dt)
        return f"Event saved to {filepath}"

    # GENERAL CHAT
    try:
        return gemini_chat(text)
    except Exception as e:
        print("Chat error:", e)
        return "Sorry, I couldn't reach the chat service."


# =============================
# VOICE CAPTURE
# =============================
def listen_voice():
    r = sr.Recognizer()
    mic = sr.Microphone()

    with mic as source:
        r.adjust_for_ambient_noise(source, duration=0.6)
        print("Listening...")
        try:
            audio = r.listen(source, timeout=5, phrase_time_limit=10)
        except sr.WaitTimeoutError:
            return None

    # save wav file
    path = "input.wav"
    with open(path, "wb") as f:
        f.write(audio.get_wav_data())

    # Try Gemini STT first
    try:
        text = transcribe_audio_google(path)
        print("You said:", text)
        return text
    except Exception as e:
        print("Gemini STT failed, falling back to Google recognizer:", e)
        try:
            text = r.recognize_google(audio)
            print("You said (fallback):", text)
            return text
        except Exception as e2:
            print("Fallback recognizer failed:", e2)
            return None


# =============================
# MAIN LOOP
# =============================
def main():
    print("Voice Assistant (Gemini) started. Press Ctrl+C to exit.\n")

    try:
        while True:
            text = listen_voice()
            if not text:
                continue

            if text.lower() in ["stop", "cancel"]:
                stop_speaking.set()
                continue

            reply = handle_command(text)
            print("Assistant:", reply)
            speak(reply)

    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
