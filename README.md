# Gemini-VoiceBot
A Python-based voice assistant using Google Gemini for speech-to-text, text-to-speech, and intelligent multi-turn conversations.

## 🚀 Features

- 🎙 **Voice Input** using SpeechRecognition + PyAudio  
- 🔊 **Text-to-Speech** via Gemini audio output (with fallback to pyttsx3)  
- 🧠 **Multi-turn conversation** using Gemini text model  
- 🗒 **Notes storage**  
- ⏰ **Timers & Reminders**  
- 📅 **Calendar event creation (.ics files)**  
- 🔁 **Fallback mechanisms** for STT and TTS  
- 🛑 **Interrupt handling** (“stop”, “cancel”)  
- 📂 Persistent files for notes & events  

---

## 📦 Requirements

Create a file named **`requirements.txt`** and include the following:

```txt
google-generativeai
SpeechRecognition
PyAudio
pygame
pyttsx3
python-dotenv
dateparser
