import os
import time
import json
from dotenv import load_dotenv
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from groq import Groq
from gtts import gTTS
import pygame

# Import tool configurations
from tools import TOOLS_SCHEMA, AVAILABLE_TOOLS

# 1. Load Environment Variables
load_dotenv()

# 2. Initialize Groq API Client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Initialize Pygame Audio Engine
pygame.mixer.init()

# Groq Orpheus Expressive TTS Voices
GROQ_ORPHEUS_VOICES = {
    "1": ("Troy (Male)", "troy"),
    "2": ("Austin (Male)", "austin"),
    "3": ("Daniel (Male)", "daniel"),
    "4": ("Hannah (Female)", "hannah"),
    "5": ("Autumn (Female)", "autumn"),
    "6": ("Diana (Female)", "diana")
}

# Free Google TTS Fallback Presets
GTTS_ACCENTS = {
    "7": ("Google TTS - American English", "us"),
    "8": ("Google TTS - British English", "co.uk")
}


def record_audio(filename="input.wav", duration=5, samplerate=16000):
    """Records audio from microphone and saves as WAV."""
    print(f"\n🎙️ Recording for {duration} seconds... Speak now!")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    
    wav.write(filename, samplerate, audio_data)
    print("✅ Recording complete!")
    return filename


def transcribe_audio(filename="input.wav"):
    """Sends WAV file to Groq Whisper for transcription."""
    print("⏳ Transcribing audio with Groq Whisper...")
    with open(filename, "rb") as file:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, file.read()),
            model="whisper-large-v3",
            response_format="json",
            language="en"
        )
    return transcription.text.strip()


def get_llm_response_with_tools(user_text, history=None):
    """
    Handles Multi-Tool Calling:
    1. Sends user request + prior turns from this session + tool schemas to Groq Llama 3.3.
    2. Executes tools locally if requested.
    3. Feeds outputs back for final response synthesis.

    `history` is a list of (user_text, ai_text) tuples from earlier in this
    run, oldest first, so the agent actually remembers previous turns.
    """
    print("\n🧠 Thinking & Evaluating Tools...")
    messages = [
        {"role": "system", "content": "You are a concise voice AI assistant. Answer briefly (1-2 sentences)."}
    ]

    for past_user, past_ai in (history or []):
        messages.append({"role": "user", "content": past_user})
        messages.append({"role": "assistant", "content": past_ai})

    messages.append({"role": "user", "content": user_text})

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto"
    )

    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        print(f"\n🛠️ [MULTI-TOOL TRIGGERED]: Agent requested {len(tool_calls)} tool call(s).")
        messages.append(response_message)

        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            print(f"   ➔ Executing Tool: {function_name}({function_args})")
            
            tool_function = AVAILABLE_TOOLS.get(function_name)
            if tool_function:
                tool_result = tool_function(**function_args)
            else:
                tool_result = json.dumps({"error": f"Tool '{function_name}' not found."})

            print(f"   ✔ Result: {tool_result}")

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": tool_result,
            })

        second_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return second_response.choices[0].message.content
    
    return response_message.content


def play_audio_file(filepath):
    """Plays generated audio file using Pygame."""
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.music.unload()


def speak_with_groq_orpheus(text, voice_persona, output_file="response.wav"):
    """Generates audio using Groq's Orpheus English TTS model."""
    print(f"\n✨ Synthesizing with Groq Orpheus TTS ({voice_persona})...")
    try:
        response = groq_client.audio.speech.create(
            model="canopylabs/orpheus-v1-english",
            voice=voice_persona,
            input=text,
            response_format="wav"
        )
        response.write_to_file(output_file)
        play_audio_file(output_file)
        print("✅ Playback finished!")
    except Exception as e:
        print(f"❌ Groq Orpheus Error: {e}")


def speak_with_gtts(text, tld, output_file="response.mp3"):
    """Generates audio using free Google TTS."""
    print("\n🌐 Synthesizing with Google TTS...")
    try:
        tts = gTTS(text=text, lang='en', tld=tld, slow=False)
        tts.save(output_file)
        play_audio_file(output_file)
        print("✅ Playback finished!")
    except Exception as e:
        print(f"❌ Google TTS Error: {e}")


def voice_menu_and_speak(text):
    """Presents voice selection menu."""
    print("\n----------------------------------------")
    print("🎭 SELECT A VOICE PERSONA:")
    print("--- Groq Orpheus Voices ---")
    for key, (name, _) in GROQ_ORPHEUS_VOICES.items():
        print(f"  [{key}] {name}")
        
    print("--- Google TTS (Fallback) ---")
    for key, (name, _) in GTTS_ACCENTS.items():
        print(f"  [{key}] {name}")
    print("----------------------------------------")
    
    choice = input("Enter choice (1-8): ").strip()
    
    if choice in GROQ_ORPHEUS_VOICES:
        _, voice_persona = GROQ_ORPHEUS_VOICES[choice]
        speak_with_groq_orpheus(text, voice_persona)
    elif choice in GTTS_ACCENTS:
        _, tld = GTTS_ACCENTS[choice]
        speak_with_gtts(text, tld)
    else:
        print("⏩ Skipped voice playback.")


def capture_input():
    """Selects between Speech-to-Text (Mic) and Text-to-Text (Keyboard)."""
    print("========================================")
    print("  SELECT INPUT MODE:")
    print("  [1] 🎙️ Speech-to-Text (Speak into Mic)")
    print("  [2] 💬 Text-to-Text   (Type via Keyboard)")
    print("========================================")
    mode = input("Enter 1 or 2 (default 1): ").strip()

    if mode == "2":
        user_text = input("\n💬 Type your prompt here: ").strip()
        return user_text
    else:
        while True:
            input("\n👉 Press Enter to start recording (5 seconds)...")
            audio_file = record_audio(duration=5)
            raw_text = transcribe_audio(audio_file)
            
            print(f"\n----------------------------------------")
            print(f"🗣️ Transcribed Input: \"{raw_text}\"")
            print(f"----------------------------------------")
            
            choice = input("\nPress [Enter] to approve, [r] to re-record, or type manual corrections: ").strip()
            
            if choice.lower() == 'r':
                print("🔄 Retrying recording...")
                continue
            elif choice == "":
                return raw_text
            else:
                print(f"✏️ Using edited prompt: \"{choice}\"")
                return choice


if __name__ == "__main__":
    # Session memory: every (question, answer) pair from this run, oldest
    # first. Lives only for as long as the script is running (same idea as
    # the Gradio app's session state) and is passed into the agent so it
    # actually remembers earlier turns instead of treating each one as new.
    session_history = []

    while True:
        # 1. Get Input (Voice OR Text)
        approved_text = capture_input()

        # 2. Process with Multi-Tool Execution Loop (now history-aware)
        ai_reply = get_llm_response_with_tools(approved_text, session_history)
        session_history.append((approved_text, ai_reply))

        # 3. Output Answer to Terminal
        print(f"\n========================================")
        print(f"🤖 AI Response:\n\"{ai_reply}\"")
        print(f"========================================\n")

        # 4. Optional Voice Playback
        tts_prompt = input("🔊 Would you like the AI to speak this answer aloud? (y/N): ").strip().lower()

        if tts_prompt == 'y':
            voice_menu_and_speak(ai_reply)
        else:
            print("⏩ Skipped audio synthesis.")

        # 5. Continue the session or exit
        again = input("\n🔁 Ask another question? (Y/n): ").strip().lower()
        if again == 'n':
            print("👋 Session ended.")
            break