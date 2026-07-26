import os
import json
import tempfile
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import gradio as gr

# Import tool configurations
from tools import TOOLS_SCHEMA, AVAILABLE_TOOLS

# 1. Load API Key
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Initialize Groq Client
groq_client = Groq(api_key=GROQ_API_KEY)

# Available Voice Personas
ORPHEUS_VOICES = {
    "Troy (Male)": "troy",
    "Austin (Male)": "austin",
    "Daniel (Male)": "daniel",
    "Hannah (Female)": "hannah",
    "Autumn (Female)": "autumn",
    "Diana (Female)": "diana"
}

GTTS_ACCENTS = {
    "Google TTS (American)": "us",
    "Google TTS (British)": "co.uk"
}


def transcribe_audio_file(audio_path):
    """Transcribes audio file using Groq Whisper. Returns empty string if cleared."""
    if not audio_path:
        return ""
    try:
        with open(audio_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model="whisper-large-v3",
                response_format="json",
                language="en"
            )
        return transcription.text.strip()
    except Exception:
        return ""


def process_agent_query(user_text, history=None):
    """Passes user prompt (plus prior turns from this session) to Llama 3.3
    and executes tools silently. `history` is a list of (user_text, ai_text)
    tuples from earlier in the session, oldest first."""
    if not user_text or not user_text.strip():
        return "Please enter a message or record audio."

    messages = [
        {"role": "system", "content": "You are a concise voice AI assistant. Keep responses brief (1-2 sentences)."}
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
        messages.append(response_message)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)

            tool_func = AVAILABLE_TOOLS.get(func_name)
            if tool_func:
                res = tool_func(**func_args)
            else:
                res = json.dumps({"error": "Tool not found"})

            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": res,
            })

        second_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages
        )
        return second_response.choices[0].message.content

    return response_message.content


def render_transcript(history):
    """Renders the whole session's turns as plain flowing text (no bubbles/boxes),
    oldest first. Returns the placeholder if nothing has been asked yet."""
    if not history:
        return "*AI response will appear here...*"

    blocks = []
    for user_text, ai_text in history:
        blocks.append(f"**You**\n\n{user_text}\n\n{ai_text}")
    return "\n\n---\n\n".join(blocks)


def generate_speech(text, voice_choice):
    """Synthesizes text to speech using Groq Orpheus or gTTS."""
    if not text or not text.strip():
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = temp_file.name
    temp_file.close()

    try:
        if voice_choice in ORPHEUS_VOICES:
            persona = ORPHEUS_VOICES[voice_choice]
            response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice=persona,
                input=text,
                response_format="wav"
            )
            response.write_to_file(output_path)
        elif voice_choice in GTTS_ACCENTS:
            tld = GTTS_ACCENTS[voice_choice]
            tts = gTTS(text=text, lang="en", tld=tld, slow=False)
            output_path = output_path.replace(".wav", ".mp3")
            tts.save(output_path)

        return output_path
    except Exception:
        return None


# ==========================================
# MINIMAL, ROUNDED, LOW-CLUTTER STYLING
# ==========================================
custom_css = """
:root {
    --radius-md: 14px;
    --radius-lg: 18px;
}

.gradio-container {
    max-width: 720px !important;
    margin: 0 auto !important;
}

.centered-container {
    padding: 32px 12px 150px 12px;
}

.title-header {
    text-align: center;
    font-weight: 600;
    letter-spacing: -0.02em;
    margin-bottom: 28px;
    opacity: 0.95;
}

/* Response lives in open space, same tone as the page — not a boxed card.
   The "processing" indicator renders inside this same block, so it
   inherits the transparent background too instead of sitting in a box. */
.response-area, .response-area .prose, .response-area * {
    background: transparent !important;
    box-shadow: none !important;
    border: none !important;
}
.response-area {
    min-height: 80px;
    padding: 18px 4px;
    line-height: 1.55;
}

/* Voice player row sits directly under the response, same visual family,
   not a separate boxed accordion */
.voice-row {
    margin-top: 10px;
    padding: 8px 10px;
    border-radius: var(--radius-md);
    background: var(--background-fill-secondary);
    opacity: 0.9;
}
.voice-row .gr-dropdown, .voice-row button {
    border-radius: 999px !important;
}

/* Rounded corners everywhere: inputs, buttons, audio widgets */
textarea, input[type="text"], .gr-box, .gr-input, .gr-form,
button, .gr-button, .gr-dropdown, .gr-audio, .wrap.svelte-1cl284s {
    border-radius: var(--radius-md) !important;
}

/* Bottom input dock: pill-shaped, single row, no drop shadow clutter.
   overflow is visible so the audio recorder's control row (stop/play/trash)
   never gets clipped by the pill boundary. */
.bottom-dock {
    position: fixed !important;
    bottom: 22px !important;
    left: 50% !important;
    transform: translateX(-50%) !important;
    width: calc(100% - 24px) !important;
    max-width: 720px !important;
    z-index: 9999 !important;
    background-color: var(--background-fill-primary) !important;
    padding: 8px !important;
    border-radius: 999px !important;
    box-shadow: 0px 6px 24px rgba(0, 0, 0, 0.18) !important;
    border: 1px solid var(--border-color-primary) !important;
    overflow: visible !important;
}

/* While recording, the dock grows into a rounded rectangle instead of a
   pill so the waveform + control row (stop/play/trash) has full room and
   nothing sits half outside the boundary */
.bottom-dock.recording-mode {
    border-radius: var(--radius-lg) !important;
    padding: 14px !important;
    align-items: stretch !important;
}
.bottom-dock.recording-mode .gr-audio,
.bottom-dock.recording-mode [data-testid="waveform-audio"] {
    overflow: visible !important;
}

.bottom-dock textarea {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}
.bottom-dock .gr-audio {
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
    overflow: visible !important;
}

.mic-btn, .send-btn {
    border-radius: 999px !important;
    min-width: 46px !important;
}

.mic-btn {
    font-size: 22px !important;
    line-height: 1 !important;
    padding: 0 !important;
}

/* Round the inner input wrapper too, so it doesn't collide with the
   pill-shaped outer dock — same radius, transparent so it reads as one
   continuous pill rather than a box-in-a-box */
.bottom-dock .block,
.bottom-dock .form,
.bottom-dock > div {
    border-radius: 999px !important;
    border: none !important;
    background: transparent !important;
    box-shadow: none !important;
}

/* Gradio's default footer: Use via API / Built with Gradio / Settings */
footer {
    display: none !important;
}
"""

with gr.Blocks(title="AI Voice Assistant", css=custom_css, theme=gr.themes.Soft()) as demo:

    with gr.Column(elem_classes=["centered-container"]):
        gr.Markdown("## 🤖 Voice AI Assistant", elem_classes=["title-header"])

        # Main response area
        text_output = gr.Markdown(
            value="*AI response will appear here...*",
            elem_classes=["response-area"]
        )

        # Voice playback — a slim row directly under the response, hidden until there's an answer
        with gr.Row(visible=False, elem_classes=["voice-row"]) as audio_panel:
            voice_select = gr.Dropdown(
                choices=list(ORPHEUS_VOICES.keys()) + list(GTTS_ACCENTS.keys()),
                value="Troy (Male)",
                label=None,
                show_label=False,
                container=False,
                scale=4
            )
            speak_btn = gr.Button("🔊 Hear response", scale=2, elem_classes=["send-btn"])
        audio_output = gr.Audio(label=None, show_label=False, autoplay=True, visible=False)

        # Bottom input dock: text mode and record mode share the same pill
        with gr.Row(equal_height=True, elem_classes=["bottom-dock"]) as dock_row:
            text_input = gr.Textbox(
                placeholder="Ask anything...",
                show_label=False,
                scale=8,
                container=False,
                visible=True
            )
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                show_label=False,
                scale=8,
                container=False,
                visible=False
            )
            mode_btn = gr.Button("🎙️", scale=1, min_width=46, elem_classes=["mic-btn"])
            submit_btn = gr.Button("Send", variant="primary", scale=2, min_width=70, elem_classes=["send-btn"])

    # State: whether the dock is currently in "record" mode
    is_audio_mode = gr.State(False)

    # Session memory: every (question, answer) pair asked this page-load, plus
    # just the most recent answer (that's the only one "Hear response" ever
    # reads from). Both reset naturally on page reload since gr.State lives
    # server-side per session.
    chat_history = gr.State([])
    latest_answer = gr.State("")

    def toggle_mode(current_is_audio):
        """Switches the dock between typing and recording. Re-clicking the mic
        while a transcript is showing is how the user 'retries' a recording.
        Also swaps the dock's shape from a pill to a rounded rectangle while
        recording, so the waveform/stop/trash controls have full room instead
        of getting clipped at the pill's edge."""
        new_is_audio = not current_is_audio
        if new_is_audio:
            return (
                gr.update(visible=False),             # text_input
                gr.update(visible=True, value=None),  # audio_input (cleared for a fresh take)
                "⏺️",
                new_is_audio,
                gr.update(elem_classes=["bottom-dock", "recording-mode"]),
            )
        else:
            return (
                gr.update(visible=True),
                gr.update(visible=False),
                "🎙️",
                new_is_audio,
                gr.update(elem_classes=["bottom-dock"]),
            )

    mode_btn.click(
        fn=toggle_mode,
        inputs=[is_audio_mode],
        outputs=[text_input, audio_input, mode_btn, is_audio_mode, dock_row]
    )

    def handle_audio_recording(audio_path):
        """Once recording stops, transcribe it straight into the text box and
        flip back to text mode (and the pill shape) so the user can review,
        edit, or hit Send."""
        if not audio_path:
            return gr.update(), gr.update(), "🎙️", False, gr.update(elem_classes=["bottom-dock"])
        transcription = transcribe_audio_file(audio_path)
        return (
            gr.update(visible=True, value=transcription),  # text_input shows transcript
            gr.update(visible=False),                        # hide recorder
            "🎙️",
            False,
            gr.update(elem_classes=["bottom-dock"]),
        )

    audio_input.stop_recording(
        fn=handle_audio_recording,
        inputs=[audio_input],
        outputs=[text_input, audio_input, mode_btn, is_audio_mode, dock_row]
    )

    def submit_query(text, history):
        if not text or not text.strip():
            return render_transcript(history), history, "", gr.update(visible=False)
        answer = process_agent_query(text, history)
        history = history + [(text, answer)]
        return render_transcript(history), history, answer, gr.update(visible=True)

    submit_btn.click(
        fn=submit_query,
        inputs=[text_input, chat_history],
        outputs=[text_output, chat_history, latest_answer, audio_panel],
        show_progress="hidden"
    )

    speak_btn.click(
        fn=generate_speech,
        inputs=[latest_answer, voice_select],
        outputs=[audio_output]
    ).then(
        fn=lambda path: gr.update(visible=bool(path)),
        inputs=[audio_output],
        outputs=[audio_output]
    )

if __name__ == "__main__":
    demo.launch()