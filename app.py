import os
import json
import tempfile
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import streamlit as st

from tools import TOOLS_SCHEMA, AVAILABLE_TOOLS

# ------------------------------------------------------------------
# Setup
# ------------------------------------------------------------------
load_dotenv()
if "GROQ_API_KEY" not in os.environ:
    try:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

ORPHEUS_VOICES = {
    "Troy (Male)": "troy",
    "Austin (Male)": "austin",
    "Daniel (Male)": "daniel",
    "Hannah (Female)": "hannah",
    "Autumn (Female)": "autumn",
    "Diana (Female)": "diana",
}
GTTS_ACCENTS = {
    "Google TTS (American)": "us",
    "Google TTS (British)": "co.uk",
}
ALL_VOICES = list(ORPHEUS_VOICES.keys()) + list(GTTS_ACCENTS.keys())


# ------------------------------------------------------------------
# Core logic — STT, TTT + tool calling, TTS
# ------------------------------------------------------------------
def transcribe_audio_file(audio_file):
    """Speech-to-text via Groq Whisper. audio_file comes from st.audio_input.
    Raises on failure instead of swallowing the error, so the caller can show
    the real reason it failed rather than the box just doing nothing."""
    if not audio_file:
        return ""
    transcription = groq_client.audio.transcriptions.create(
        file=(audio_file.name or "recording.wav", audio_file.getvalue()),
        model="whisper-large-v3",
        response_format="json",
        language="en",
    )
    return transcription.text.strip()


def process_agent_query(user_text, history=None):
    """Reasoning core: sends the message + session history to Llama 3.3,
    executing weather/crypto/stock tools when the model asks for them."""
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
        tool_choice="auto",
    )
    response_message = response.choices[0].message
    tool_calls = response_message.tool_calls

    if tool_calls:
        messages.append(response_message)
        for tool_call in tool_calls:
            func_name = tool_call.function.name
            func_args = json.loads(tool_call.function.arguments)
            tool_func = AVAILABLE_TOOLS.get(func_name)
            result = tool_func(**func_args) if tool_func else json.dumps({"error": "Tool not found"})
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": func_name,
                "content": result,
            })

        second_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
        )
        return second_response.choices[0].message.content

    return response_message.content


def generate_speech(text, voice_choice):
    """Text-to-speech via Groq Orpheus or gTTS. Returns a file path."""
    if not text or not text.strip():
        return None

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    output_path = temp_file.name
    temp_file.close()

    try:
        if voice_choice in ORPHEUS_VOICES:
            response = groq_client.audio.speech.create(
                model="canopylabs/orpheus-v1-english",
                voice=ORPHEUS_VOICES[voice_choice],
                input=text,
                response_format="wav",
            )
            response.write_to_file(output_path)
        elif voice_choice in GTTS_ACCENTS:
            tts = gTTS(text=text, lang="en", tld=GTTS_ACCENTS[voice_choice], slow=False)
            output_path = output_path.replace(".wav", ".mp3")
            tts.save(output_path)
        return output_path
    except Exception:
        return None


# ------------------------------------------------------------------
# UI — clean and minimal
# ------------------------------------------------------------------
st.set_page_config(page_title="Voice Assistant", page_icon="🤖", layout="centered")

st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    [data-testid="stToolbar"] {visibility: hidden;}

    /* Streamlit dims/fades the whole app while a rerun is in progress —
       that's the "bright to dim" flashing. Force it to stay fully visible
       and skip the transition entirely. */
    [data-testid="stAppViewContainer"], [data-testid="stMain"], .main {
        opacity: 1 !important;
        transition: none !important;
    }

    /* Hiding the header above also hides Streamlit's own small circular
       "running" spinner (top-right) — explicitly bring just that back,
       since it's exactly the small side-indicator that's wanted here. */
    [data-testid="stStatusWidget"] {
        visibility: visible !important;
        opacity: 1 !important;
    }

    .block-container {max-width: 700px; padding-top: 2.5rem; padding-bottom: 220px;}
    .stTextInput input, .stButton button, .stSelectbox div[data-baseweb="select"] {
        border-radius: 999px !important;
    }
    div[data-testid="stAudioInput"] {border-radius: 16px !important; overflow: hidden;}

    /* The input dock (recorder + text box + Send) — pinned to the bottom of
       the viewport so chat history scrolls underneath it instead of pushing
       it down the page. st.container(key="input_dock") is what gets this
       class name. */
    .st-key-input_dock {
        position: fixed !important;
        bottom: 0;
        left: 50%;
        transform: translateX(-50%);
        width: calc(100% - 24px);
        max-width: 700px;
        z-index: 999;
        background-color: var(--background-color);
        padding: 14px 14px 18px 14px;
        border-radius: 20px 20px 0 0;
        box-shadow: 0 -6px 24px rgba(0, 0, 0, 0.12);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## Assistant Jared")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "latest_answer" not in st.session_state:
    st.session_state.latest_answer = ""
if "last_audio_bytes" not in st.session_state:
    st.session_state.last_audio_bytes = None
if "user_input_box" not in st.session_state:
    st.session_state.user_input_box = ""

# Transcript
if not st.session_state.chat_history:
    st.markdown("*Ask me anything...*")
else:
    for user_text, ai_text in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(user_text)
        with st.chat_message("assistant"):
            st.write(ai_text)

# Voice playback for the latest reply only
if st.session_state.latest_answer:
    col1, col2 = st.columns([3, 2])
    with col1:
        voice_choice = st.selectbox("Voice", ALL_VOICES, label_visibility="collapsed")
    with col2:
        if st.button("Hear Response", use_container_width=True):
            audio_path = generate_speech(st.session_state.latest_answer, voice_choice)
            if audio_path:
                st.audio(audio_path, autoplay=True)
            else:
                st.warning("Couldn't generate audio for that reply.")

st.divider()

if "audio_key_version" not in st.session_state:
    st.session_state.audio_key_version = 0

with st.container(key="input_dock"):
    # Mic input — auto-fills the text box for review before sending
    audio_value = st.audio_input(
        "Record a message",
        label_visibility="collapsed",
        key=f"audio_recorder_{st.session_state.audio_key_version}",
    )
    if audio_value is not None:
        audio_bytes = audio_value.getvalue()
        if audio_bytes != st.session_state.last_audio_bytes:
            st.session_state.last_audio_bytes = audio_bytes
            try:
                transcript = transcribe_audio_file(audio_value)
            except Exception as e:
                transcript = ""
                st.error(f"Transcription failed: {e}")
            if transcript:
                st.session_state.user_input_box = transcript
            # Swap in a brand-new recorder instance instead of ever showing
            # Streamlit the same completed clip again on a later rerun — this is
            # what was causing the stale "an error has occurred" in the recorder
            # after the reply had already come back fine.
            st.session_state.audio_key_version += 1
            st.rerun()

    def submit_message():
        text = st.session_state.user_input_box.strip()
        if not text:
            return
        answer = process_agent_query(text, st.session_state.chat_history)
        st.session_state.chat_history.append((text, answer))
        st.session_state.latest_answer = answer
        st.session_state.user_input_box = ""

    input_col, send_col = st.columns([4, 1])
    with input_col:
        st.text_input(
            "Ask anything...",
            key="user_input_box",
            label_visibility="collapsed",
            placeholder="Ask anything...",
            on_change=submit_message,
        )
    with send_col:
        st.button("Send", on_click=submit_message, use_container_width=True, type="primary")