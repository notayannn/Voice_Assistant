# 🎙️ Voice AI Assistant

A tool-calling AI assistant that understands typed **and** spoken input, reasons with **Llama 3.3 70B** on Groq, pulls live data (weather, crypto, stocks) when it needs to, and can talk back in a choice of expressive voices — available both as a terminal app and a Gradio web UI.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Usage](#usage)
  - [Web UI (`app.py`)](#web-ui-apppy)
  - [Terminal App (`main.py`)](#terminal-app-mainpy)
- [How It Works](#how-it-works)
  - [Text-to-Text (TTT)](#text-to-text-ttt)
  - [Speech-to-Text (STT)](#speech-to-text-stt)
  - [Tool / Function Calling](#tool--function-calling)
  - [Text-to-Speech (TTS)](#text-to-speech-tts)
  - [Session Memory](#session-memory)
- [Available Tools](#available-tools)
- [Voice Options](#voice-options)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Voice AI Assistant is a lightweight, dependency-conscious assistant built entirely on **free or keyless external APIs** (aside from Groq, which needs one API key). It's designed to demonstrate a complete, real-world voice-agent pipeline — recording, transcription, reasoning with tool calling, and speech synthesis — without any heavyweight infrastructure. Everything runs from a single Python process; no database, no backend server beyond Gradio's own, no cloud deployment required.

It ships in two forms that share the same reasoning core:

| | Terminal (`main.py`) | Web UI (`app.py`) |
|---|---|---|
| Interface | Command line | Gradio browser app |
| Microphone | Local hardware via `sounddevice` | Browser mic via Gradio's audio component |
| Playback | `pygame` mixer | HTML5 `<audio>` via Gradio |
| Best for | Scripting, quick testing, low-level control | Demoing, day-to-day use, sharing with others |

## Features

- 💬 **Text or voice input** — type a question or speak it; both are treated identically once transcribed.
- 🧠 **Tool-aware reasoning** — the assistant autonomously decides when it needs live data and calls the right function to get it, instead of guessing or hallucinating.
- 🌦️ **Real-time data tools** — live weather (Open-Meteo), cryptocurrency prices (CoinGecko), and stock prices (Yahoo Finance) out of the box.
- 🔊 **Dual TTS engines** — expressive, multi-persona speech via Groq's Orpheus model, or a free Google TTS fallback with selectable accents.
- 🗣️ **Six Orpheus voice personas** to choose from, plus regional English accents via gTTS.
- 🧵 **Conversational memory** — every turn remembers everything said earlier in the same session, so natural follow-ups just work.
- ✏️ **Editable transcripts** — review, correct, or completely retype what the speech recognizer heard before it's ever sent to the model.
- 🖥️ **Clean, minimal web interface** — a single scrolling transcript, a pill-shaped input dock, and on-demand audio playback, with no visual clutter.
- 🔑 **Minimal setup** — only one API key (Groq) is required; every data tool used is free and keyless.

## Architecture

```
                       ┌─────────────────────────┐
   🎤 / ⌨️  Input  ───▶│   STT (Whisper, if       │
                       │   speaking) → plain text │
                       └────────────┬────────────┘
                                    ▼
                       ┌─────────────────────────┐
                       │  Reasoning Core          │
                       │  Llama 3.3 70B (Groq)    │
                       │  + session history       │
                       │  + tool schemas          │
                       └────────────┬────────────┘
                                    │
                     tool call? ────┼──── no tool needed
                                    ▼                 │
                       ┌─────────────────────────┐    │
                       │  tools.py                │    │
                       │  weather / crypto / stock│    │
                       └────────────┬────────────┘    │
                                    ▼                 ▼
                       ┌─────────────────────────────────┐
                       │   Final natural-language reply    │
                       └────────────────┬────────────────┘
                                        ▼
                       ┌─────────────────────────────────┐
                       │  Optional TTS (Orpheus / gTTS)    │
                       │  → spoken audio playback          │
                       └─────────────────────────────────┘
```

## Project Structure

```
.
├── app.py
├── main.py
├── tools.py
├── requirements.txt
├── .env
└── README.md
```

## Tech Stack

| Layer | Technology |
|---|---|
| LLM reasoning | [Groq](https://groq.com) — Llama 3.3 70B (`llama-3.3-70b-versatile`) |
| Speech-to-text | Groq — Whisper (`whisper-large-v3`) |
| Text-to-speech | Groq — Orpheus (`canopylabs/orpheus-v1-english`) + [gTTS](https://pypi.org/project/gTTS/) fallback |
| Web interface | [Gradio](https://www.gradio.app/) |
| Local audio (CLI) | `sounddevice`, `scipy.io.wavfile`, `pygame` |
| Live data | [Open-Meteo](https://open-meteo.com/), [CoinGecko](https://www.coingecko.com/en/api), [yfinance](https://pypi.org/project/yfinance/) |
| Secrets management | `python-dotenv` |

## Getting Started

### Prerequisites

- Python 3.10 or later
- A free Groq API key
- A working microphone (only needed for speech input)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/<notayannn>/<Voice_Assistant>.git
cd <Voice_Assistant>

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

That's the only secret this project needs — every data tool (weather, crypto, stocks) uses a free, keyless public API.

## Usage

### Web UI (`app.py`)

```bash
python app.py
```

Gradio will start a local server and print a URL (typically `http://127.0.0.1:7860`) — open it in your browser. Type a message or tap the microphone icon to speak, then press **Send**. Once a reply appears, pick a voice and press **🔊 Hear response** to listen to it.

### Terminal App (`main.py`)

```bash
python main.py
```

You'll be asked to choose speech or typed input for each turn, review/confirm transcriptions before they're sent, and optionally hear each reply spoken aloud. The session keeps going — and keeps remembering — until you choose to stop.

## How It Works

### Text-to-Text (TTT)

Every message — typed or transcribed — is sent to Llama 3.3 70B alongside a system prompt and the full conversation so far. The model replies directly if it already knows the answer, or requests a tool call first if it needs live data.

### Speech-to-Text (STT)

Spoken input is recorded (locally via `sounddevice` in the CLI, or via the browser's own microphone in the web UI) and sent to Groq's `whisper-large-v3` model for transcription at a fixed 16kHz mono sample rate — the format Whisper is trained on. The transcript is shown back to you for review before anything is sent to the LLM.

### Tool / Function Calling

The model is given a schema describing three tools (weather, crypto price, stock price) and decides for itself, per message, whether calling one is necessary (`tool_choice="auto"`). If it calls a tool, the corresponding Python function runs locally, its result is fed back to the model, and the model is called a second time to turn that raw data into a natural sentence.

### Text-to-Speech (TTS)

Speech is generated only when explicitly requested — never automatically — using either Groq's expressive Orpheus model or a free Google TTS fallback, and played back through the browser (web UI) or the system's speakers (CLI).

### Session Memory

Every (question, answer) pair from the current session is kept in memory and resent with each new request, so the assistant has real context for follow-up questions. This memory is entirely in-process: it resets when the app restarts (CLI) or the page is reloaded (web UI) — nothing is written to disk or a database.

## Available Tools

| Tool | Description | Data Source |
|---|---|---|
| `get_weather` | Current temperature and wind speed for any city | Open-Meteo |
| `get_crypto_price` | Live price of any cryptocurrency in a given currency | CoinGecko |
| `get_stock_price` | Latest closing price for a stock ticker | Yahoo Finance (`yfinance`) |

## Voice Options

| Engine | Voices | Notes |
|---|---|---|
| Groq Orpheus | Troy, Austin, Daniel, Hannah, Autumn, Diana | Expressive, higher quality; uses Groq quota |
| Google TTS | American, British *(app.py also offers Australian & Indian in the CLI)* | Free, unlimited, no API key required |

## Known Limitations

- Speech recordings in the CLI are a **fixed 5-second window** — there's no automatic silence/voice-activity detection, so longer sentences can get cut off.
- The LLM's `temperature` is not explicitly configured anywhere, so it runs on Groq's API default rather than a value tuned for this use case.
- Free-tier limits apply to CoinGecko, Open-Meteo, `yfinance`, and gTTS — expect occasional rate-limiting or slower responses under heavy use.
- Session memory is not persisted — closing the terminal app or reloading the web page clears the conversation.

## Roadmap

- [ ] Voice-activity detection for variable-length recordings
- [ ] Persistent chat history (local file or database)
- [ ] Streaming responses for lower perceived latency
- [ ] Additional tools (news, currency conversion, calendar/reminders)
- [ ] Configurable system prompt / persona from the UI

## Contributing

Contributions are welcome. If you'd like to add a feature or fix a bug:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Commit your changes with a clear message
4. Open a pull request describing what you changed and why

Please keep new tools stateless and keyless where possible, consistent with the rest of `tools.py`.

## Acknowledgments

- [Groq](https://groq.com) for Llama 3.3, Whisper, and Orpheus inference
- [Open-Meteo](https://open-meteo.com/), [CoinGecko](https://www.coingecko.com/en/api), and [Yahoo Finance](https://finance.yahoo.com/) for free public data APIs
- [Gradio](https://www.gradio.app/) for the web UI framework