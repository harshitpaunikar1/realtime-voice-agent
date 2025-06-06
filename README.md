# Real-Time Voice Agent

This repository documents a room-based voice AI system where a user speaks naturally, the system transcribes the audio, generates a response, and plays it back as speech.

## Domain
Voice AI / Real-Time Systems

## Overview
Designed to keep the full STT-LLM-TTS loop understandable and fast enough to feel conversational.

## Methodology
1. Scoped the product to one user and one AI agent in one room so latency and turn-taking quality stayed visible from the beginning.
2. Used LiveKit for real-time room transport, keeping audio capture and playback reliable without building the media layer from scratch.
3. Connected faster-whisper for transcription, Gemini Flash for response generation, and Piper for lightweight speech synthesis.
4. Organized the backend around clear pipeline stages so delays in transcription, reasoning, or speech output could be traced separately.
5. Focused on cumulative latency because even small delays across each stage add up quickly and make voice interactions feel unnatural.
6. Logged room and pipeline events so session behaviour, response timing, and future tuning work had a concrete debugging trail.

## Skills
- LiveKit
- faster-whisper
- Gemini Flash
- Piper TTS
- FastAPI
- Voice Pipeline Orchestration
- Latency Optimization
- Session Logging

## Source
This README was generated from the portfolio project data used by `/Users/harshitpanikar/Documents/Test_Projs/harshitpaunikar1.github.io/index.html`.
