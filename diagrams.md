# Real-Time Voice Agent Diagrams

Generated on 2026-04-26T04:29:37Z from README narrative plus project blueprint requirements.

## STT → LLM → TTS pipeline latency breakdown

```mermaid
flowchart TD
    N1["Step 1\nScoped the product to one user and one AI agent in one room so latency and turn-ta"]
    N2["Step 2\nUsed LiveKit for real-time room transport, keeping audio capture and playback reli"]
    N1 --> N2
    N3["Step 3\nConnected faster-whisper for transcription, Gemini Flash for response generation, "]
    N2 --> N3
    N4["Step 4\nOrganized the backend around clear pipeline stages so delays in transcription, rea"]
    N3 --> N4
    N5["Step 5\nFocused on cumulative latency because even small delays across each stage add up q"]
    N4 --> N5
```

## LiveKit room architecture

```mermaid
flowchart LR
    N1["Inputs\nInbound API requests and job metadata"]
    N2["Decision Layer\nLiveKit room architecture"]
    N1 --> N2
    N3["User Surface\nAPI-facing integration surface described in the README"]
    N2 --> N3
    N4["Business Outcome\nInference or response latency"]
    N3 --> N4
```

## Evidence Gap Map

```mermaid
flowchart LR
    N1["Present\nREADME, diagrams.md, local SVG assets"]
    N2["Missing\nSource code, screenshots, raw datasets"]
    N1 --> N2
    N3["Next Task\nReplace inferred notes with checked-in artifacts"]
    N2 --> N3
```
