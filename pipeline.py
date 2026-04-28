"""
Real-time voice agent pipeline.
Orchestrates the STT-LLM-TTS loop with stage-level latency tracking.
"""
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    import faster_whisper
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    import subprocess
    PIPER_AVAILABLE = True
except ImportError:
    PIPER_AVAILABLE = False


@dataclass
class AudioChunk:
    session_id: str
    chunk_id: str
    audio_bytes: bytes
    sample_rate: int = 16000
    channels: int = 1
    captured_at: float = field(default_factory=time.time)


@dataclass
class TranscriptionResult:
    text: str
    language: str
    confidence: float
    duration_ms: float


@dataclass
class LLMResponse:
    text: str
    tokens_used: int
    latency_ms: float


@dataclass
class TTSResult:
    audio_bytes: bytes
    duration_ms: float
    character_count: int


@dataclass
class PipelineResult:
    session_id: str
    turn_id: int
    user_text: str
    agent_text: str
    stt_latency_ms: float
    llm_latency_ms: float
    tts_latency_ms: float
    total_latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        return (
            f"Turn {self.turn_id} | "
            f"STT: {self.stt_latency_ms:.0f}ms | "
            f"LLM: {self.llm_latency_ms:.0f}ms | "
            f"TTS: {self.tts_latency_ms:.0f}ms | "
            f"Total: {self.total_latency_ms:.0f}ms"
        )


class WhisperTranscriber:
    """Transcribes audio using faster-whisper or stub fallback."""

    def __init__(self, model_size: str = "base"):
        self.model_size = model_size
        self._model = None

    def _load(self) -> None:
        if not WHISPER_AVAILABLE or self._model is not None:
            return
        try:
            self._model = faster_whisper.WhisperModel(self.model_size, device="cpu",
                                                       compute_type="int8")
        except Exception as exc:
            logger.warning("Whisper load failed: %s", exc)

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        t0 = time.perf_counter()
        self._load()
        if self._model is not None:
            try:
                import numpy as np
                audio_np = (np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0)
                segments, info = self._model.transcribe(audio_np, beam_size=5)
                text = " ".join(s.text.strip() for s in segments)
                dur_ms = (time.perf_counter() - t0) * 1000
                return TranscriptionResult(
                    text=text,
                    language=info.language,
                    confidence=float(info.language_probability),
                    duration_ms=round(dur_ms, 1),
                )
            except Exception as exc:
                logger.error("Whisper transcribe error: %s", exc)
        stub_text = "Hello, I need help with my order status."
        dur_ms = (time.perf_counter() - t0) * 1000
        return TranscriptionResult(
            text=stub_text, language="en", confidence=0.90, duration_ms=round(dur_ms, 1)
        )


class GeminiLLM:
    """Generates agent responses via Gemini Flash or stub fallback."""

    SYSTEM_PROMPT = (
        "You are a helpful voice assistant. Keep responses concise and conversational. "
        "Respond in 1-3 sentences only. Do not use bullet points or headers."
    )

    def __init__(self, model_name: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        self.model_name = model_name
        self._model = None
        if GEMINI_AVAILABLE and api_key:
            try:
                genai.configure(api_key=api_key)
                self._model = genai.GenerativeModel(model_name)
            except Exception as exc:
                logger.warning("Gemini init failed: %s", exc)

    def generate(self, prompt: str, history: List[Dict[str, str]]) -> LLMResponse:
        t0 = time.perf_counter()
        if self._model is not None:
            try:
                full_prompt = self.SYSTEM_PROMPT + "\n\n" + prompt
                response = self._model.generate_content(full_prompt)
                text = response.text.strip()
                dur_ms = (time.perf_counter() - t0) * 1000
                return LLMResponse(text=text, tokens_used=len(text.split()),
                                   latency_ms=round(dur_ms, 1))
            except Exception as exc:
                logger.error("Gemini generate error: %s", exc)
        stub_text = self._stub_response(prompt)
        dur_ms = (time.perf_counter() - t0) * 1000
        return LLMResponse(text=stub_text, tokens_used=len(stub_text.split()),
                           latency_ms=round(dur_ms, 1))

    def _stub_response(self, prompt: str) -> str:
        lower = prompt.lower()
        if "order" in lower:
            return "Your order is currently being processed and should arrive within 2-3 business days."
        if "help" in lower or "support" in lower:
            return "I am here to help. Could you please describe the issue you are experiencing?"
        if "cancel" in lower:
            return "I can help you with the cancellation. Please confirm your order number to proceed."
        return "Thank you for reaching out. I will do my best to assist you with your query."


class PiperTTS:
    """Synthesizes speech using Piper TTS binary or stub fallback."""

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path

    def synthesize(self, text: str) -> TTSResult:
        t0 = time.perf_counter()
        audio_bytes = b""
        if self.model_path and PIPER_AVAILABLE:
            try:
                result = subprocess.run(
                    ["piper", "--model", self.model_path, "--output-raw"],
                    input=text.encode(),
                    capture_output=True,
                    timeout=10,
                )
                audio_bytes = result.stdout
            except Exception as exc:
                logger.warning("Piper TTS error: %s", exc)
        if not audio_bytes:
            audio_bytes = b"\x00\x00" * (len(text) * 100)
        dur_ms = (time.perf_counter() - t0) * 1000
        return TTSResult(
            audio_bytes=audio_bytes,
            duration_ms=round(dur_ms, 1),
            character_count=len(text),
        )


class VoicePipeline:
    """
    Orchestrates the full STT -> LLM -> TTS pipeline for a single session.
    """

    def __init__(self, session_id: str,
                 transcriber: Optional[WhisperTranscriber] = None,
                 llm: Optional[GeminiLLM] = None,
                 tts: Optional[PiperTTS] = None):
        self.session_id = session_id
        self.transcriber = transcriber or WhisperTranscriber()
        self.llm = llm or GeminiLLM()
        self.tts = tts or PiperTTS()
        self._history: List[Dict[str, str]] = []
        self._turn_count = 0
        self._results: List[PipelineResult] = []

    def process_turn(self, audio_bytes: bytes) -> PipelineResult:
        self._turn_count += 1
        t_total = time.perf_counter()

        transcript = self.transcriber.transcribe(audio_bytes)
        logger.info("STT [%dms]: %s", transcript.duration_ms, transcript.text)

        self._history.append({"role": "user", "content": transcript.text})
        prompt = "\n".join(f"{m['role'].upper()}: {m['content']}"
                           for m in self._history[-6:])

        llm_resp = self.llm.generate(prompt, self._history)
        logger.info("LLM [%dms]: %s", llm_resp.latency_ms, llm_resp.text[:60])
        self._history.append({"role": "assistant", "content": llm_resp.text})

        tts_resp = self.tts.synthesize(llm_resp.text)
        logger.info("TTS [%dms]: %d chars", tts_resp.duration_ms, tts_resp.character_count)

        total_ms = (time.perf_counter() - t_total) * 1000
        result = PipelineResult(
            session_id=self.session_id,
            turn_id=self._turn_count,
            user_text=transcript.text,
            agent_text=llm_resp.text,
            stt_latency_ms=transcript.duration_ms,
            llm_latency_ms=llm_resp.latency_ms,
            tts_latency_ms=tts_resp.duration_ms,
            total_latency_ms=round(total_ms, 1),
        )
        self._results.append(result)
        return result

    def session_stats(self) -> Dict[str, Any]:
        if not self._results:
            return {}
        import numpy as np
        totals = [r.total_latency_ms for r in self._results]
        return {
            "session_id": self.session_id,
            "turns": len(self._results),
            "avg_total_latency_ms": round(float(sum(totals) / len(totals)), 1),
            "p95_latency_ms": round(float(sorted(totals)[int(len(totals) * 0.95)]), 1),
            "avg_stt_ms": round(sum(r.stt_latency_ms for r in self._results) / len(self._results), 1),
            "avg_llm_ms": round(sum(r.llm_latency_ms for r in self._results) / len(self._results), 1),
            "avg_tts_ms": round(sum(r.tts_latency_ms for r in self._results) / len(self._results), 1),
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    pipeline = VoicePipeline(session_id="demo_session_001")

    test_utterances = [
        b"\x00\x01" * 8000,
        b"\x00\x02" * 8000,
        b"\x00\x03" * 8000,
    ]

    print("Voice Pipeline Demo\n")
    for audio in test_utterances:
        result = pipeline.process_turn(audio)
        print(result.summary())
        print(f"  User: {result.user_text}")
        print(f"  Agent: {result.agent_text}\n")

    print("Session stats:")
    stats = pipeline.session_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
