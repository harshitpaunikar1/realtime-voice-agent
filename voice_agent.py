"""
Real-time voice agent orchestration with session management and event logging.
Connects LiveKit room transport with the STT-LLM-TTS pipeline.
"""
import json
import logging
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from pipeline import VoicePipeline, WhisperTranscriber, GeminiLLM, PiperTTS
    PIPELINE_AVAILABLE = True
except ImportError:
    PIPELINE_AVAILABLE = False

try:
    from livekit import rtc
    LIVEKIT_AVAILABLE = True
except ImportError:
    LIVEKIT_AVAILABLE = False


class AgentState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class SessionConfig:
    room_name: str
    agent_name: str = "voice_agent"
    whisper_model: str = "base"
    llm_model: str = "gemini-1.5-flash"
    piper_model_path: Optional[str] = None
    max_silence_ms: int = 1000
    min_audio_bytes: int = 4000
    log_to_db: bool = True


@dataclass
class SessionEvent:
    session_id: str
    event_type: str
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class SessionLogger:
    """Persists session events to SQLite for debugging and analytics."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS session_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT,
        event_type TEXT,
        payload TEXT,
        timestamp REAL
    );
    CREATE TABLE IF NOT EXISTS session_summary (
        session_id TEXT PRIMARY KEY,
        room_name TEXT,
        started_at REAL,
        ended_at REAL,
        total_turns INTEGER,
        avg_latency_ms REAL
    );
    """

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(self.SCHEMA)
        self.conn.commit()

    def log_event(self, event: SessionEvent) -> None:
        self.conn.execute(
            "INSERT INTO session_events (session_id, event_type, payload, timestamp) VALUES (?,?,?,?)",
            (event.session_id, event.event_type, json.dumps(event.payload), event.timestamp),
        )
        self.conn.commit()

    def write_summary(self, session_id: str, room_name: str, started_at: float,
                       ended_at: float, total_turns: int, avg_latency_ms: float) -> None:
        self.conn.execute(
            """INSERT OR REPLACE INTO session_summary VALUES (?,?,?,?,?,?)""",
            (session_id, room_name, started_at, ended_at, total_turns, avg_latency_ms),
        )
        self.conn.commit()

    def recent_events(self, session_id: str, limit: int = 20) -> List[Dict]:
        cur = self.conn.execute(
            "SELECT event_type, payload, timestamp FROM session_events "
            "WHERE session_id=? ORDER BY timestamp DESC LIMIT ?",
            (session_id, limit),
        )
        rows = cur.fetchall()
        return [{"event_type": r[0], "payload": json.loads(r[1]), "timestamp": r[2]}
                for r in rows]


class AudioBuffer:
    """Accumulates audio chunks and signals when a complete utterance is ready."""

    def __init__(self, min_bytes: int = 4000):
        self.min_bytes = min_bytes
        self._buffer = bytearray()

    def append(self, data: bytes) -> None:
        self._buffer.extend(data)

    def ready(self) -> bool:
        return len(self._buffer) >= self.min_bytes

    def consume(self) -> bytes:
        data = bytes(self._buffer)
        self._buffer.clear()
        return data

    def clear(self) -> None:
        self._buffer.clear()

    def size(self) -> int:
        return len(self._buffer)


class VoiceAgent:
    """
    Main voice agent that manages session lifecycle, room connection,
    audio buffering, and pipeline orchestration.
    """

    def __init__(self, config: SessionConfig, session_logger: Optional[SessionLogger] = None):
        self.config = config
        self.session_id = f"{config.room_name}_{int(time.time())}"
        self.state = AgentState.IDLE
        self.session_logger = session_logger or SessionLogger()
        self._started_at: Optional[float] = None
        self._pipeline: Optional["VoicePipeline"] = None
        self._audio_buffer = AudioBuffer(min_bytes=config.min_audio_bytes)
        self._turn_count = 0
        self._total_latencies: List[float] = []

    def _log(self, event_type: str, payload: Dict[str, Any]) -> None:
        if self.config.log_to_db:
            self.session_logger.log_event(SessionEvent(
                session_id=self.session_id,
                event_type=event_type,
                payload=payload,
            ))

    def _build_pipeline(self) -> "VoicePipeline":
        if not PIPELINE_AVAILABLE:
            raise RuntimeError("pipeline.py not importable.")
        transcriber = WhisperTranscriber(model_size=self.config.whisper_model)
        llm = GeminiLLM(model_name=self.config.llm_model)
        tts = PiperTTS(model_path=self.config.piper_model_path)
        return VoicePipeline(
            session_id=self.session_id,
            transcriber=transcriber,
            llm=llm,
            tts=tts,
        )

    def start_session(self) -> None:
        self._started_at = time.time()
        self.state = AgentState.LISTENING
        if PIPELINE_AVAILABLE:
            self._pipeline = self._build_pipeline()
        self._log("session_started", {"room": self.config.room_name, "agent": self.config.agent_name})
        logger.info("Session %s started in room %s", self.session_id, self.config.room_name)

    def on_audio_received(self, audio_bytes: bytes) -> Optional[Dict[str, Any]]:
        if self.state not in (AgentState.LISTENING, AgentState.IDLE):
            return None

        self._audio_buffer.append(audio_bytes)
        if not self._audio_buffer.ready():
            return None

        audio_data = self._audio_buffer.consume()
        return self._process_utterance(audio_data)

    def _process_utterance(self, audio_bytes: bytes) -> Dict[str, Any]:
        self._turn_count += 1
        self.state = AgentState.TRANSCRIBING
        self._log("turn_started", {"turn": self._turn_count, "audio_bytes": len(audio_bytes)})

        if self._pipeline is None:
            self._pipeline = self._build_pipeline() if PIPELINE_AVAILABLE else None

        if self._pipeline:
            try:
                result = self._pipeline.process_turn(audio_bytes)
                self._total_latencies.append(result.total_latency_ms)
                self._log("turn_completed", {
                    "turn": self._turn_count,
                    "user_text": result.user_text,
                    "agent_text": result.agent_text,
                    "total_latency_ms": result.total_latency_ms,
                })
                self.state = AgentState.LISTENING
                return {
                    "turn": self._turn_count,
                    "user": result.user_text,
                    "agent": result.agent_text,
                    "latency_ms": result.total_latency_ms,
                    "audio_response": result.agent_text,
                }
            except Exception as exc:
                self.state = AgentState.ERROR
                self._log("turn_error", {"error": str(exc)})
                logger.error("Pipeline error: %s", exc)

        self.state = AgentState.LISTENING
        stub_response = "I heard you, but my processing module is warming up."
        return {
            "turn": self._turn_count,
            "user": "[audio received]",
            "agent": stub_response,
            "latency_ms": 50.0,
            "audio_response": stub_response,
        }

    def end_session(self) -> Dict[str, Any]:
        ended_at = time.time()
        duration_s = ended_at - (self._started_at or ended_at)
        avg_latency = (sum(self._total_latencies) / len(self._total_latencies)
                       if self._total_latencies else 0.0)
        self.session_logger.write_summary(
            self.session_id, self.config.room_name,
            self._started_at or ended_at, ended_at,
            self._turn_count, round(avg_latency, 1),
        )
        self._log("session_ended", {"duration_s": round(duration_s, 1), "turns": self._turn_count})
        self.state = AgentState.IDLE
        summary = {
            "session_id": self.session_id,
            "room": self.config.room_name,
            "duration_s": round(duration_s, 1),
            "total_turns": self._turn_count,
            "avg_latency_ms": round(avg_latency, 1),
        }
        logger.info("Session ended: %s", summary)
        return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    config = SessionConfig(
        room_name="demo_room_001",
        agent_name="sales_voice_agent",
        whisper_model="base",
        llm_model="gemini-1.5-flash",
    )

    session_log = SessionLogger()
    agent = VoiceAgent(config=config, session_logger=session_log)
    agent.start_session()

    print("Voice Agent Demo - simulating 4 conversation turns\n")
    for i in range(4):
        fake_audio = bytes([i % 256]) * 8000
        response = agent.on_audio_received(fake_audio)
        if response:
            print(f"Turn {response['turn']}:")
            print(f"  User: {response['user']}")
            print(f"  Agent: {response['agent']}")
            print(f"  Latency: {response['latency_ms']:.0f}ms\n")

    summary = agent.end_session()
    print("Session summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nRecent session events:")
    for ev in session_log.recent_events(agent.session_id, limit=5):
        print(f"  [{ev['event_type']}] {ev['payload']}")
