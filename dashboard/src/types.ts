/* Generated from docs/contract.json. Do not edit by hand. */

export interface AlertState {
  "alert_id": string;
  "type": "distraction" | "microsleep" | "drowsiness" | "phone" | "no_face" | "road_rage" | "zoning_out" | "crash_candidate" | "checkin_due" | "break_due";
  "tier": "T1" | "T2" | "T3";
  "status": "queued" | "held" | "speaking" | "listening" | "closed" | "escalated";
  "text": string;
  "reply"?: string | null;
  "intent"?: "OK" | "NOT_OK" | "ANSWER" | "HELP" | "UNCLEAR" | "NONE" | null;
  "latency_ms"?: number | null;
}

export interface AudioFeatures {
  "loudness_db": number;
  "baseline_db": number;
  "shouting": boolean;
  "speaking": boolean;
  "swear_count_60s": number;
  "anger_voice_p"?: number | null;
}

export interface BioHR {
  "bpm": number;
  "rr_ms"?: Array<number>;
  "source": "suunto" | "strap" | "sim";
}

export interface CameraFrame {
  "jpeg_b64": string;
}

export interface CheckinSchedule {
  "interval_min": number;
  "next_checkin_ts": number;
  "next_break_ts"?: number | null;
}

export interface ContextRoad {
  "load": "LOW" | "MEDIUM" | "HIGH";
  "traffic"?: string | null;
  "next_maneuver"?: string | null;
  "dist_m"?: number | null;
}

export interface ControlInject {
  "event": "harsh_brake" | "crash" | "zoning_out";
}

export interface ControlMusic {
  "duck": boolean;
  "level": number;
}

export interface ControlReplay {
  "action": "play" | "pause" | "seek";
  "trip"?: string | null;
  "segment"?: number | null;
}

export interface ControlTrip {
  "action": "start" | "stop";
  "pretrip"?: Pretrip | null;
}

export interface DetectorEvent {
  "event_id": string;
  "type": "distraction" | "microsleep" | "drowsiness" | "phone" | "no_face" | "road_rage" | "zoning_out" | "crash_candidate" | "checkin_due" | "break_due";
  "tier": "T1" | "T2" | "T3";
  "score": number;
  "signals"?: Record<string, unknown>;
}

export interface DriverState {
  "face_present": boolean;
  "eye_closed_p": number;
  "eyes_closed_ms": number;
  "perclos_60s": number;
  "blink_rate_pm": number;
  "gaze_dispersion_30s": number;
  "head": HeadPose;
  "yawn_p": number;
  "yawns_10m": number;
  "anger_p": number;
  "zoning_visual_p": number;
  "phone_in_hand": boolean;
  "fps": number;
}

export interface HeadPose {
  "yaw": number;
  "pitch": number;
  "roll": number;
}

export interface MusicVolume {
  "volume": number;
  "playing": boolean;
}

export interface Pretrip {
  "planned_min": number;
  "sleep_h": number;
  "awake_h": number;
  "tiredness": number;
  "caffeine": boolean;
  "engagement": "trivia" | "fact" | "quiet";
  "lang": "ro" | "en";
  "emergency_contact": string;
  "resting_hr": number;
}

export interface TripSummary {
  "text": string;
  "stats"?: Record<string, unknown>;
}

export interface VehicleMotion {
  "accel_long_g": number;
  "accel_lat_g": number;
  "accel_mag_g": number;
  "speed_kmh": number;
  "lat": number;
  "lon": number;
  "road": "motorway" | "secondary";
  "source": "replay" | "sim";
}

export interface VoiceReply {
  "alert_id": string;
  "transcript": string;
  "intent": "OK" | "NOT_OK" | "ANSWER" | "HELP" | "UNCLEAR" | "NONE";
  "latency_ms": number;
  "path": "keyword" | "llm";
}

export interface VoiceSay {
  "alert_id": string;
  "text": string;
  "tier": "T1" | "T2" | "T3";
  "lang": "ro" | "en";
  "listen_s"?: number;
  "audio_key"?: string | null;
}

export interface VoiceStatus {
  "alert_id"?: string | null;
  "state": "speaking" | "listening" | "idle";
}

export const TOPICS = [
  "driver.state",
  "camera.frame",
  "detector.event",
  "audio.features",
  "voice.status",
  "voice.reply",
  "voice.say",
  "alert.state",
  "vehicle.motion",
  "control.music",
  "checkin.schedule",
  "trip.summary",
  "music.volume",
  "bio.hr",
  "context.road",
  "control.trip",
  "control.replay",
  "control.inject"
] as const
export type Topic = (typeof TOPICS)[number]

export interface TopicDataMap {
  "driver.state": DriverState;
  "camera.frame": CameraFrame;
  "detector.event": DetectorEvent;
  "audio.features": AudioFeatures;
  "voice.status": VoiceStatus;
  "voice.reply": VoiceReply;
  "voice.say": VoiceSay;
  "alert.state": AlertState;
  "vehicle.motion": VehicleMotion;
  "control.music": ControlMusic;
  "checkin.schedule": CheckinSchedule;
  "trip.summary": TripSummary;
  "music.volume": MusicVolume;
  "bio.hr": BioHR;
  "context.road": ContextRoad;
  "control.trip": ControlTrip;
  "control.replay": ControlReplay;
  "control.inject": ControlInject;
}

export type Src = "cv" | "voice" | "core" | "ui" | "hr" | "maps"

export interface Envelope<TData = Record<string, unknown>> {
  topic: string
  ts: number
  src: Src
  data: TData
}

export type TopicEnvelope<K extends Topic = Topic> = Omit<Envelope<TopicDataMap[K]>, 'topic'> & {
  topic: K
}

export type AnyTopicEnvelope = { [K in Topic]: TopicEnvelope<K> }[Topic]
