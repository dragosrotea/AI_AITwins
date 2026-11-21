import { useEffect, useRef, useState, type FormEvent } from 'react'
import { bus } from './bus'
import { CrashSafetyNotice } from './CrashStatus'
import type { AlertState, CheckinSchedule, Pretrip, TopicEnvelope, VoiceStatus } from './types'

function useBusUpdates() {
  const [, setRevision] = useState(0)

  useEffect(() => {
    let timer: number | undefined
    const unsubscribe = bus.subscribe(() => {
      if (timer !== undefined) return
      timer = window.setTimeout(() => {
        timer = undefined
        setRevision((revision) => revision + 1)
      }, 150)
    })
    return () => {
      unsubscribe()
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])
}

function latest<K extends 'alert.state' | 'voice.say' | 'voice.status' | 'checkin.schedule' | 'vehicle.motion'>(topic: K) {
  return bus.getLatest(topic) as TopicEnvelope<K> | undefined
}

const TIER_RANK: Record<AlertState['tier'], number> = { T1: 1, T2: 2, T3: 3 }

function formatCountdown(seconds: number) {
  const safeSeconds = Math.max(0, seconds)
  const hours = Math.floor(safeSeconds / 3600)
  const minutes = Math.floor((safeSeconds % 3600) / 60)
  const remainder = safeSeconds % 60
  return hours > 0
    ? `${hours}:${minutes.toString().padStart(2, '0')}:${remainder.toString().padStart(2, '0')}`
    : `${minutes}:${remainder.toString().padStart(2, '0')}`
}

export function DriverStatusPanel() {
  useBusUpdates()
  const [activeAlerts, setActiveAlerts] = useState<Array<TopicEnvelope<'alert.state'>>>([])
  const [now, setNow] = useState(() => Date.now() / 1000)
  const seenAlert = useRef<TopicEnvelope<'alert.state'> | undefined>(undefined)

  useEffect(() => {
    function readAlert() {
      const message = latest('alert.state')
      if (!message || message === seenAlert.current) return
      seenAlert.current = message
      setActiveAlerts((current) => {
        const byId = new Map(current.map((item) => [item.data.alert_id, item]))
        if (message.data.status === 'closed') byId.delete(message.data.alert_id)
        else byId.set(message.data.alert_id, message)
        return [...byId.values()].slice(-30)
      })
    }

    readAlert()
    return bus.subscribe(readAlert)
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => window.clearInterval(timer)
  }, [])

  const alert = [...activeAlerts].sort((a, b) =>
    TIER_RANK[b.data.tier] - TIER_RANK[a.data.tier] || b.ts - a.ts,
  )[0]
  const say = latest('voice.say')
  const voiceStatus = latest('voice.status')?.data as VoiceStatus | undefined
  const voiceIsActive = voiceStatus?.state === 'speaking' || voiceStatus?.state === 'listening'
  const voiceMatchesAlert = !alert || !voiceStatus?.alert_id || voiceStatus.alert_id === alert.data.alert_id
  const sayMatchesVoice = Boolean(voiceIsActive && voiceMatchesAlert && (!voiceStatus?.alert_id || say?.data.alert_id === voiceStatus.alert_id))
  const spokenText = alert?.data.text || (sayMatchesVoice ? say?.data.text : undefined) || 'DriveMate is monitoring your drive.'
  const isSpeaking = voiceStatus?.state === 'speaking' && voiceMatchesAlert
  const voiceActivity = voiceMatchesAlert && voiceIsActive ? voiceStatus?.state : undefined
  const schedule = latest('checkin.schedule')?.data as CheckinSchedule | undefined
  const remaining = schedule ? Math.ceil(schedule.next_checkin_ts - now) : undefined
  const level = !alert ? 'clear' : alert.data.tier === 'T1' ? 'caution' : 'urgent'
  const statusLabel = level === 'clear' ? 'ALL CLEAR' : level === 'caution' ? 'CAUTION' : 'ATTENTION REQUIRED'

  return (
    <section className={`driver-status-card driver-status-card--${level}${isSpeaking ? ' driver-status-card--speaking' : ''}`} aria-label="Driver safety status">
        <div className="driver-status-topline">
          <span className="driver-status-indicator" aria-hidden="true" />
          <span>{statusLabel}</span>
          {voiceActivity && (
            <span className={`driver-voice-indicator${isSpeaking ? ' driver-voice-indicator--speaking' : ''}`} role="status" aria-label={isSpeaking ? 'DriveMate is speaking' : 'DriveMate is listening'}>
              <span className="driver-voice-wave" aria-hidden="true"><i /><i /><i /></span>
              {isSpeaking ? 'Speaking' : 'Listening'}
            </span>
          )}
        </div>
        <p className="driver-status-message" aria-live="polite">{spokenText}</p>
        <CrashSafetyNotice compact />
        <div className="driver-next-checkin">
          <span>NEXT CHECK-IN</span>
          <strong>
            {remaining === undefined
              ? 'Waiting for schedule'
              : remaining <= 0 ? 'Due now' : formatCountdown(remaining)}
          </strong>
        </div>
    </section>
  )
}

const INITIAL_PRETRIP: Pretrip = {
  planned_min: 120,
  sleep_h: 7,
  awake_h: 1,
  tiredness: 1,
  caffeine: false,
  engagement: 'quiet',
  lang: 'en',
  emergency_contact: '',
  resting_hr: 65,
}

export function PreTripForm() {
  const [pretrip, setPretrip] = useState<Pretrip>(INITIAL_PRETRIP)
  const [tripActive, setTripActive] = useState(false)
  const [speed, setSpeed] = useState(() => latest('vehicle.motion')?.data.speed_kmh ?? 0)
  const [message, setMessage] = useState<{ text: string; kind: 'success' | 'error' } | null>(null)
  const isMoving = speed > 5

  useEffect(() => {
    let timer: number | undefined
    const readSpeed = () => {
      if (timer !== undefined) return
      timer = window.setTimeout(() => {
        timer = undefined
        const currentSpeed = latest('vehicle.motion')?.data.speed_kmh ?? 0
        setSpeed((previous) => previous === currentSpeed ? previous : currentSpeed)
      }, 200)
    }
    readSpeed()
    const unsubscribe = bus.subscribe(readSpeed)
    return () => {
      unsubscribe()
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  function update<K extends keyof Pretrip>(key: K, value: Pretrip[K]) {
    setPretrip((current) => ({ ...current, [key]: value }))
  }

  function startTrip(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    if (isMoving) return
    if (!bus.publish('control.trip', { action: 'start', pretrip })) {
      setMessage({ text: 'Could not send trip start. Check the broker connection and try again.', kind: 'error' })
      return
    }
    setTripActive(true)
    setMessage({ text: 'Trip start sent to broker.', kind: 'success' })
  }

  function stopTrip() {
    if (isMoving) return
    if (!bus.publish('control.trip', { action: 'stop' })) {
      setMessage({ text: 'Could not send trip stop. Check the broker connection and try again.', kind: 'error' })
      return
    }
    setTripActive(false)
    setMessage({ text: 'Trip stop sent to broker.', kind: 'success' })
  }

  return (
    <section className="panel-placeholder driver-pretrip-panel" aria-labelledby="pretrip-title">
      <div className="pretrip-heading">
        <div>
          <p className="panel-label">TRIP SETUP</p>
          <h3 id="pretrip-title">Pre-trip check-in</h3>
          <p className="pretrip-description">Departure time is recorded automatically when you start.</p>
        </div>
        {isMoving && <span className="pretrip-lock" role="status">Locked while moving · {speed.toFixed(0)} km/h</span>}
      </div>

      {tripActive ? (
        <div className="pretrip-active" role="status">
          <span>Trip is active. Trip setup is locked until you stop the trip.</span>
          <button type="button" className="pretrip-stop" onClick={stopTrip} disabled={isMoving}>Stop trip</button>
        </div>
      ) : (
        <form
          className="pretrip-form"
          onSubmit={startTrip}
          onInvalidCapture={() => setMessage({ text: 'Complete the required fields and check the emergency contact format (+country code).', kind: 'error' })}
          onChange={() => setMessage(null)}
        >
          <fieldset disabled={isMoving}>
            <label>
              Planned drive (minutes)
              <input type="number" min="1" step="1" required value={pretrip.planned_min} onChange={(event) => update('planned_min', Number(event.currentTarget.value))} />
            </label>
            <label>
              Sleep last night (hours)
              <input type="number" min="0" max="24" step="0.25" required value={pretrip.sleep_h} onChange={(event) => update('sleep_h', Number(event.currentTarget.value))} />
            </label>
            <label>
              Hours awake
              <input type="number" min="0" step="0.25" required value={pretrip.awake_h} onChange={(event) => update('awake_h', Number(event.currentTarget.value))} />
            </label>
            <label>
              Tiredness (1–5)
              <select required value={pretrip.tiredness} onChange={(event) => update('tiredness', Number(event.currentTarget.value))}>
                {[1, 2, 3, 4, 5].map((value) => <option key={value} value={value}>{value}</option>)}
              </select>
            </label>
            <label>
              Engagement preference
              <select value={pretrip.engagement} onChange={(event) => update('engagement', event.currentTarget.value as Pretrip['engagement'])}>
                <option value="trivia">Trivia</option>
                <option value="fact">Fun facts</option>
                <option value="quiet">Quiet</option>
              </select>
            </label>
            <label>
              Language
              <select value={pretrip.lang} onChange={(event) => update('lang', event.currentTarget.value as Pretrip['lang'])}>
                <option value="en">English</option>
                <option value="ro">Romanian</option>
              </select>
            </label>
            <label>
              Resting heart rate (BPM)
              <input type="number" min="1" step="1" required value={pretrip.resting_hr} onChange={(event) => update('resting_hr', Number(event.currentTarget.value))} />
            </label>
            <label>
              Emergency contact (E.164)
              <input type="tel" required pattern="\+[1-9][0-9]{1,14}" placeholder="+407XXXXXXXX" value={pretrip.emergency_contact} onChange={(event) => update('emergency_contact', event.currentTarget.value)} />
            </label>
            <label className="pretrip-caffeine">
              <input type="checkbox" checked={pretrip.caffeine} onChange={(event) => update('caffeine', event.currentTarget.checked)} />
              Caffeine in the last 2 hours
            </label>
          </fieldset>
          <div className="pretrip-actions">
            <button type="submit" disabled={isMoving}>Start trip</button>
            {isMoving && <span>Setup is disabled above 5 km/h.</span>}
          </div>
        </form>
      )}
      {message && <p className={`pretrip-message pretrip-message--${message.kind}`} role={message.kind === 'error' ? 'alert' : 'status'}>{message.text}</p>}
    </section>
  )
}
