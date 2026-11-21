import { useEffect, useState } from 'react'
import { bus } from './bus'
import type { TopicEnvelope } from './types'

// Keep aligned with config/thresholds.yaml: ui.crash_countdown_s and core.crash.countdown_s.
const CRASH_COUNTDOWN_SECONDS = 20

type CrashNotice = {
  alertId: string
  phase: 'countdown' | 'emergency'
  startedAt: number
  contact?: string
  lat?: number
  lon?: number
}

let notice: CrashNotice | null = null
let lastAlert: TopicEnvelope<'alert.state'> | undefined
let lastTrip: TopicEnvelope<'control.trip'> | undefined
let emergencyContact: string | undefined
const subscribers = new Set<() => void>()

function latest<K extends 'alert.state' | 'control.trip' | 'vehicle.motion'>(topic: K) {
  return bus.getLatest(topic) as TopicEnvelope<K> | undefined
}

function rememberTripContact(message: TopicEnvelope<'control.trip'> | undefined) {
  if (!message || message === lastTrip) return
  lastTrip = message
  if (message.data.action === 'start' && message.data.pretrip?.emergency_contact) {
    emergencyContact = message.data.pretrip.emergency_contact
  }
}

function publishNotice(next: CrashNotice | null) {
  notice = next
  subscribers.forEach((subscriber) => subscriber())
}

function processBus() {
  rememberTripContact(latest('control.trip'))

  const message = latest('alert.state')
  if (!message || message === lastAlert) return
  lastAlert = message
  const alert = message.data
  if (alert.type !== 'crash_candidate' || alert.tier !== 'T3') return

  if (alert.status === 'closed') {
    if (notice?.alertId === alert.alert_id) publishNotice(null)
    return
  }

  if (alert.status === 'escalated') {
    if (notice?.alertId === alert.alert_id && notice.phase === 'emergency') return
    const motion = latest('vehicle.motion')?.data
    publishNotice({
      alertId: alert.alert_id,
      phase: 'emergency',
      startedAt: notice?.alertId === alert.alert_id ? notice.startedAt : message.ts,
      contact: emergencyContact,
      lat: motion?.lat,
      lon: motion?.lon,
    })
    return
  }

  if (alert.status !== 'speaking' && alert.status !== 'listening') return
  if (notice?.alertId === alert.alert_id) return
  const voiceSay = bus.getLatest('voice.say') as TopicEnvelope<'voice.say'> | undefined
  const startedAt = voiceSay?.data.alert_id === alert.alert_id
    ? Math.min(message.ts, voiceSay.ts)
    : message.ts
  publishNotice({ alertId: alert.alert_id, phase: 'countdown', startedAt })
}

rememberTripContact(latest('control.trip'))
processBus()
bus.subscribe(processBus)

export function useCrashNotice() {
  const [, setRevision] = useState(0)
  const [now, setNow] = useState(() => Date.now() / 1000)

  useEffect(() => {
    const update = () => setRevision((revision) => revision + 1)
    subscribers.add(update)
    return () => { subscribers.delete(update) }
  }, [])

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 250)
    return () => window.clearInterval(timer)
  }, [])

  const remaining = notice?.phase === 'countdown'
    ? Math.max(0, Math.ceil(notice.startedAt + CRASH_COUNTDOWN_SECONDS - now))
    : undefined
  return { notice, remaining }
}

export function CrashSafetyNotice({ compact = false }: { compact?: boolean }) {
  const { notice: current, remaining } = useCrashNotice()
  if (!current) return null

  if (current.phase === 'countdown') {
    return (
      <aside className={`crash-safety-notice crash-safety-notice--countdown${compact ? ' crash-safety-notice--compact' : ''}`} role="alert" aria-live="assertive" aria-label="Crash response countdown">
        <strong className="crash-countdown-number">{remaining}<small>s</small></strong>
        <span className="crash-countdown-copy">Are you okay? Calling for help in {remaining}…</span>
      </aside>
    )
  }

  const contact = current.contact || 'emergency contact unavailable'
  const location = typeof current.lat === 'number' && typeof current.lon === 'number'
    ? `${current.lat.toFixed(5)}, ${current.lon.toFixed(5)}`
    : 'location unavailable'

  return (
    <aside className={`crash-safety-notice crash-safety-notice--emergency${compact ? ' crash-safety-notice--compact' : ''}`} role="alert" aria-live="assertive">
      <span className="crash-emergency-label">EMERGENCY</span>
      <strong>Emergency: contacting {contact} · location {location}</strong>
    </aside>
  )
}
