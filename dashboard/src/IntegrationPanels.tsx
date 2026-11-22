import { useEffect, useState, useRef } from 'react'
import { bus, type SeriesPoint } from './bus'
import type { AudioFeatures, BioHR, CheckinSchedule, ControlTrip, DriverState, TopicEnvelope } from './types'

const ATTENTION = {
  weights: { drowsiness: 0.35, distraction: 0.3, rage: 0.2, zoning: 0.15 },
  perclosFullRisk: 0.15,
  yawFullRiskDeg: 30,
  pitchDownFullRiskDeg: 20,
  gazeDispersionFullRiskDeg: 30,
  yawnsFullRisk: 3,
  angerCueThreshold: 0.5,
  musicRiseDelta: 10,
  musicLowVolume: 25,
  zoningCueThreshold: 0.6,
  cueWindowS: 30,
} as const

function useBusRefresh(periodMs = 150) {
  const [, setRevision] = useState(0)

  useEffect(() => {
    let timer: number | undefined
    const unsubscribe = bus.subscribe(() => {
      if (timer !== undefined) return
      timer = window.setTimeout(() => {
        timer = undefined
        setRevision((revision) => revision + 1)
      }, periodMs)
    })
    return () => {
      unsubscribe()
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [periodMs])
}

function latest<K extends 'driver.state' | 'audio.features' | 'music.volume' | 'bio.hr' | 'control.trip' | 'checkin.schedule' | 'context.road'>(topic: K) {
  return bus.getLatest(topic) as TopicEnvelope<K> | undefined
}

function clamp01(value: number) {
  return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0))
}

function attentionParts(driver: DriverState, audio?: AudioFeatures) {
  const drowsiness = Math.max(
    driver.eye_closed_p,
    driver.eyes_closed_ms >= 1000 ? 1 : 0,
    driver.perclos_60s / ATTENTION.perclosFullRisk,
    driver.yawns_10m / ATTENTION.yawnsFullRisk,
  )
  const distraction = Math.max(
    driver.face_present ? 0 : 1,
    Math.abs(driver.head.yaw) / ATTENTION.yawFullRiskDeg,
    Math.max(0, -driver.head.pitch) / ATTENTION.pitchDownFullRiskDeg,
    driver.gaze_dispersion_30s / ATTENTION.gazeDispersionFullRiskDeg,
    driver.phone_in_hand ? 1 : 0,
  )
  const rage = Math.max(
    driver.anger_p,
    audio?.anger_voice_p ?? 0,
    audio?.shouting ? 0.75 : 0,
    Math.min(1, (audio?.swear_count_60s ?? 0) / 3),
  )
  const zoning = driver.zoning_visual_p
  const risks = { drowsiness, distraction, rage, zoning }
  const weightedRisk = Object.entries(ATTENTION.weights).reduce(
    (sum, [key, weight]) => sum + clamp01(risks[key as keyof typeof risks]) * weight,
    0,
  )
  return { risks, score: Math.round(100 * (1 - weightedRisk)) }
}

function RiskMeter({ label, risk }: { label: string; risk: number }) {
  const percent = Math.round(clamp01(risk) * 100)
  return (
    <div className="attention-risk">
      <div><span>{label}</span><strong>{percent}</strong></div>
      <div className="attention-risk-track" aria-hidden="true"><span style={{ width: `${percent}%` }} /></div>
    </div>
  )
}

export function AttentionPanel() {
  useBusRefresh(150)
  const now = Date.now() / 1000
  const driverMessage = latest('driver.state')
  const driver = driverMessage?.data
  const audioMessage = latest('audio.features')
  const audio = audioMessage && now - audioMessage.ts <= 2 ? audioMessage.data : undefined
  const currentVolume = latest('music.volume')?.data.volume
  const angerSeries: SeriesPoint[] = bus.getSeries('driver.state.anger_p')
  const volumeSeries: SeriesPoint[] = bus.getSeries('music.volume.volume')
  const windowStart = now - ATTENTION.cueWindowS
  const angerHigh = angerSeries.some((point) => point.ts >= windowStart && point.value > ATTENTION.angerCueThreshold)
  const recentVolume = volumeSeries.filter((point) => point.ts >= windowStart)
  const volumeRise = recentVolume.length >= 2
    && recentVolume[recentVolume.length - 1].value - recentVolume[0].value >= ATTENTION.musicRiseDelta
  const freshDriver = Boolean(driver && driverMessage && now - driverMessage.ts <= 2)
  const zoningHigh = freshDriver && (driver?.zoning_visual_p ?? 0) >= ATTENTION.zoningCueThreshold
  const stressCue = volumeRise && angerHigh
  const lowEngagementCue = currentVolume !== undefined
    && currentVolume <= ATTENTION.musicLowVolume
    && zoningHigh
  const parts = driver && freshDriver ? attentionParts(driver, audio) : undefined

  return (
    <section className="panel-placeholder copilot-attention-panel" aria-label="Attention score and cues">
      <div className="attention-heading">
        <div>
          <p className="panel-label">ATTENTION</p>
          <p className="telemetry-detail">Higher is better · context only, not an alert trigger</p>
        </div>
        <strong className="attention-score">{parts ? parts.score : '—'}<small>/100</small></strong>
      </div>
      {parts ? (
        <div className="attention-risks">
          <RiskMeter label="Drowsiness" risk={parts.risks.drowsiness} />
          <RiskMeter label="Distraction" risk={parts.risks.distraction} />
          <RiskMeter label="Rage" risk={parts.risks.rage} />
          <RiskMeter label="Zoning out" risk={parts.risks.zoning} />
        </div>
      ) : (
        <p className="integration-empty">Waiting for fresh driver.state data.</p>
      )}
      <div className="attention-cues" aria-live="polite">
        <span className={`attention-cue${stressCue ? ' attention-cue--active attention-cue--stress' : ''}`}>Stress{stressCue ? ' · detected' : ''}</span>
        <span className={`attention-cue${lowEngagementCue ? ' attention-cue--active attention-cue--engagement' : ''}`}>Low engagement{lowEngagementCue ? ' · detected' : ''}</span>
      </div>
    </section>
  )
}

function countdown(timestamp: number | null | undefined, now: number) {
  if (timestamp === undefined || timestamp === null) return 'Not scheduled'
  const seconds = Math.ceil(timestamp - now)
  if (seconds <= 0) return 'Due now'
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const remainder = seconds % 60
  return hours
    ? `${hours}:${minutes.toString().padStart(2, '0')}:${remainder.toString().padStart(2, '0')}`
    : `${minutes}:${remainder.toString().padStart(2, '0')}`
}

export function CheckinSchedulePanel() {
  useBusRefresh(1000)
  const now = Date.now() / 1000
  const schedule = latest('checkin.schedule')?.data as CheckinSchedule | undefined

  return (
    <section className="panel-placeholder copilot-checkin-panel" aria-label="Check-in schedule">
      <p className="panel-label">CHECK-IN SCHEDULE</p>
      {schedule ? (
        <div className="checkin-schedule-grid">
          <div><span>INTERVAL</span><strong>{schedule.interval_min} min</strong></div>
          <div><span>NEXT CHECK-IN</span><strong>{countdown(schedule.next_checkin_ts, now)}</strong></div>
          <div><span>NEXT BREAK</span><strong>{countdown(schedule.next_break_ts, now)}</strong></div>
        </div>
      ) : <p className="integration-empty">Waiting for checkin.schedule.</p>}
    </section>
  )
}

export function TripSummaryPanel() {
  useBusRefresh(500)
  const summary = bus.getLatest('trip.summary') as TopicEnvelope<'trip.summary'> | undefined
  if (!summary) return null

  return (
    <section className="panel-placeholder copilot-summary-panel" aria-label="Trip summary" aria-live="polite">
      <div className="summary-heading">
        <div>
          <p className="panel-label">TRIP SUMMARY</p>
          <p className="telemetry-detail">Generated at trip end</p>
        </div>
        <time dateTime={new Date(summary.ts * 1000).toISOString()}>
          {new Date(summary.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </time>
      </div>
      <p className="summary-text">{summary.data.text}</p>
    </section>
  )
}

export function RoadContextPanel() {
  useBusRefresh(500)
  const now = Date.now() / 1000
  const message = latest('context.road')
  const road = message?.data
  const fresh = Boolean(message && now - message.ts <= 3)
  const distance = road?.dist_m == null
    ? '—'
    : road.dist_m >= 1000
      ? `${(road.dist_m / 1000).toFixed(1)} km`
      : `${Math.round(road.dist_m)} m`

  return (
    <section className="panel-placeholder copilot-road-panel" aria-label="Road context">
      <div className="road-context-heading">
        <div>
          <p className="panel-label">ROAD CONTEXT</p>
          <p className="telemetry-detail">Demand and next manoeuvre</p>
        </div>
        <span className={`road-context-live${fresh ? ' road-context-live--active' : ''}`} role="status">
          <i aria-hidden="true" />{fresh ? 'Live' : 'Waiting for maps'}
        </span>
      </div>
      {fresh && road ? (
        <>
          <div className={`road-load road-load--${road.load.toLowerCase()}`}>
            <span>ROAD LOAD</span>
            <strong>{road.load}</strong>
          </div>
          <div className="road-context-details">
            <div><span>TRAFFIC</span><strong>{road.traffic || 'No traffic detail'}</strong></div>
            <div><span>NEXT MANOEUVRE</span><strong>{road.next_maneuver || 'No manoeuvre reported'}</strong></div>
            <div><span>DISTANCE</span><strong>{distance}</strong></div>
          </div>
        </>
      ) : (
        <p className="integration-empty">Start the maps simulator to see road load.</p>
      )}
    </section>
  )
}

const HR_CHART = { width: 560, height: 132, left: 8, right: 552, top: 10, bottom: 112 }

function hrY(value: number, min: number, max: number) {
  return HR_CHART.top + (1 - (value - min) / (max - min)) * (HR_CHART.bottom - HR_CHART.top)
}

export function HeartRatePanel() {
  useBusRefresh(500)
  const now = Date.now() / 1000
  const hrMessage = latest('bio.hr')
  const hr = hrMessage?.data as BioHR | undefined
  const points: SeriesPoint[] = bus.getSeries('bio.hr.bpm').filter((point) => point.ts >= now - 60)
  const [restingHr, setRestingHr] = useState<number | undefined>()
  const seenTrip = useRef<TopicEnvelope<'control.trip'> | undefined>(undefined)

  useEffect(() => {
    function readTrip() {
      const message = bus.getLatest('control.trip') as TopicEnvelope<'control.trip'> | undefined
      if (!message || message === seenTrip.current) return
      seenTrip.current = message
      const data = message.data as ControlTrip
      if (data.action === 'start' && data.pretrip) setRestingHr(data.pretrip.resting_hr)
    }
    readTrip()
    return bus.subscribe(readTrip)
  }, [])

  const values = points.map((point) => point.value).concat(restingHr ?? [])
  const min = values.length ? Math.max(30, Math.floor((Math.min(...values) - 10) / 10) * 10) : 40
  const max = values.length ? Math.min(220, Math.ceil((Math.max(...values) + 10) / 10) * 10) : 180
  const range = Math.max(20, max - min)
  const line = points.map((point, index) => {
    const x = HR_CHART.left + Math.max(0, Math.min(1, (point.ts - (now - 60)) / 60)) * (HR_CHART.right - HR_CHART.left)
    const y = hrY(point.value, min, min + range)
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
  const fresh = hrMessage && now - hrMessage.ts <= 5
  const restingVisible = restingHr !== undefined && restingHr >= min && restingHr <= min + range

  return (
    <section className="panel-placeholder copilot-hr-panel" aria-label="Live heart rate">
      <div className="hr-panel-heading">
        <div>
          <p className="panel-label">HEART RATE</p>
          <p className="telemetry-detail">Last 60 seconds · resting line from trip setup</p>
        </div>
        <div className="hr-current">
          <strong>{fresh ? Math.round(hr!.bpm) : '—'}</strong><span>BPM · {fresh ? hr!.source : 'waiting'}</span>
        </div>
      </div>
      {points.length ? (
        <div className="hr-chart-wrap">
          <svg className="hr-chart" viewBox={`0 0 ${HR_CHART.width} ${HR_CHART.height}`} role="img" aria-label="Heart rate over the last 60 seconds">
            {[min, min + range / 2, min + range].map((tick) => {
              const y = hrY(tick, min, min + range)
              return <g key={tick}><line className="hr-gridline" x1={HR_CHART.left} x2={HR_CHART.right} y1={y} y2={y} /><text className="hr-axis-label" x={HR_CHART.left} y={y - 3}>{Math.round(tick)}</text></g>
            })}
            {restingVisible && <><line className="hr-resting-line" x1={HR_CHART.left} x2={HR_CHART.right} y1={hrY(restingHr!, min, min + range)} y2={hrY(restingHr!, min, min + range)} /><text className="hr-resting-label" x={HR_CHART.right} y={hrY(restingHr!, min, min + range) - 4} textAnchor="end">REST {Math.round(restingHr!)} BPM</text></>}
            <path className="hr-line" d={line} />
            <text className="hr-axis-label" x={HR_CHART.left} y="128">−60 s</text>
            <text className="hr-axis-label" x={HR_CHART.right} y="128" textAnchor="end">now</text>
          </svg>
        </div>
      ) : (
        <p className="integration-empty">Waiting for bio.hr data.</p>
      )}
      {restingHr !== undefined && !restingVisible && <p className="hr-resting-note">Resting HR: {Math.round(restingHr)} BPM</p>}
    </section>
  )
}
