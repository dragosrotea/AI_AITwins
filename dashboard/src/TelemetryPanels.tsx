import { useEffect, useState } from 'react'
import { bus, type SeriesPoint } from './bus'
import type { Topic, TopicEnvelope } from './types'

const CHART_WIDTH = 640
const CHART_HEIGHT = 210
const PLOT = { left: 48, top: 14, width: 544, height: 150 }

function useBusUpdates() {
  const [, setRevision] = useState(0)

  useEffect(() => {
    let refreshTimer: number | undefined
    const unsubscribe = bus.subscribe(() => {
      if (refreshTimer !== undefined) return
      refreshTimer = window.setTimeout(() => {
        refreshTimer = undefined
        setRevision((revision) => revision + 1)
      }, 100)
    })
    return () => {
      unsubscribe()
      if (refreshTimer !== undefined) window.clearTimeout(refreshTimer)
    }
  }, [])
}

function latest<K extends Topic>(topic: K): TopicEnvelope<K> | undefined {
  return bus.getLatest(topic) as TopicEnvelope<K> | undefined
}

function downsample(points: SeriesPoint[], maxPoints = 240) {
  if (points.length <= maxPoints) return points
  const stride = (points.length - 1) / (maxPoints - 1)
  return Array.from({ length: maxPoints }, (_, index) => points[Math.round(index * stride)])
}

function pathFor(
  points: SeriesPoint[],
  now: number,
  minValue: number,
  maxValue: number,
  left = PLOT.left,
  top = PLOT.top,
  width = PLOT.width,
  height = PLOT.height,
) {
  const recent = downsample(points.filter((point) => point.ts >= now - 60))
  if (!recent.length) return ''
  return recent.map((point, index) => {
    const x = left + Math.max(0, Math.min(1, (point.ts - (now - 60)) / 60)) * width
    const normalized = (point.value - minValue) / (maxValue - minValue)
    const y = top + (1 - Math.max(0, Math.min(1, normalized))) * height
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function yFor(value: number, minValue: number, maxValue: number) {
  return PLOT.top + (1 - (value - minValue) / (maxValue - minValue)) * PLOT.height
}

function PanelHeading({ title, detail, badge }: { title: string; detail?: string; badge?: string }) {
  return (
    <div className="telemetry-heading">
      <div>
        <p className="panel-label">{title}</p>
        {detail && <p className="telemetry-detail">{detail}</p>}
      </div>
      {badge && <span className="telemetry-badge">{badge}</span>}
    </div>
  )
}

function CameraPreview() {
  useBusUpdates()
  const frame = latest('camera.frame')
  const [failedAt, setFailedAt] = useState<number | undefined>()
  const frameTs = frame?.ts
  const jpeg = frame?.data.jpeg_b64
  const source = jpeg
    ? jpeg.startsWith('data:') ? jpeg : `data:image/jpeg;base64,${jpeg}`
    : undefined
  const failed = frameTs !== undefined && failedAt === frameTs

  return (
    <section className="camera-preview" aria-label="Camera preview">
      <div className="camera-preview-topline">
        <span className="camera-live-dot" />
        <span>CV CAMERA</span>
        <span className="camera-overlay-note">overlay as sent by CV</span>
      </div>
      {source && !failed ? (
        <img
          className="camera-frame"
          src={source}
          alt="Latest camera frame with the CV overlay, if provided"
          onError={() => setFailedAt(frameTs)}
        />
      ) : (
        <div className="camera-empty" role="status">
          <span>{failed ? 'Camera frame could not be decoded' : 'Waiting for camera.frame'}</span>
          <small>Frames stay in memory and are not saved.</small>
        </div>
      )}
      {frame && <p className="camera-frame-time">Frame age {Math.max(0, Date.now() / 1000 - frame.ts).toFixed(1)} s</p>}
    </section>
  )
}

function MetricGauge({
  label,
  value,
  display,
  fill,
  hint,
}: {
  label: string
  value?: number
  display: string
  fill?: number
  hint?: string
}) {
  const ratio = Math.max(0, Math.min(1, fill ?? 0))
  return (
    <div className="metric-gauge">
      <div className="metric-gauge-label">
        <span>{label}</span>
        <strong>{display}</strong>
      </div>
      {value === undefined ? (
        <div className="metric-gauge-track metric-gauge-track--empty" />
      ) : (
        <div
          className={`metric-gauge-track${ratio >= 0.7 ? ' metric-gauge-track--high' : ''}`}
          role="meter"
          aria-label={label}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-valuenow={Math.round(ratio * 100)}
        >
          <span style={{ width: `${ratio * 100}%` }} />
        </div>
      )}
      {hint && <small>{hint}</small>}
    </div>
  )
}

function DriverMetrics() {
  useBusUpdates()
  const state = latest('driver.state')?.data
  const percent = (value: number | undefined) => value === undefined ? undefined : value * 100
  const displayPercent = (value: number | undefined) => value === undefined ? '—' : `${Math.round(value * 100)}%`

  return (
    <section className="driver-metrics" aria-label="Live driver measurements">
      <p className="panel-label">DRIVER GAUGES</p>
      <div className="driver-gauge-grid">
        <MetricGauge label="Eye closure" value={state?.eye_closed_p} display={displayPercent(state?.eye_closed_p)} fill={percent(state?.eye_closed_p)} />
        <MetricGauge label="PERCLOS · 60 s" value={state?.perclos_60s} display={displayPercent(state?.perclos_60s)} fill={percent(state?.perclos_60s)} />
        <MetricGauge label="Head yaw" value={state?.head.yaw} display={state ? `${state.head.yaw.toFixed(0)}°` : '—'} fill={state ? Math.abs(state.head.yaw) / 60 : undefined} hint="±60° scale" />
        <MetricGauge label="Head pitch" value={state?.head.pitch} display={state ? `${state.head.pitch.toFixed(0)}°` : '—'} fill={state ? Math.abs(state.head.pitch) / 45 : undefined} hint="±45° scale" />
        <MetricGauge label="Anger cue" value={state?.anger_p} display={displayPercent(state?.anger_p)} fill={percent(state?.anger_p)} />
        <MetricGauge label="Yawn cue" value={state?.yawn_p} display={displayPercent(state?.yawn_p)} fill={percent(state?.yawn_p)} hint={state ? `${state.yawns_10m} in 10 min` : 'Count: —'} />
      </div>
    </section>
  )
}

export function DriverCameraPanel() {
  return (
    <section className="panel-placeholder copilot-driver-panel telemetry-driver-panel">
      <div className="driver-camera-layout">
        <CameraPreview />
        <DriverMetrics />
      </div>
    </section>
  )
}

export function AccelerationChart() {
  useBusUpdates()
  const now = Date.now() / 1000
  const motion = latest('vehicle.motion')
  const longSeries = bus.getSeries('vehicle.motion.accel_long_g')
  const latSeries = bus.getSeries('vehicle.motion.accel_lat_g')
  const hasData = longSeries.length > 0 || latSeries.length > 0
  const sourceLabel = motion?.data.source === 'replay' ? 'REPLAY' : motion?.data.source === 'sim' ? 'SIM' : undefined

  return (
    <section className="panel-placeholder copilot-motion-panel telemetry-panel" aria-label="Acceleration chart">
      <PanelHeading title="ACCELERATION" detail="Longitudinal and lateral · last 60 seconds" badge={sourceLabel} />
      {hasData ? (
        <>
          <div className="chart-wrap">
            <svg className="telemetry-chart" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label="Longitudinal and lateral acceleration over the last 60 seconds, with plus and minus 0.4 g markers">
              {[1, 0.4, 0, -0.4, -1].map((value) => {
                const y = yFor(value, -1, 1)
                const threshold = Math.abs(value) === 0.4
                return (
                  <g key={value}>
                    <line className={threshold ? 'chart-threshold' : 'chart-gridline'} x1={PLOT.left} x2={PLOT.left + PLOT.width} y1={y} y2={y} />
                    <text className="chart-axis-label" x={PLOT.left - 8} y={y + 4} textAnchor="end">{value > 0 ? '+' : ''}{value.toFixed(1)}</text>
                  </g>
                )
              })}
              <path className="chart-line chart-line--long" d={pathFor(longSeries, now, -1, 1)} />
              <path className="chart-line chart-line--lat" d={pathFor(latSeries, now, -1, 1)} />
              <text className="chart-axis-label" x={PLOT.left} y="190">−60 s</text>
              <text className="chart-axis-label" x={PLOT.left + PLOT.width / 2} y="190" textAnchor="middle">−30 s</text>
              <text className="chart-axis-label" x={PLOT.left + PLOT.width} y="190" textAnchor="end">now</text>
            </svg>
          </div>
          <div className="chart-legend">
            <span><i className="legend-swatch legend-swatch--long" />Longitudinal {motion ? `${motion.data.accel_long_g.toFixed(2)} g` : ''}</span>
            <span><i className="legend-swatch legend-swatch--lat" />Lateral {motion ? `${motion.data.accel_lat_g.toFixed(2)} g` : ''}</span>
            <span className="threshold-key">±0.4 g markers</span>
          </div>
        </>
      ) : <p className="telemetry-empty">Waiting for vehicle.motion data</p>}
    </section>
  )
}

export function CabinChart() {
  useBusUpdates()
  const now = Date.now() / 1000
  const volume = bus.getSeries('music.volume.volume')
  const loudness = bus.getSeries('audio.features.loudness_db')
  const latestVolume = latest('music.volume')?.data
  const latestAudio = latest('audio.features')?.data
  const hasData = volume.length > 0 || loudness.length > 0
  const volumePath = pathFor(volume, now, 0, 100)
  const loudnessPath = pathFor(loudness, now, -70, 0)

  return (
    <section className="panel-placeholder copilot-signals-panel telemetry-panel" aria-label="Music volume and cabin loudness chart">
      <PanelHeading title="CABIN SOUND" detail="Music volume and cabin loudness · last 60 seconds" />
      {hasData ? (
        <>
          <div className="chart-wrap">
            <svg className="telemetry-chart" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label="Music volume from zero to one hundred and cabin loudness in decibels, over the last 60 seconds">
              {[0, 50, 100].map((value) => {
                const y = yFor(value, 0, 100)
                return (
                  <g key={value}>
                    <line className="chart-gridline" x1={PLOT.left} x2={PLOT.left + PLOT.width} y1={y} y2={y} />
                    <text className="chart-axis-label" x={PLOT.left - 8} y={y + 4} textAnchor="end">{value}</text>
                  </g>
                )
              })}
              <path className="chart-line chart-line--volume" d={volumePath} />
              <path className="chart-line chart-line--loudness" d={loudnessPath} />
              <text className="chart-axis-label" x={PLOT.left + PLOT.width + 8} y={PLOT.top + 4}>0 dB</text>
              <text className="chart-axis-label" x={PLOT.left + PLOT.width + 8} y={PLOT.top + PLOT.height / 2 + 4}>−35</text>
              <text className="chart-axis-label" x={PLOT.left + PLOT.width + 8} y={PLOT.top + PLOT.height + 4}>−70</text>
              <text className="chart-axis-label" x={PLOT.left} y="190">−60 s</text>
              <text className="chart-axis-label" x={PLOT.left + PLOT.width / 2} y="190" textAnchor="middle">−30 s</text>
              <text className="chart-axis-label" x={PLOT.left + PLOT.width} y="190" textAnchor="end">now</text>
            </svg>
          </div>
          <div className="chart-legend">
            <span><i className="legend-swatch legend-swatch--volume" />Music {latestVolume ? `${latestVolume.volume}%` : ''}</span>
            <span><i className="legend-swatch legend-swatch--loudness" />Cabin {latestAudio ? `${latestAudio.loudness_db.toFixed(1)} dB` : ''}</span>
          </div>
        </>
      ) : <p className="telemetry-empty">Waiting for music.volume and audio.features data</p>}
    </section>
  )
}

export function DebugCorner() {
  const [now, setNow] = useState(() => Date.now())
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 250)
    return () => window.clearInterval(timer)
  }, [])

  const driver = latest('driver.state')?.data
  const newestTs = [...bus.getAllLatest().values()].reduce<number | undefined>(
    (current, message) => current === undefined || message.ts > current ? message.ts : current,
    undefined,
  )
  const delayMs = newestTs === undefined ? undefined : Math.max(0, now / 1000 - newestTs) * 1000

  return (
    <aside className="debug-corner" aria-label="Live debug metrics">
      <span className="debug-corner-title">DEBUG</span>
      <span>Camera FPS <strong>{driver ? driver.fps.toFixed(1) : '—'}</strong></span>
      <span>Message delay <strong>{delayMs === undefined ? '—' : `${Math.round(delayMs)} ms`}</strong></span>
    </aside>
  )
}
