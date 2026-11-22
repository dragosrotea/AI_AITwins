import { useEffect, useState } from 'react'
import { bus, type BusStatus } from './bus'

const TRIPS = ['normal', 'drowsy', 'aggressive'] as const
const MAX_SEGMENT_SECONDS = 15 * 60

type Trip = (typeof TRIPS)[number]

function formatSegment(seconds: number) {
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  return `${minutes}:${remainder.toString().padStart(2, '0')}`
}

export default function ReplayControls() {
  const [busStatus, setBusStatus] = useState<BusStatus>(bus.getStatus())
  const [trip, setTrip] = useState<Trip>('normal')
  const [segment, setSegment] = useState(0)
  const [crashRequested, setCrashRequested] = useState(false)
  const [crashWasMoving, setCrashWasMoving] = useState(false)
  const [crashStopped, setCrashStopped] = useState(false)
  const [feedback, setFeedback] = useState('')
  const connected = busStatus === 'connected'

  useEffect(() => bus.subscribeStatus(setBusStatus), [])

  useEffect(() => {
    if (!crashRequested) return
    return bus.subscribe(() => {
      const motion = bus.getLatest('vehicle.motion')
      const speed = motion?.data.speed_kmh
      if (typeof speed === 'number' && speed > 0) {
        setCrashWasMoving(true)
      } else if (typeof speed === 'number' && speed <= 0 && crashWasMoving) {
        setCrashStopped(true)
      }
    })
  }, [crashRequested, crashWasMoving])

  function sendReplay(data: { action: 'play' | 'pause' | 'seek'; trip?: string; segment?: number }, label: string) {
    if (!bus.publish('control.replay', data)) {
      setFeedback('Broker disconnected. Replay command was not sent.')
      return
    }
    setCrashRequested(false)
    setCrashWasMoving(false)
    setCrashStopped(false)
    setFeedback(label)
  }

  function sendInjector(event: 'harsh_brake' | 'crash' | 'zoning_out') {
    if (!bus.publish('control.inject', { event })) {
      setFeedback('Broker disconnected. Demo event was not sent.')
      return
    }
    if (event === 'crash') {
      setCrashRequested(true)
      setCrashWasMoving(false)
      setCrashStopped(false)
      setFeedback('Crash simulation sent. Waiting for the replay to stop…')
    } else if (event === 'zoning_out') {
      setFeedback('Zone-out injected. Trivia needs the pre-trip opt-in and a calm road.')
    } else {
      setFeedback('Harsh brake injected: a −0.6 g dip on the chart; it counts as driving in the rage score for 60 s.')
    }
  }

  const waitingForCrashStop = crashRequested && !crashStopped
  const replayDisabled = !connected || crashRequested

  return (
    <section className="panel-placeholder copilot-context-panel replay-controls-panel" aria-label="Replay and event controls">
      <div className="replay-controls-heading">
        <div>
          <p className="panel-label">REPLAY & DEMO CONTROLS</p>
          <p className="replay-controls-detail">Replay trips and inject test events</p>
        </div>
        <span className={`replay-connection replay-connection--${busStatus}`}>
          <span className="connection-dot" />{connected ? 'Connected' : 'Offline'}
        </span>
      </div>

      <div className="replay-trip-controls">
        <label className="replay-field-label" htmlFor="replay-trip">Replay trip</label>
        <select id="replay-trip" value={trip} onChange={(event) => setTrip(event.target.value as Trip)} disabled={replayDisabled}>
          {TRIPS.map((name) => <option key={name} value={name}>{name[0].toUpperCase()}{name.slice(1)}</option>)}
        </select>
        <div className="replay-buttons">
          <button type="button" onClick={() => sendReplay({ action: 'play', trip }, `Playing ${trip} from the start.`)} disabled={replayDisabled}>Play trip</button>
          <button type="button" className="replay-button-secondary" onClick={() => sendReplay({ action: 'pause' }, 'Replay paused.')} disabled={replayDisabled}>Pause</button>
        </div>
        <p className="replay-hint">Changing trips and pressing Play starts at 0:00. Use Seek to start at a segment.</p>
      </div>

      <div className="replay-seek-controls">
        <label className="replay-field-label" htmlFor="replay-segment">
          <span>Seek segment</span><strong>{formatSegment(segment)}</strong>
        </label>
        <input
          id="replay-segment"
          type="range"
          min="0"
          max={MAX_SEGMENT_SECONDS}
          step="1"
          value={segment}
          onChange={(event) => setSegment(Number(event.target.value))}
          disabled={replayDisabled}
        />
        <div className="replay-seek-footer">
          <span>0:00</span><span>15:00</span>
          <button type="button" onClick={() => sendReplay({ action: 'seek', trip, segment }, `Seeking ${trip} to ${formatSegment(segment)}.`)} disabled={replayDisabled}>
            Seek
          </button>
        </div>
      </div>

      <div className="inject-controls">
        <p className="replay-field-label">Demo injector</p>
        <div className="inject-buttons">
          <button type="button" className="inject-button-secondary" onClick={() => sendInjector('harsh_brake')} disabled={!connected || crashRequested}>
            Harsh brake
          </button>
          <button type="button" className="inject-button-danger" onClick={() => sendInjector('crash')} disabled={!connected || crashRequested}>
            Simulate crash
          </button>
          <button type="button" className="inject-button-secondary" onClick={() => sendInjector('zoning_out')} disabled={!connected || crashRequested} title="Raise a zoning-out event: core asks a trivia question">
            Zone out (trivia)
          </button>
        </div>
      </div>

      {crashRequested && (
        <div className={`crash-resume-state${crashStopped ? ' crash-resume-state--stopped' : ''}`} role="status" aria-live="polite">
          <span>{crashStopped ? 'Replay stopped after crash.' : 'Crash sent; waiting for the car to stop.'}</span>
          {crashStopped && (
            <button type="button" onClick={() => sendReplay({ action: 'play' }, 'Resume command sent.')} disabled={!connected}>
              Resume replay
            </button>
          )}
        </div>
      )}

      {feedback && !crashRequested && <p className="replay-feedback" role="status">{feedback}</p>}
      {!connected && <p className="replay-feedback replay-feedback--offline">Connect to the broker to use replay controls.</p>}
      {waitingForCrashStop && <span className="sr-only">Replay controls are paused until the crash sequence stops the vehicle.</span>}
    </section>
  )
}
