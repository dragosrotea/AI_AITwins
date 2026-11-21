import { useEffect, useRef, useState } from 'react'
import { bus } from './bus'
import type { AlertState, DetectorEvent, TopicEnvelope } from './types'

type TimelineEntry = {
  key: string
  ts: number
  tier: 'T1' | 'T2' | 'T3'
  title: string
  detail: string
  kind: 'detector' | 'closed-alert'
}

const MAX_TIMELINE_ENTRIES = 100

function label(value: string) {
  return value.replaceAll('_', ' ')
}

export default function EventTimeline() {
  const [entries, setEntries] = useState<TimelineEntry[]>([])
  const seenDetector = useRef<TopicEnvelope<'detector.event'> | undefined>(undefined)
  const seenAlert = useRef<TopicEnvelope<'alert.state'> | undefined>(undefined)

  useEffect(() => {
    function add(entry: TimelineEntry) {
      setEntries((current) => {
        if (current.some((item) => item.key === entry.key)) return current
        return [entry, ...current].sort((a, b) => b.ts - a.ts).slice(0, MAX_TIMELINE_ENTRIES)
      })
    }

    function readLatest() {
      const detector = bus.getLatest('detector.event') as TopicEnvelope<'detector.event'> | undefined
      if (detector && detector !== seenDetector.current) {
        seenDetector.current = detector
        const data: DetectorEvent = detector.data
        add({
          key: `detector:${data.event_id}`,
          ts: detector.ts,
          tier: data.tier,
          title: label(data.type),
          detail: `Detector event · score ${Math.round(data.score)}`,
          kind: 'detector',
        })
      }

      const alert = bus.getLatest('alert.state') as TopicEnvelope<'alert.state'> | undefined
      if (alert && alert !== seenAlert.current) {
        seenAlert.current = alert
        const data: AlertState = alert.data
        if (data.status === 'closed') {
          add({
            key: `closed:${data.alert_id}`,
            ts: alert.ts,
            tier: data.tier,
            title: data.text || label(data.type),
            detail: data.intent ? `Alert closed · ${data.intent}` : 'Alert closed',
            kind: 'closed-alert',
          })
        }
      }
    }

    readLatest()
    return bus.subscribe(readLatest)
  }, [])

  return (
    <section className="panel-placeholder copilot-timeline-panel event-timeline" aria-label="Event timeline">
      <div className="timeline-heading">
        <p className="panel-label">EVENT TIMELINE</p>
        <span>{entries.length ? `${entries.length} recent` : 'Live events'}</span>
      </div>
      {entries.length ? (
        <ol className="timeline-list" aria-live="polite">
          {entries.map((entry) => (
            <li className={`timeline-entry timeline-entry--${entry.tier.toLowerCase()}`} key={entry.key}>
              <span className={`timeline-marker timeline-marker--${entry.tier.toLowerCase()}`} aria-hidden="true" />
              <div className="timeline-entry-copy">
                <div className="timeline-entry-topline">
                  <strong>{entry.title}</strong>
                  <time dateTime={new Date(entry.ts * 1000).toISOString()}>
                    {new Date(entry.ts * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                  </time>
                </div>
                <span>{entry.detail}</span>
              </div>
              <span className="timeline-tier">{entry.tier}</span>
            </li>
          ))}
        </ol>
      ) : (
        <p className="timeline-empty">Waiting for detector events and closed alerts.</p>
      )}
    </section>
  )
}
