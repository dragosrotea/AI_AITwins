import { useEffect, useState } from 'react'
import { bus, busUrl, type Envelope } from './bus'
import { TOPICS, type Topic } from './types'

interface TopicRate {
  rate_hz: number
  total: number
}

interface BrokerStats {
  topics: Record<string, TopicRate>
}

function getStatsUrl() {
  const url = new URL(busUrl)
  url.protocol = url.protocol === 'wss:' ? 'https:' : 'http:'
  url.pathname = '/stats'
  url.search = ''
  url.hash = ''
  return url.toString()
}

function useTopicRates() {
  const [rates, setRates] = useState<Record<string, TopicRate>>({})
  const [available, setAvailable] = useState(true)

  useEffect(() => {
    let stopped = false
    let timer: number | undefined

    async function poll() {
      try {
        const response = await fetch(getStatsUrl())
        if (!response.ok) throw new Error(`Broker stats returned ${response.status}`)
        const stats = await response.json() as BrokerStats
        if (!stopped) {
          setRates(stats.topics ?? {})
          setAvailable(true)
        }
      } catch {
        if (!stopped) setAvailable(false)
      } finally {
        if (!stopped) timer = window.setTimeout(poll, 1000)
      }
    }

    void poll()
    return () => {
      stopped = true
      if (timer !== undefined) window.clearTimeout(timer)
    }
  }, [])

  return { rates, available }
}

function useLatestMessages() {
  const [messages, setMessages] = useState(() => bus.getAllLatest())

  useEffect(() => {
    const update = () => setMessages(bus.getAllLatest())
    update()
    return bus.subscribe(update)
  }, [])

  return messages
}

function compactPayload(value: unknown, key = ''): unknown {
  if (key === 'jpeg_b64' && typeof value === 'string') {
    return `[JPEG frame omitted (${value.length.toLocaleString()} base64 characters)]`
  }
  if (typeof value === 'string' && value.length > 500) {
    return `${value.slice(0, 500)}… [${value.length - 500} more characters]`
  }
  if (Array.isArray(value)) {
    const shown = value.slice(0, 20).map((item) => compactPayload(item))
    if (value.length > shown.length) shown.push(`… ${value.length - shown.length} more items`)
    return shown
  }
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value).map(([childKey, child]) => [childKey, compactPayload(child, childKey)]),
    )
  }
  return value
}

function formatTs(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString()
}

export default function MessageLog() {
  const messages = useLatestMessages()
  const { rates, available } = useTopicRates()

  return (
    <section className="message-log" aria-labelledby="message-log-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">DEBUG · LIVE BUS</p>
          <h2 id="message-log-title">Message log</h2>
          <p className="page-description">Topic rates come from the broker. Payloads update as messages arrive.</p>
        </div>
        <span className={`stats-state ${available ? 'stats-state--live' : 'stats-state--offline'}`} role="status">
          <span className="connection-dot" />
          {available ? 'Rates live' : 'Rates unavailable'}
        </span>
      </div>

      <div className="message-table-wrap">
        <table className="message-table">
          <thead>
            <tr>
              <th scope="col">Topic</th>
              <th scope="col">Rate</th>
              <th scope="col">Last received</th>
              <th scope="col">Latest payload</th>
            </tr>
          </thead>
          <tbody>
            {TOPICS.map((topic: Topic) => {
              const message: Envelope | undefined = messages.get(topic)
              const rate = rates[topic]
              return (
                <tr key={topic}>
                  <th className="topic-name" scope="row">{topic}</th>
                  <td className="rate-cell">{(rate?.rate_hz ?? 0).toFixed(2)} <span>Hz</span></td>
                  <td className="timestamp-cell">{message ? formatTs(message.ts) : 'Waiting'}</td>
                  <td>
                    {message ? (
                      <pre className="payload-preview">{JSON.stringify(compactPayload(message.data), null, 2)}</pre>
                    ) : (
                      <span className="waiting-payload">No message received yet</span>
                    )}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}
