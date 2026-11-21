import type { Envelope, Topic, TopicDataMap } from './types'

export type { Envelope } from './types'

export interface SeriesPoint {
  ts: number
  value: number
}

export type BusStatus = 'connecting' | 'connected' | 'reconnecting'
type Listener = () => void

const WINDOW_SECONDS = 60
const MAX_POINTS_PER_SERIES = 1_200
const INITIAL_RECONNECT_MS = 250
const MAX_RECONNECT_MS = 10_000

/** A bounded, time-windowed ring for one numeric chart series. */
class TimeRingBuffer<T extends { ts: number }> {
  private values: T[] = []

  push(value: T) {
    this.values.push(value)
    const cutoff = value.ts - WINDOW_SECONDS
    let first = 0
    while (first < this.values.length && this.values[first].ts < cutoff) first += 1
    if (first > 0) this.values.splice(0, first)
    if (this.values.length > MAX_POINTS_PER_SERIES) {
      this.values.splice(0, this.values.length - MAX_POINTS_PER_SERIES)
    }
  }

  snapshot() {
    return this.values.slice()
  }
}

function flattenNumbers(value: unknown, prefix = ''): Array<[string, number]> {
  if (typeof value === 'number' && Number.isFinite(value)) return [[prefix, value]]
  if (!value || typeof value !== 'object' || Array.isArray(value)) return []

  return Object.entries(value).flatMap(([key, child]) =>
    flattenNumbers(child, prefix ? `${prefix}.${key}` : key),
  )
}

/** Small external store so React can consume bus updates without a UI dependency. */
class BusStore {
  private socket?: WebSocket
  private reconnectTimer?: ReturnType<typeof setTimeout>
  private reconnectDelay = INITIAL_RECONNECT_MS
  private started = false
  private status: BusStatus = 'connecting'
  private readonly messages = new Map<string, Envelope>()
  private readonly series = new Map<string, TimeRingBuffer<SeriesPoint>>()
  private readonly messageListeners = new Set<Listener>()
  private readonly statusListeners = new Set<(status: BusStatus) => void>()

  constructor(private readonly url: string) {}

  start() {
    if (this.started) return
    this.started = true
    this.connect()
  }

  close() {
    this.started = false
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.reconnectTimer = undefined
    this.socket?.close()
    this.socket = undefined
    this.setStatus('reconnecting')
  }

  getStatus = () => this.status

  subscribe(listener: Listener) {
    this.messageListeners.add(listener)
    return () => {
      this.messageListeners.delete(listener)
    }
  }

  subscribeStatus(listener: (status: BusStatus) => void) {
    this.statusListeners.add(listener)
    return () => {
      this.statusListeners.delete(listener)
    }
  }

  getLatest(topic: string) {
    return this.messages.get(topic)
  }

  getAllLatest() {
    return new Map(this.messages)
  }

  getSeries(key: string) {
    return this.series.get(key)?.snapshot() ?? []
  }

  publish<K extends Topic>(topic: K, data: TopicDataMap[K]) {
    if (!this.socket || this.socket.readyState !== WebSocket.OPEN) return false
    this.socket.send(JSON.stringify({
      op: 'pub',
      topic,
      ts: Date.now() / 1000,
      src: 'ui',
      data,
    }))
    return true
  }

  private connect() {
    if (!this.started) return
    this.setStatus(this.reconnectDelay === INITIAL_RECONNECT_MS ? 'connecting' : 'reconnecting')

    let socket: WebSocket
    try {
      socket = new WebSocket(this.url)
    } catch {
      this.scheduleReconnect()
      return
    }
    this.socket = socket

    socket.addEventListener('open', () => {
      if (this.socket !== socket) return
      this.reconnectDelay = INITIAL_RECONNECT_MS
      this.setStatus('connected')
      // The broker contract accepts a trailing * as a prefix wildcard.
      socket.send(JSON.stringify({ op: 'sub', topics: ['*'] }))
    })
    socket.addEventListener('message', (event: MessageEvent<string>) => {
      if (typeof event.data !== 'string') return
      try {
        const envelope = JSON.parse(event.data) as Partial<Envelope>
        if (typeof envelope.topic !== 'string' || typeof envelope.ts !== 'number' || !('data' in envelope)) return
        this.accept(envelope as Envelope)
      } catch {
        // Ignore malformed frames and keep the connection alive.
      }
    })
    socket.addEventListener('close', () => {
      if (this.socket === socket) this.scheduleReconnect()
    })
    socket.addEventListener('error', () => socket.close())
  }

  private accept(envelope: Envelope) {
    this.messages.set(envelope.topic, envelope)
    for (const [field, value] of flattenNumbers(envelope.data)) {
      if (!field) continue
      const key = `${envelope.topic}.${field}`
      let buffer = this.series.get(key)
      if (!buffer) {
        buffer = new TimeRingBuffer<SeriesPoint>()
        this.series.set(key, buffer)
      }
      buffer.push({ ts: envelope.ts, value })
    }
    this.messageListeners.forEach((listener) => listener())
  }

  private scheduleReconnect() {
    if (!this.started || this.reconnectTimer) return
    this.socket = undefined
    this.setStatus('reconnecting')
    this.reconnectTimer = setTimeout(() => {
      this.reconnectTimer = undefined
      this.connect()
    }, this.reconnectDelay)
    this.reconnectDelay = Math.min(this.reconnectDelay * 2, MAX_RECONNECT_MS)
  }

  private setStatus(status: BusStatus) {
    if (this.status === status) return
    this.status = status
    this.statusListeners.forEach((listener) => listener(status))
  }
}

export const busUrl = import.meta.env.VITE_BUS_URL || 'ws://localhost:8765/bus'
export const bus = new BusStore(busUrl)
bus.start()
