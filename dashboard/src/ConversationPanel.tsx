import { useEffect, useState } from 'react'
import { bus } from './bus'
import type { Topic, TopicEnvelope } from './types'

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

function statusLabel(status?: string) {
  if (!status) return 'Waiting'
  return status.replaceAll('_', ' ')
}

export default function ConversationPanel() {
  useBusUpdates()
  const [now, setNow] = useState(() => Date.now())

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 100)
    return () => window.clearInterval(timer)
  }, [])

  const alertMessage = latest('alert.state')
  const voiceStatusMessage = latest('voice.status')
  const voiceSayMessage = latest('voice.say')
  const voiceReplyMessage = latest('voice.reply')
  const alert = alertMessage?.data
  const voiceStatus = voiceStatusMessage?.data
  const voiceSay = voiceSayMessage?.data
  const voiceReply = voiceReplyMessage?.data

  const currentAlertId = alert?.alert_id ?? voiceSay?.alert_id ?? voiceStatus?.alert_id ?? undefined
  const relatedStatus = voiceStatus && (!currentAlertId || voiceStatus.alert_id === currentAlertId)
    ? voiceStatus
    : undefined
  const relatedSay = voiceSay && (!currentAlertId || voiceSay.alert_id === currentAlertId)
    ? voiceSay
    : undefined
  const relatedReply = voiceReply && (!currentAlertId || voiceReply.alert_id === currentAlertId)
    ? voiceReply
    : undefined

  const voiceState = relatedStatus?.state
  const stateLabel = voiceState ?? alert?.status ?? 'waiting'
  const messageText = alert?.text ?? relatedSay?.text
  const transcript = relatedReply?.transcript ?? alert?.reply
  const intent = relatedReply?.intent ?? alert?.intent
  const latencyMs = relatedReply?.latency_ms ?? alert?.latency_ms
  const isListening = voiceState === 'listening' || (!voiceState && alert?.status === 'listening')
  const listenSeconds = relatedSay?.listen_s ?? 8
  const listenStartedAt = voiceState === 'listening' && relatedStatus
    ? voiceStatusMessage?.ts
    : alert?.status === 'listening' ? alertMessage?.ts : undefined
  const remainingSeconds = isListening && listenStartedAt !== undefined
    ? Math.max(0, listenSeconds - (now / 1000 - listenStartedAt))
    : undefined
  const countdownPercent = remainingSeconds === undefined || listenSeconds <= 0
    ? 0
    : Math.max(0, Math.min(100, remainingSeconds / listenSeconds * 100))

  return (
    <section className="panel-placeholder copilot-conversation-panel conversation-panel" aria-label="Live conversation">
      <div className="conversation-heading">
        <div>
          <p className="panel-label">CONVERSATION</p>
          {alert && <p className="conversation-alert-type">{alert.type.replaceAll('_', ' ')} · {alert.tier}</p>}
        </div>
        <span className={`conversation-state conversation-state--${stateLabel.replaceAll('_', '-')}`} role="status" aria-live="polite">
          <span className="conversation-state-dot" />
          {statusLabel(stateLabel)}
        </span>
      </div>

      <div className="conversation-message-wrap">
        <span className="conversation-field-label">CURRENT MESSAGE</span>
        <p className="conversation-message">{messageText ?? 'Waiting for the next co-pilot message.'}</p>
      </div>

      {isListening && remainingSeconds !== undefined && (
        <div className="listen-countdown" aria-label={`${remainingSeconds.toFixed(1)} seconds remaining in the listening window`}>
          <div className="listen-countdown-label">
            <span>LISTEN WINDOW</span>
            <strong>{remainingSeconds.toFixed(1)} s</strong>
          </div>
          <div className="listen-countdown-track"><span style={{ width: `${countdownPercent}%` }} /></div>
        </div>
      )}

      <div className="conversation-reply-wrap">
        <span className="conversation-field-label">DRIVER REPLY</span>
        <p className={`conversation-transcript${transcript ? '' : ' conversation-transcript--empty'}`}>
          {transcript || (isListening ? 'Listening for a reply…' : 'No reply received yet')}
        </p>
        <div className="conversation-reply-meta">
          <span className={`conversation-intent${intent ? ' conversation-intent--set' : ''}`}>
            {intent ? `Intent · ${intent}` : 'Intent · —'}
          </span>
          <span>{latencyMs !== undefined && latencyMs !== null ? `${latencyMs} ms` : 'Latency · —'}</span>
        </div>
      </div>
    </section>
  )
}
