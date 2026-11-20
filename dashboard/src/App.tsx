import { useEffect, useState } from 'react'
import { bus, type BusStatus } from './bus'

export default function App() {
  const [status, setStatus] = useState<BusStatus>(bus.getStatus())

  useEffect(() => bus.subscribeStatus(setStatus), [])

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">D</div>
        <div>
          <p className="eyebrow">DRIVER SAFETY COPILOT</p>
          <h1>DriveMate</h1>
        </div>
        <div className={`connection connection--${status}`}>
          <span className="connection-dot" />
          {status === 'connected' ? 'Broker connected' : status === 'connecting' ? 'Connecting' : 'Reconnecting'}
        </div>
      </header>
      <section className="welcome-card">
        <p className="eyebrow">LIVE DASHBOARD</p>
        <h2>Your drive, with a second pair of eyes.</h2>
        <p className="intro">Live driver and vehicle signals will appear here as soon as the broker is online.</p>
        <div className="bus-address"><span className="pulse" />{import.meta.env.VITE_BUS_URL || 'ws://localhost:8765/bus'}</div>
      </section>
      <footer>DriveMate <span>·</span> Safety support for every drive</footer>
    </main>
  )
}
