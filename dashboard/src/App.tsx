import { useEffect, useState, type MouseEvent } from 'react'
import { bus, type BusStatus } from './bus'
import MessageLog from './MessageLog'
import MusicPlayer from './MusicPlayer'
import ConversationPanel from './ConversationPanel'
import ReplayControls from './ReplayControls'
import { AccelerationChart, CabinChart, DebugCorner, DriverCameraPanel, DriverMetricsPanel } from './TelemetryPanels'
import EventTimeline from './EventTimeline'
import { DriverStatusPanel, PreTripForm } from './DriverPanels'
import { CrashSafetyNotice } from './CrashStatus'
import { AttentionPanel, CheckinSchedulePanel, HeartRatePanel, RoadContextPanel, TripSummaryPanel } from './IntegrationPanels'

type RoutePath = '/driver' | '/copilot' | '/debug'

function routeFromPath(path: string): RoutePath {
  if (path === '/driver' || path === '/debug') return path
  return '/copilot'
}

function RouteLink({
  to,
  current,
  onNavigate,
  children,
}: {
  to: RoutePath
  current: RoutePath
  onNavigate: (event: MouseEvent<HTMLAnchorElement>, to: RoutePath) => void
  children: string
}) {
  return (
    <a
      className="route-link"
      href={to}
      aria-current={current === to ? 'page' : undefined}
      onClick={(event) => onNavigate(event, to)}
    >
      {children}
    </a>
  )
}

function DriverLayout() {
  return (
    <section className="route-page driver-page" aria-labelledby="driver-page-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">DRIVER VIEW</p>
          <h2 id="driver-page-title">DriveMate</h2>
        </div>
      </div>
      <div className="driver-panels">
        <DriverStatusPanel />
        <PreTripForm />
      </div>
    </section>
  )
}

function CopilotLayout() {
  return (
    <section className="route-page copilot-page" aria-labelledby="copilot-page-title">
      <div className="page-heading">
        <div>
          <p className="eyebrow">PASSENGER CONSOLE</p>
          <h2 id="copilot-page-title">Co-pilot</h2>
        </div>
      </div>
      <CrashSafetyNotice />
      <TripSummaryPanel />
      <div className="copilot-panels">
        <DriverMetricsPanel />
        <DriverCameraPanel />
        <EventTimeline />
        <ConversationPanel />
        <AttentionPanel />
        <HeartRatePanel />
        <RoadContextPanel />
        <CheckinSchedulePanel />
        <CabinChart />
        <MusicPlayer />
        <AccelerationChart />
        <ReplayControls />
      </div>
      <DebugCorner />
    </section>
  )
}

export default function App() {
  const [status, setStatus] = useState<BusStatus>(bus.getStatus())
  const [route, setRoute] = useState<RoutePath>(() => routeFromPath(window.location.pathname))

  useEffect(() => {
    const syncRoute = () => setRoute(routeFromPath(window.location.pathname))
    if (window.location.pathname === '/') window.history.replaceState(null, '', '/copilot')
    window.addEventListener('popstate', syncRoute)
    return () => window.removeEventListener('popstate', syncRoute)
  }, [])

  useEffect(() => bus.subscribeStatus(setStatus), [])

  function navigate(event: MouseEvent<HTMLAnchorElement>, to: RoutePath) {
    if (event.button !== 0 || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return
    event.preventDefault()
    if (window.location.pathname !== to) window.history.pushState(null, '', to)
    setRoute(to)
  }

  const page = route === '/driver'
    ? <DriverLayout />
    : route === '/debug'
      ? <MessageLog />
      : <CopilotLayout />

  return (
    <div className={`shell app-shell${route === '/driver' ? ' app-shell--driver' : ''}`}>
      <header className="topbar">
        <div className="brand-mark" aria-hidden="true">
          <img src="/brand/drivemate-mark.png" alt="" />
        </div>
        <div className="brand-copy">
          <p className="eyebrow">DRIVER SAFETY COPILOT</p>
          <h1>DriveMate</h1>
        </div>
        {route !== '/driver' && (
          <nav className="route-nav" aria-label="Dashboard views">
            <RouteLink to="/driver" current={route} onNavigate={navigate}>Driver</RouteLink>
            <RouteLink to="/copilot" current={route} onNavigate={navigate}>Co-pilot</RouteLink>
            <RouteLink to="/debug" current={route} onNavigate={navigate}>Message log</RouteLink>
          </nav>
        )}
        <div className={`connection connection--${status}`} role="status" aria-live="polite">
          <span className="connection-dot" />
          {status === 'connected' ? 'Broker connected' : status === 'connecting' ? 'Connecting' : 'Reconnecting'}
        </div>
      </header>
      {page}
      <footer>DriveMate <span>·</span> Safety support for every drive</footer>
    </div>
  )
}
