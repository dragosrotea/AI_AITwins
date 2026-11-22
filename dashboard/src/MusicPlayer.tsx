import { useEffect, useRef, useState, type CSSProperties } from 'react'
import { bus, type Envelope } from './bus'
import type { ControlMusic } from './types'

const tracks = Object.entries(
  import.meta.glob<string>('../../assets/music/*.mp3', {
    eager: true,
    query: '?url',
    import: 'default',
  }),
).map(([path, src]) => ({
  src,
  title: path.split('/').at(-1)?.replace(/\.mp3$/i, '') ?? 'S.A.M. track',
}))

function isControlMusic(value: unknown): value is ControlMusic {
  if (!value || typeof value !== 'object') return false
  const data = value as Record<string, unknown>
  return typeof data.duck === 'boolean'
    && typeof data.level === 'number'
    && data.level >= 0
    && data.level <= 1
}

export default function MusicPlayer() {
  const audioRef = useRef<HTMLAudioElement>(null)
  const lastControlMessage = useRef<Envelope | undefined>(undefined)
  const [trackIndex, setTrackIndex] = useState(0)
  const [volume, setVolume] = useState(65)
  const [playing, setPlaying] = useState(false)
  const [duckLevel, setDuckLevel] = useState<number | null>(null)
  const [playbackError, setPlaybackError] = useState('')
  const track = tracks[trackIndex]
  const effectiveVolume = Math.min(volume / 100, duckLevel ?? 1)

  useEffect(() => {
    const readMusicControl = () => {
      const message = bus.getLatest('control.music')
      if (!message || message === lastControlMessage.current) return
      lastControlMessage.current = message
      if (!isControlMusic(message.data)) return
      const control = message.data
      setDuckLevel(control.duck ? control.level : null)
    }

    readMusicControl()
    return bus.subscribe(readMusicControl)
  }, [])

  useEffect(() => {
    if (!track) return
    const audio = audioRef.current
    if (audio) audio.volume = effectiveVolume
    bus.publish('music.volume', {
      volume: Math.round(effectiveVolume * 100),
      playing,
    })
  }, [effectiveVolume, playing, track, trackIndex])

  async function togglePlayback() {
    const audio = audioRef.current
    if (!audio || !track) return
    setPlaybackError('')
    if (!audio.paused) {
      audio.pause()
      return
    }

    try {
      await audio.play()
    } catch {
      setPlaybackError('This track could not be played in the browser.')
    }
  }

  function chooseTrack(index: number) {
    audioRef.current?.pause()
    setPlaying(false)
    setPlaybackError('')
    setTrackIndex(index)
  }

  return (
    <section className="music-player" aria-labelledby="music-player-title">
      {track && (
        <audio
          key={track.src}
          ref={audioRef}
          src={track.src}
          preload="metadata"
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onError={() => setPlaybackError('This track could not be loaded.')}
        />
      )}
      <div className="music-player-heading">
        <div>
          <p className="panel-label">MUSIC PLAYER</p>
          <h3 id="music-player-title">Cabin audio</h3>
        </div>
        <span className={`music-state${playing ? ' music-state--playing' : ''}`}>
          <span className="connection-dot" />
          {playing ? (duckLevel === null ? 'Playing' : 'Ducked') : 'Paused'}
        </span>
      </div>

      {track ? (
        <>
          <label className="track-select-label" htmlFor="music-track">Track</label>
          <select
            id="music-track"
            className="track-select"
            value={trackIndex}
            onChange={(event) => chooseTrack(Number(event.currentTarget.value))}
          >
            {tracks.map((item, index) => <option key={item.src} value={index}>{item.title}</option>)}
          </select>

          <div className="music-controls">
            <button className="music-play-button" type="button" onClick={() => void togglePlayback()}>
              {playing ? 'Pause' : 'Play'}
            </button>
            <label className="volume-control">
              <span>Volume <strong>{Math.round(effectiveVolume * 100)}%</strong></span>
              <input
                type="range"
                min="0"
                max="100"
                value={volume}
                aria-label="Music volume"
                style={{ '--volume-progress': `${volume}%` } as CSSProperties}
                onChange={(event) => setVolume(Number(event.currentTarget.value))}
              />
            </label>
          </div>
          {duckLevel !== null && <p className="duck-note">Volume is temporarily lowered by the safety copilot.</p>}
          {playbackError && <p className="playback-error" role="status">{playbackError}</p>}
        </>
      ) : (
        <p className="music-empty">Add an MP3 track to <code>assets/music/</code> to enable playback.</p>
      )}
    </section>
  )
}
