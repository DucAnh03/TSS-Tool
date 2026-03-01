import { useState, useEffect, useRef } from 'react'

const WS_URL     = 'ws://localhost:8765'
const STATUS_MAP = {
  idle:       'IDLE',
  detected:   'WAKE WORD !!',
  recording:  'ĐANG NGHE',
  processing: 'XỬ LÝ…',
}

// ── WebSocket hook ─────────────────────────────────────────────────────────

function useJarvisSocket() {
  const [data, setData] = useState({
    wsConnected:  false,
    vpsConnected: false,
    status:       'idle',
    rms:          0,
    score:        0,
    transcripts:  [],
    logs:         [],
  })

  useEffect(() => {
    let ws, timer
    const connect = () => {
      ws = new WebSocket(WS_URL)
      ws.onopen  = () => setData(d => ({ ...d, wsConnected: true }))
      ws.onclose = () => {
        setData(d => ({ ...d, wsConnected: false }))
        timer = setTimeout(connect, 2000)
      }
      ws.onerror = () => ws.close()
      ws.onmessage = ({ data: raw }) => {
        const msg = JSON.parse(raw)
        setData(d => {
          switch (msg.type) {
            case 'rms':        return { ...d, rms: msg.value }
            case 'score':      return { ...d, score: msg.value }
            case 'status':     return { ...d, status: msg.status }
            case 'vps_status': return { ...d, vpsConnected: msg.connected }
            case 'transcript': return {
              ...d,
              transcripts: [
                { text: msg.text, time: now() },
                ...d.transcripts,
              ].slice(0, 30),
            }
            case 'log': return {
              ...d,
              logs: [
                { level: msg.level, msg: msg.msg, time: now() },
                ...d.logs,
              ].slice(0, 100),
            }
            default: return d
          }
        })
      }
    }
    connect()
    return () => { clearTimeout(timer); ws?.close() }
  }, [])

  return data
}

function now() {
  return new Date().toLocaleTimeString('vi-VN', { hour12: false })
}

// ── Components ─────────────────────────────────────────────────────────────

function Meter({ label, value, cls, unit = '' }) {
  const pct = Math.min(100, Math.round(value * 100))
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
      <div className="meter-label">
        <span>{label}</span>
        <span style={{ fontVariantNumeric: 'tabular-nums' }}>
          {value.toFixed(2)}{unit}
        </span>
      </div>
      <div className="meter-wrap">
        <div className={`meter-fill ${cls}`} style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}

function ConnBadge({ label, on }) {
  return (
    <span style={{ display: 'flex', alignItems: 'center', fontSize: 12, color: on ? 'var(--green)' : 'var(--muted)' }}>
      <span className={`dot ${on ? 'on' : 'off'}`} />
      {label}
    </span>
  )
}

// ── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const d          = useJarvisSocket()
  const logRef     = useRef(null)
  const txRef      = useRef(null)

  // auto-scroll logs
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = 0
  }, [d.logs.length])

  return (
    <div className="app">
      {/* ── Header ── */}
      <div className="header">
        <h1>🤖 Jarvis Dashboard</h1>
        <div style={{ display: 'flex', gap: 16, alignItems: 'center' }}>
          <ConnBadge label="Python client" on={d.wsConnected} />
          <ConnBadge label="VPS Contabo"   on={d.vpsConnected} />
        </div>
      </div>

      {/* ── Top row: meters + status ── */}
      <div className="grid-top">
        <div className="card">
          <div className="card-title">Microphone (RMS)</div>
          <Meter label="Âm lượng mic" value={d.rms} cls="rms" />
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            {d.rms < 0.01 ? '⚠ Mic có thể đang tắt hoặc quá yếu' : 'Mic hoạt động ✓'}
          </div>
        </div>

        <div className="card">
          <div className="card-title">Wake Word Score — hey_jarvis</div>
          <Meter label="Confidence" value={d.score} cls="score" />
          <div style={{ fontSize: 11, color: 'var(--muted)' }}>
            Ngưỡng kích hoạt: 0.50 &nbsp;|&nbsp;
            {d.score > 0.3
              ? <span style={{ color: 'var(--yellow)' }}>Đang gần ngưỡng…</span>
              : 'Chưa phát hiện'}
          </div>
        </div>

        <div className="card" style={{ alignItems: 'center', justifyContent: 'center' }}>
          <div className="card-title">Trạng thái hệ thống</div>
          <span className={`badge ${d.status}`} style={{ fontSize: 15, padding: '8px 20px' }}>
            {STATUS_MAP[d.status] ?? d.status.toUpperCase()}
          </span>
          {!d.wsConnected && (
            <div style={{ fontSize: 11, color: 'var(--red)', marginTop: 4 }}>
              Mất kết nối — đang thử lại…
            </div>
          )}
        </div>
      </div>

      {/* ── Bottom row: transcripts + logs ── */}
      <div className="grid-bot">
        <div className="card">
          <div className="card-title">Transcript ({d.transcripts.length})</div>
          <div className="transcript-list" ref={txRef}>
            {d.transcripts.length === 0
              ? <div className="empty">Chưa có transcript nào.<br />Nói "Hey Jarvis" để bắt đầu.</div>
              : d.transcripts.map((t, i) => (
                <div className="transcript-item" key={i}>
                  <div className="transcript-text">{t.text || <em style={{ color: 'var(--muted)' }}>(trống)</em>}</div>
                  <div className="transcript-time">{t.time}</div>
                </div>
              ))
            }
          </div>
        </div>

        <div className="card">
          <div className="card-title">Debug Log</div>
          <div className="log-list" ref={logRef}>
            {d.logs.length === 0
              ? <div className="empty">Chưa có log.</div>
              : d.logs.map((l, i) => (
                <div className="log-row" key={i}>
                  <span className="log-time">{l.time}</span>
                  <span className={`log-level ${l.level}`}>{l.level.toUpperCase()}</span>
                  <span className={`log-msg ${l.level}`}>{l.msg}</span>
                </div>
              ))
            }
          </div>
        </div>
      </div>
    </div>
  )
}
