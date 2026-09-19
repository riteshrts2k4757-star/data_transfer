import { useEffect, useRef, useState } from 'react'
import { Activity, AudioLines, Check, ChevronDown, CircleHelp, Cpu, Disc3, Download, Gauge, Headphones, Mic, Pause, Play, Plus, Radio, Settings, SlidersHorizontal, Sparkles, Upload, Volume2, Waves, X, Zap } from 'lucide-react'
import AudioDenoisingPage from './pages/AudioDenoising.jsx'
import { audioInfo, createPlayableBuffer, decodeAudioFile, playBuffer, startMicrophone } from './services/audioService.js'
import { analyzeSignal, simulateAIEnhancement } from './services/aiService.js'
import { downsample, formatDuration, spectrumFromSamples } from './utils/waveform.js'

const noiseLibrary = [
  { name: 'helicopter_0001.wav', type: 'Helicopter', duration: '00:10', color: 'cyan' },
  { name: 'heavyEngine_0001.wav', type: 'Heavy engine', duration: '00:10', color: 'amber' },
  { name: 'artilleryFire_0001.wav', type: 'Artillery fire', duration: '00:10', color: 'rose' },
  { name: 'gunFire_0001.wav', type: 'Gunfire', duration: '00:10', color: 'violet' },
]

const stages = ['INPUT RECEIVED', 'ANALYZING SIGNAL', 'EXTRACTING FEATURES', 'NEURAL FILTER', 'NOISE SUPPRESSION', 'OUTPUT GENERATED']

function makeDemoSamples(length = 9000, clean = false) {
  return Float32Array.from({ length }, (_, index) => {
    const t = index / 16000
    const voice = Math.sin(t * 31) * 0.25 + Math.sin(t * 87) * 0.14 + Math.sin(t * 142) * 0.08
    const noise = Math.sin(index * 1.91) * 0.08 + Math.sin(index * 5.7) * 0.035
    return clean ? voice : voice + noise
  })
}

function Header({ live, onLive, activeView, onChangeView }) {
  return <header className="topbar">
    <div className="brand"><div className="brand-mark"><Activity size={18} /></div><div><strong>NEUROFILTER</strong><span>AI SPEECH ENHANCEMENT SYSTEM</span></div></div>
    <nav className="app-nav" aria-label="Main navigation">
      <button className={activeView === 'dashboard' ? 'nav-button active' : 'nav-button'} onClick={() => onChangeView('dashboard')}>Dashboard</button>
      <button className={activeView === 'denoise' ? 'nav-button active' : 'nav-button'} onClick={() => onChangeView('denoise')}>AI Noise Cancellation</button>
    </nav>
    <div className="header-readout"><span className="online-dot" /> SYSTEM ONLINE <i /> <b>MODEL: ANC-NET v1.0</b> <i /> <b>16 kHz</b> <i /> <b>REAL-TIME</b></div>
    <div className="header-actions"><button className={`live-toggle ${live ? 'is-live' : ''}`} onClick={onLive}><Radio size={14} /> {live ? 'LIVE' : 'LIVE MODE'}</button><button className="icon-button" title="Settings"><Settings size={17} /></button></div>
  </header>
}

function Waveform({ samples, color, playing, label }) {
  const canvasRef = useRef(null)
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return undefined
    const context = canvas.getContext('2d')
    const ratio = window.devicePixelRatio || 1
    const rect = canvas.getBoundingClientRect()
    canvas.width = rect.width * ratio
    canvas.height = rect.height * ratio
    context.scale(ratio, ratio)
    const values = downsample(samples, Math.max(80, Math.floor(rect.width / 2.4)))
    let frame = 0
    const draw = () => {
      context.clearRect(0, 0, rect.width, rect.height)
      const gradient = context.createLinearGradient(0, 0, rect.width, 0)
      gradient.addColorStop(0, color === 'cyan' ? '#35d9ff' : '#52f5a3')
      gradient.addColorStop(1, color === 'cyan' ? '#8cecff' : '#bbffd8')
      context.strokeStyle = gradient
      context.lineWidth = 1.6
      context.shadowColor = color === 'cyan' ? 'rgba(53,217,255,.48)' : 'rgba(82,245,163,.44)'
      context.shadowBlur = 10
      context.beginPath()
      values.forEach((value, index) => {
        const x = (index / Math.max(1, values.length - 1)) * rect.width
        const wobble = playing ? Math.sin(index * 0.38 + frame * 0.05) * 2 : 0
        const y = rect.height / 2 + (index % 3 === 0 ? wobble : 0) - value * rect.height * 0.39
        const y2 = rect.height / 2 + wobble + value * rect.height * 0.39
        context.moveTo(x, y)
        context.lineTo(x, y2)
      })
      context.stroke()
      if (playing) { frame += 1; requestAnimationFrame(draw) }
    }
    draw()
    return () => { frame = -1 }
  }, [samples, color, playing])
  return <div className="waveform-wrap"><canvas ref={canvasRef} aria-label={`${label} waveform`} /><div className="waveform-axis"><span>0:00</span><span>00:10</span></div></div>
}

function AudioPanel({ kind, samples, info, playing, onPlay, onUpload, onRemove, category, setCategory }) {
  const noisy = kind === 'noisy'
  return <section className={`audio-panel ${noisy ? 'noisy-panel' : 'clean-panel'}`}>
    <div className="panel-kicker"><span className="signal-index">0{noisy ? 1 : 3}</span><span>{noisy ? 'SOURCE SIGNAL' : 'ENHANCED OUTPUT'}</span><span className="panel-live"><span /> {playing ? 'PLAYING' : 'READY'}</span></div>
    <div className="panel-heading"><div><h2>{noisy ? 'NOISY VOICE' : 'CLEAN VOICE'}</h2><p>{noisy ? 'Raw acoustic input' : 'Demo-mode reconstruction'}</p></div><span className={`signal-badge ${noisy ? 'cyan' : 'green'}`}>{noisy ? 'INPUT' : 'OUTPUT'}</span></div>
    <div className="wave-area">{info ? <Waveform samples={samples} color={noisy ? 'cyan' : 'green'} playing={playing} label={kind} /> : <div className="empty-wave"><AudioLines size={27} /><span>AWAITING AUDIO INPUT</span><small>Upload a recording to inspect the signal</small></div>}</div>
    {noisy && <div className="upload-actions"><label className="upload-button"><Upload size={15} /> {info ? 'Change audio' : 'Upload audio'}<input type="file" accept="audio/*" onChange={onUpload} /></label>{info && <button className="text-button" onClick={onRemove}><X size={14} /> Remove</button>}</div>}
    <div className="audio-meta"><div><span>FILE</span><strong>{info?.name || 'No source loaded'}</strong></div><div><span>DURATION</span><strong>{formatDuration(info?.duration)}</strong></div><div><span>FORMAT</span><strong>{info ? `${Math.round(info.sampleRate / 1000)} kHz / ${info.channels === 1 ? 'MONO' : 'STEREO'}` : '--'}</strong></div></div>
    {noisy && <div className="category-select"><span>NOISE TYPE</span><label><select value={category} onChange={(event) => setCategory(event.target.value)}>{['Helicopter', 'Engine', 'Artillery', 'Gunfire', 'Explosion', 'Drone', 'Wind', 'Siren', 'Custom'].map((item) => <option key={item}>{item}</option>)}</select><ChevronDown size={14} /></label></div>}
    <button className={`play-button ${playing ? 'playing' : ''}`} onClick={onPlay} disabled={!info}><span>{playing ? <Pause size={16} fill="currentColor" /> : <Play size={16} fill="currentColor" />}</span>{playing ? 'Pause audio' : `Play ${noisy ? 'noisy' : 'enhanced'} audio`}</button>
  </section>
}

function AIFilter({ processing, progress, stage }) {
  return <section className={`filter-core ${processing ? 'processing' : ''}`}><div className="core-orbit orbit-one" /><div className="core-orbit orbit-two" /><div className="core-label"><span className="signal-index">02</span><span>PROCESSING CORE</span></div><div className="neural-network">{Array.from({ length: 15 }, (_, index) => <span key={index} className={`node node-${index + 1}`} />)}{Array.from({ length: 11 }, (_, index) => <i key={index} className={`connection connection-${index + 1}`} />)}</div><div className="filter-title"><Cpu size={18} /><span>AI NEURAL FILTER</span></div><div className="filter-status"><span className={processing ? 'pulse-dot' : 'idle-dot'} />{processing ? stage : 'SYSTEM READY'}</div><div className="progress-bar"><span style={{ width: `${progress}%` }} /></div><div className="core-details"><div><span>MODEL</span><b>ANC-NET v1.0</b></div><div><span>MODE</span><b>DEMO / DSP</b></div><div><span>STATUS</span><b className={processing ? 'accent-text' : ''}>{processing ? 'PROCESSING' : 'IDLE'}</b></div></div></section>
}

function SignalFlow({ processing }) { return <div className={`signal-flow ${processing ? 'active' : ''}`}><span /><span /><span /><span /><span /></div> }

function Spectrum({ input, output }) {
  const inputBars = spectrumFromSamples(input, 34)
  const outputBars = spectrumFromSamples(output, 34)
  return <section className="spectrum-section"><div className="section-heading"><div><span className="eyebrow"><Waves size={14} /> FREQUENCY ANALYSIS</span><h2>Signal spectrum</h2></div><span className="range-label">20 Hz <i /> 8 kHz</span></div><div className="spectra"><div className="spectrum-block"><div className="spectrum-title"><span>INPUT / NOISY</span><b>HIGH NOISE FLOOR</b></div><div className="bars input-bars">{inputBars.map((value, index) => <i key={index} style={{ height: `${Math.max(8, value * 100)}%` }} />)}</div></div><div className="spectral-arrow"><Zap size={15} /></div><div className="spectrum-block"><div className="spectrum-title"><span>OUTPUT / ENHANCED</span><b className="green-text">SUPPRESSED</b></div><div className="bars output-bars">{outputBars.map((value, index) => <i key={index} style={{ height: `${Math.max(5, value * 76)}%` }} />)}</div></div></div></section>
}

function Metrics({ metrics }) { return <div className="metrics"><div><span>INPUT SNR</span><strong>{metrics.inputSnr} <small>dB</small></strong><em>SIMULATED</em></div><div><span>OUTPUT SNR</span><strong>{metrics.outputSnr} <small>dB</small></strong><em>SIMULATED</em></div><div><span>NOISE REDUCTION</span><strong>{metrics.reduction}</strong><em>SIMULATED</em></div><div><span>LATENCY</span><strong>{metrics.latency}</strong><em>ESTIMATED</em></div><div className="metric-status"><span>PIPELINE STATUS</span><strong><Check size={14} /> ENHANCED</strong><em>DEMO MODE</em></div></div> }

function Controls({ settings, setSettings, onProcess, processing, hasInput }) { return <section className="side-panel controls"><div className="side-heading"><div><span className="eyebrow"><SlidersHorizontal size={14} /> DSP PARAMETERS</span><h2>Processing controls</h2></div><Gauge size={17} /></div><label className="range-control"><span>Noise reduction <b>{settings.reduction}%</b></span><input type="range" min="0" max="100" value={settings.reduction} onChange={(event) => setSettings({ ...settings, reduction: event.target.value })} /></label><label className="range-control"><span>Speech preservation <b>{settings.preservation}%</b></span><input type="range" min="0" max="100" value={settings.preservation} onChange={(event) => setSettings({ ...settings, preservation: event.target.value })} /></label><div className="switch-row"><span>AI enhancement</span><button className="switch on"><i /></button></div><div className="switch-row"><span>Adaptive filtering</span><button className="switch on"><i /></button></div><button className="process-button" onClick={onProcess} disabled={!hasInput || processing}><Sparkles size={17} /> {processing ? 'Processing signal...' : 'Process audio'}<span>⌘ ↵</span></button></section> }

function NoiseLibrary({ items, onUse, onAdd }) { return <section className="side-panel library"><div className="side-heading"><div><span className="eyebrow"><Disc3 size={14} /> SESSION ASSETS</span><h2>Noise library</h2></div><span className="asset-count">{items.length} FILES</span></div>{items.map((item) => <div className="library-item" key={item.name}><div className={`asset-icon ${item.color || 'cyan'}`}><Volume2 size={15} /></div><div className="asset-info"><strong>{item.name}</strong><span>{item.type} <i /> {item.duration}</span></div><button onClick={() => onUse(item)} className="use-button">Use</button></div>)}<button className="add-button" onClick={onAdd}><Plus size={15} /> Add noise sample</button></section> }

export default function App() {
  const [activeView, setActiveView] = useState('dashboard')
  const [inputSamples, setInputSamples] = useState(makeDemoSamples())
  const [outputSamples, setOutputSamples] = useState(makeDemoSamples(9000, true))
  const [inputBuffer, setInputBuffer] = useState(null)
  const [outputBuffer, setOutputBuffer] = useState(null)
  const [info, setInfo] = useState(null)
  const [category, setCategory] = useState('Helicopter')
  const [playing, setPlaying] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stage, setStage] = useState('SYSTEM READY')
  const [live, setLive] = useState(false)
  const [settings, setSettings] = useState({ reduction: 72, preservation: 86 })
  const [metrics, setMetrics] = useState({ inputSnr: '3.2', outputSnr: '15.8', reduction: '12.6 dB', latency: '24 ms' })
  const [noiseItems, setNoiseItems] = useState(noiseLibrary)
  const [liveStream, setLiveStream] = useState(null)
  const contextRef = useRef(null)
  const fileInputRef = useRef(null)
  const noiseInputRef = useRef(null)

  useEffect(() => {
    if (!liveStream) return undefined
    const context = contextRef.current || new AudioContext()
    contextRef.current = context
    const analyser = context.createAnalyser()
    analyser.fftSize = 1024
    const source = context.createMediaStreamSource(liveStream)
    source.connect(analyser)
    const data = new Float32Array(analyser.fftSize)
    let frame
    const update = () => {
      analyser.getFloatTimeDomainData(data)
      setInputSamples(data.slice())
      setInfo({ name: 'microphone_live.wav', duration: 0, sampleRate: context.sampleRate, channels: 1, size: 0 })
      frame = requestAnimationFrame(update)
    }
    update()
    return () => { cancelAnimationFrame(frame); source.disconnect(); analyser.disconnect() }
  }, [liveStream])

  const upload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    const context = contextRef.current || new AudioContext()
    contextRef.current = context
    try {
      const buffer = await decodeAudioFile(file, context)
      const samples = buffer.getChannelData(0).slice()
      setInputBuffer(buffer); setInputSamples(samples); setOutputBuffer(null); setOutputSamples(simulateAIEnhancement(samples, settings.reduction, settings.preservation)); setInfo({ name: file.name, ...audioInfo(buffer, file) }); setPlaying(null)
    } catch { setStage('UNREADABLE AUDIO FILE') }
    event.target.value = ''
  }
  const process = async () => {
    if (!inputSamples.length) return
    setProcessing(true); setProgress(0)
    for (let index = 0; index < stages.length; index += 1) { setStage(stages[index]); setProgress(Math.round(((index + 1) / stages.length) * 100)); await new Promise((resolve) => setTimeout(resolve, 360)) }
    const enhanced = simulateAIEnhancement(inputSamples, settings.reduction, settings.preservation)
    setOutputSamples(enhanced)
    const context = contextRef.current || new AudioContext(); contextRef.current = context
    setOutputBuffer(createPlayableBuffer(context, enhanced, info?.sampleRate || 16000)); setMetrics(analyzeSignal(inputSamples, enhanced, info?.sampleRate || 16000)); setProcessing(false); setStage('OUTPUT GENERATED')
  }
  const play = (which) => { const buffer = which === 'input' ? inputBuffer : outputBuffer; if (buffer) { playBuffer(buffer, contextRef.current); setPlaying(which); setTimeout(() => setPlaying(null), buffer.duration * 1000) } }
  const remove = () => { setInfo(null); setInputBuffer(null); setOutputBuffer(null); setInputSamples(makeDemoSamples()); setOutputSamples(makeDemoSamples(9000, true)); setStage('SYSTEM READY') }
  const microphone = async () => { if (live) { liveStream?.getTracks().forEach((track) => track.stop()); setLiveStream(null); setLive(false); return } try { const stream = await startMicrophone(() => {}); setLiveStream(stream); setLive(true) } catch { setStage('MIC ACCESS DENIED') } }
  const addNoiseFiles = (event) => { const files = Array.from(event.target.files || []); setNoiseItems((current) => [...current, ...files.map((file) => ({ name: file.name, type: 'Custom', duration: 'SESSION', color: 'cyan' }))]); event.target.value = '' }

  if (activeView === 'denoise') {
    return <div className="app-shell"><Header live={live} onLive={microphone} activeView={activeView} onChangeView={setActiveView} /><AudioDenoisingPage /></div>
  }

  return <div className="app-shell"><Header live={live} onLive={microphone} activeView={activeView} onChangeView={setActiveView} /><main><div className="hero-intro"><div><span className="eyebrow"><span className="eyebrow-pulse" /> REAL-TIME AUDIO LABORATORY</span><h1>Make signal <em>intelligent.</em></h1><p>Separate voice from noise with an adaptive neural processing pipeline.</p></div><div className="mode-tag"><span /> MODE: <b>DEMO</b><CircleHelp size={14} /></div></div>
    <div className="pipeline"><AudioPanel kind="noisy" samples={inputSamples} info={info} playing={playing === 'input'} onPlay={() => play('input')} onUpload={upload} onRemove={remove} category={category} setCategory={setCategory} /><SignalFlow processing={processing} /><AIFilter processing={processing} progress={progress} stage={stage} /><SignalFlow processing={processing} /><AudioPanel kind="clean" samples={outputSamples} info={outputBuffer ? { ...info, name: 'enhanced_demo.wav' } : null} playing={playing === 'output'} onPlay={() => play('output')} /></div>
    <Metrics metrics={metrics} /><Spectrum input={inputSamples} output={outputSamples} /><div className="lower-grid"><Controls settings={settings} setSettings={setSettings} onProcess={process} processing={processing} hasInput={Boolean(info)} /><NoiseLibrary items={noiseItems} onUse={(item) => setCategory(item.type)} onAdd={() => noiseInputRef.current?.click()} /></div>
    <input ref={fileInputRef} type="file" accept="audio/*" hidden onChange={upload} /><input ref={noiseInputRef} type="file" accept="audio/*" multiple hidden onChange={addNoiseFiles} />
  </main><footer><span><Headphones size={14} /> Browser audio engine active</span><span>AI MODEL NOT CONNECTED <i /> LOCAL DSP PREVIEW</span><span>NEUROFILTER / 2026</span></footer></div>
}