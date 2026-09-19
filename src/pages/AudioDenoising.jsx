import { useEffect, useRef, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8001'

function formatDuration(seconds) {
  if (!Number.isFinite(seconds) || seconds <= 0) return '00:00'
  const minutes = Math.floor(seconds / 60)
  const remainder = Math.floor(seconds % 60)
  return `${String(minutes).padStart(2, '0')}:${String(remainder).padStart(2, '0')}`
}

function formatFileSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '0 KB'
  const units = ['B', 'KB', 'MB', 'GB']
  const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1)
  const value = bytes / (1024 ** index)
  return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`
}

function toMonoSamples(audioBuffer) {
  const channelData = []
  for (let channel = 0; channel < audioBuffer.numberOfChannels; channel += 1) {
    channelData.push(audioBuffer.getChannelData(channel))
  }

  if (channelData.length === 1) return channelData[0]

  const mono = new Float32Array(audioBuffer.length)
  for (let index = 0; index < audioBuffer.length; index += 1) {
    let mix = 0
    for (const channel of channelData) mix += channel[index]
    mono[index] = mix / channelData.length
  }
  return mono
}

function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)

  const writeString = (offset, text) => {
    for (let index = 0; index < text.length; index += 1) {
      view.setUint8(offset + index, text.charCodeAt(index))
    }
  }

  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true)
  view.setUint16(20, 1, true)
  view.setUint16(22, 1, true)
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true)
  view.setUint16(32, 2, true)
  view.setUint16(34, 16, true)
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true)

  let offset = 44
  for (let index = 0; index < samples.length; index += 1) {
    const sample = Math.max(-1, Math.min(1, samples[index]))
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true)
    offset += 2
  }

  return new Blob([buffer], { type: 'audio/wav' })
}

function drawWaveform(canvas, data, color) {
  const context = canvas.getContext('2d')
  if (!context) return

  const ratio = window.devicePixelRatio || 1
  const rect = canvas.getBoundingClientRect()
  const width = rect.width || 420
  const height = rect.height || 170

  canvas.width = width * ratio
  canvas.height = height * ratio
  context.setTransform(ratio, 0, 0, ratio, 0, 0)
  context.clearRect(0, 0, width, height)

  const gradient = context.createLinearGradient(0, 0, width, 0)
  gradient.addColorStop(0, color)
  gradient.addColorStop(1, color === '#23c48e' ? '#7de3b6' : '#7bd9ff')

  context.strokeStyle = gradient
  context.lineWidth = 1.6
  context.shadowBlur = 10
  context.shadowColor = color
  context.beginPath()

  const samples = data || new Float32Array(1024)
  const targetBars = Math.min(220, Math.max(80, Math.floor(width / 3)))
  const step = Math.max(1, Math.floor(samples.length / targetBars))

  for (let index = 0; index < targetBars; index += 1) {
    const start = index * step
    const end = Math.min(samples.length, start + step)
    let peak = 0
    for (let offset = start; offset < end; offset += 1) {
      peak = Math.max(peak, Math.abs(samples[offset] || 0))
    }

    const x = (index / (targetBars - 1)) * width
    const y = height / 2
    const amplitude = peak * (height * 0.42)
    context.moveTo(x, y - amplitude)
    context.lineTo(x, y + amplitude)
  }

  context.stroke()
  context.shadowBlur = 0
}

function WaveformCard({ title, color, data, caption, accent }) {
  const canvasRef = useRef(null)

  useEffect(() => {
    if (!canvasRef.current) return
    drawWaveform(canvasRef.current, data, color)
  }, [data, color])

  return (
    <div className="denoise-wave-card">
      <div className="denoise-wave-header">
        <div>
          <span className="eyebrow small">{title}</span>
          <strong>{caption}</strong>
        </div>
        <span className={`signal-pill ${accent}`}>{accent === 'green' ? 'CLEAN' : 'INPUT'}</span>
      </div>
      <canvas ref={canvasRef} className="denoise-wave-canvas" />
    </div>
  )
}

function StatusBadge({ state }) {
  const config = {
    idle: { label: 'IDLE', className: 'status-idle' },
    processing: { label: 'PROCESSING', className: 'status-processing' },
    success: { label: 'SUCCESS', className: 'status-success' },
    error: { label: 'ERROR', className: 'status-error' },
  }
  const current = config[state] || config.idle

  return <span className={`status-badge ${current.className}`}>{current.label}</span>
}

export default function AudioDenoisingPage() {
  const [backendHealth, setBackendHealth] = useState({ status: 'checking', model_loaded: false })
  const [sourceFile, setSourceFile] = useState(null)
  const [originalUrl, setOriginalUrl] = useState('')
  const [denoisedUrl, setDenoisedUrl] = useState('')
  const [resultBlob, setResultBlob] = useState(null)
  const [status, setStatus] = useState('idle')
  const [message, setMessage] = useState('Upload or record audio to begin')
  const [error, setError] = useState('')
  const [isRecording, setIsRecording] = useState(false)
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [isProcessing, setIsProcessing] = useState(false)
  const [originalAudioBuffer, setOriginalAudioBuffer] = useState(null)
  const [originalSamples, setOriginalSamples] = useState(new Float32Array())
  const [denoisedSamples, setDenoisedSamples] = useState(new Float32Array())
  const [processingProgress, setProcessingProgress] = useState(0)
  const [fileInfo, setFileInfo] = useState(null)
  const [metrics, setMetrics] = useState({
    input: 'n/a',
    output: 'n/a',
    improvement: 'n/a',
    processingTime: 'n/a',
  })

  const inputRef = useRef(null)
  const mediaRecorderRef = useRef(null)
  const streamRef = useRef(null)
  const recordingIntervalRef = useRef(null)
  const audioContextRef = useRef(null)

  useEffect(() => {
    checkBackendHealth()
    return () => {
      if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current)
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop())
      }
      if (originalUrl) URL.revokeObjectURL(originalUrl)
      if (denoisedUrl) URL.revokeObjectURL(denoisedUrl)
    }
  }, [])

  useEffect(() => {
    if (!isRecording) return undefined
    recordingIntervalRef.current = setInterval(() => {
      setRecordingSeconds((prev) => prev + 1)
    }, 1000)

    return () => {
      if (recordingIntervalRef.current) clearInterval(recordingIntervalRef.current)
    }
  }, [isRecording])

  const checkBackendHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/api/health`)
      const payload = await response.json()
      setBackendHealth(payload)
      if (!payload.model_loaded) {
        setError(payload.error || 'The AI backend is not ready.')
        setStatus('error')
        setMessage('Model not loaded; check backend startup.')
      }
      return payload
    } catch (fetchError) {
      const offline = { status: 'offline', model_loaded: false, error: 'Backend unavailable' }
      setBackendHealth(offline)
      setError(`Backend unavailable. Start the FastAPI server on port ${API_BASE.split(':').pop() || '8001'} or set VITE_API_BASE.`)
      setStatus('error')
      setMessage('Backend unavailable')
      return offline
    }
  }

  const prepareUploadFile = async (file) => {
    if (!file) return null
    const fileName = (file.name || 'audio').replace(/\.[^.]+$/, '')
    const context = audioContextRef.current || new (window.AudioContext || window.webkitAudioContext)()
    audioContextRef.current = context

    const arrayBuffer = await file.arrayBuffer()
    const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0))
    const mono = toMonoSamples(audioBuffer)
    const wavBlob = encodeWav(mono, audioBuffer.sampleRate)
    return new File([wavBlob], `${fileName}.wav`, { type: 'audio/wav' })
  }

  const decodeAudioFile = async (file) => {
    const context = audioContextRef.current || new (window.AudioContext || window.webkitAudioContext)()
    audioContextRef.current = context
    const arrayBuffer = await file.arrayBuffer()
    const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0))
    const mono = toMonoSamples(audioBuffer)
    return { audioBuffer, mono }
  }

  const setAudioSource = async (file, label) => {
    if (!file) return
    const fileType = (file.type || '').toLowerCase()
    const accepted = ['audio/', 'application/octet-stream'].some((prefix) => fileType.startsWith(prefix)) || file.name.match(/\.(wav|mp3|flac|m4a|ogg|webm)$/i)
    if (!accepted) {
      setStatus('error')
      setError('Unsupported audio format. Please upload WAV, MP3, FLAC, M4A, OGG, or WebM.')
      return
    }
    if (file.size > 25 * 1024 * 1024) {
      setStatus('error')
      setError('Audio file is too large. Please use a smaller recording or upload.')
      return
    }

    try {
      setSourceFile(file)
      const { audioBuffer, mono } = await decodeAudioFile(file)
      const nextUrl = URL.createObjectURL(file)
      if (originalUrl) URL.revokeObjectURL(originalUrl)
      setOriginalUrl(nextUrl)
      setOriginalAudioBuffer(audioBuffer)
      setOriginalSamples(mono)
      setFileInfo({
        name: file.name || label,
        size: formatFileSize(file.size),
        duration: formatDuration(audioBuffer.duration),
        sampleRate: `${audioBuffer.sampleRate} Hz`,
        channels: audioBuffer.numberOfChannels === 1 ? 'Mono' : `${audioBuffer.numberOfChannels} channels`,
      })
      setStatus('idle')
      setMessage('Upload or record audio to begin')
      setError('')
    } catch (decodeError) {
      console.error(decodeError)
      setStatus('error')
      setError('The file could not be decoded as audio. Try another file or a different recording.')
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return
    await setAudioSource(file, file.name)
    event.target.value = ''
  }

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const recorder = new MediaRecorder(stream)
      const chunks = []

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) chunks.push(event.data)
      }

      recorder.onstop = async () => {
        const blob = new Blob(chunks, { type: recorder.mimeType || 'audio/webm' })
        const recordFile = new File([blob], 'microphone-recording.webm', { type: blob.type })
        stream.getTracks().forEach((track) => track.stop())
        streamRef.current = null
        setIsRecording(false)
        setRecordingSeconds(0)
        await setAudioSource(recordFile, 'microphone-recording.webm')
      }

      mediaRecorderRef.current = recorder
      recorder.start()
      setIsRecording(true)
      setStatus('idle')
      setMessage('Recording in progress')
      setError('')
    } catch (permissionError) {
      console.error(permissionError)
      setStatus('error')
      setError('Microphone permission was denied. Please allow access and try again.')
      setMessage('Permission required')
    }
  }

  const stopRecording = () => {
    if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') return
    mediaRecorderRef.current.stop()
  }

  const playAudioUrl = (url) => {
    if (!url) return
    const audio = new Audio(url)
    audio.play().catch(() => {
      setError('The browser blocked automatic playback. Use the player controls to listen.')
    })
  }

  const processAudio = async () => {
    if (!sourceFile) {
      setStatus('error')
      setError('No audio is selected yet.')
      return
    }

    if (!backendHealth.model_loaded) {
      const health = await checkBackendHealth()
      if (!health.model_loaded) {
        setStatus('error')
        setError('The backend is unavailable. Start FastAPI and ensure the model is loaded.')
        return
      }
    }

    const startedAt = Date.now()
    setIsProcessing(true)
    setStatus('processing')
    setMessage('AI is removing noise...')
    setError('')
    setProcessingProgress(15)

    const formData = new FormData()
    const uploadFile = await prepareUploadFile(sourceFile)
    formData.append('audio', uploadFile || sourceFile)

    try {
      const response = await fetch(`${API_BASE}/api/denoise`, {
        method: 'POST',
        body: formData,
      })

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}))
        throw new Error(payload.detail || 'The backend rejected the audio request.')
      }

      const blob = await response.blob()
      if (!blob || blob.size === 0) {
        throw new Error('The backend returned an empty response.')
      }

      if (denoisedUrl) URL.revokeObjectURL(denoisedUrl)
      const nextUrl = URL.createObjectURL(blob)
      setDenoisedUrl(nextUrl)
      setResultBlob(blob)

      const decoded = await decodeAudioFile(new File([blob], 'denoised.wav', { type: 'audio/wav' }))
      setDenoisedSamples(decoded.mono)
      setProcessingProgress(100)
      setStatus('success')
      setMessage('Audio successfully denoised')
      const elapsedSeconds = ((Date.now() - startedAt) / 1000).toFixed(1)
      setMetrics({
        input: 'n/a',
        output: 'n/a',
        improvement: 'Not available — no clean reference signal was supplied',
        processingTime: `${elapsedSeconds}s`,
      })
    } catch (processingError) {
      console.error(processingError)
      setStatus('error')
      setError(processingError.message || 'Unable to process audio with the trained model.')
      setMessage('Unable to process audio')
    } finally {
      setIsProcessing(false)
      setProcessingProgress(100)
    }
  }

  const clearAudio = () => {
    if (originalUrl) URL.revokeObjectURL(originalUrl)
    if (denoisedUrl) URL.revokeObjectURL(denoisedUrl)
    setOriginalUrl('')
    setDenoisedUrl('')
    setResultBlob(null)
    setSourceFile(null)
    setOriginalAudioBuffer(null)
    setOriginalSamples(new Float32Array())
    setDenoisedSamples(new Float32Array())
    setFileInfo(null)
    setStatus('idle')
    setMessage('Upload or record audio to begin')
    setError('')
    setProcessingProgress(0)
    setMetrics({ input: 'n/a', output: 'n/a', improvement: 'n/a', processingTime: 'n/a' })
  }

  return (
    <div className="ai-denoise-page">
      <section className="denoise-hero">
        <div>
          <span className="eyebrow"><span className="eyebrow-pulse" /> AI AUDIO DENOISING</span>
          <h1>Remove noise with the trained deep-learning model.</h1>
        </div>
        <div className="backend-indicator">
          <StatusBadge state={status === 'processing' ? 'processing' : status === 'success' ? 'success' : status === 'error' ? 'error' : 'idle'} />
          <span>{backendHealth.model_loaded ? 'Model ready' : 'Backend check pending'}</span>
        </div>
      </section>

      <section className="denoise-status-panel">
        <div className="status-copy">
          <div className={`status-dot ${status}`} />
          <div>
            <strong>{status === 'processing' ? 'AI is removing noise...' : status === 'success' ? 'Audio successfully denoised' : status === 'error' ? 'Unable to process audio' : 'Upload or record audio to begin'}</strong>
            <p>{error || message}</p>
          </div>
        </div>
        <div className="status-log-block">
          <span>MODEL</span>
          <b>{backendHealth.model_loaded ? 'best_model.pth loaded' : 'Awaiting backend model'}</b>
        </div>
      </section>

      <section className="denoise-controls-panel">
        <div className="record-card">
          <div className="icon-box"><span>🎙</span></div>
          <h3>Microphone</h3>
          {!isRecording ? (
            <button className="primary-button" onClick={startRecording}>Start Recording</button>
          ) : (
            <>
              <button className="secondary-button danger" onClick={stopRecording}>Stop Recording</button>
              <span className="recording-time">Recording... {formatDuration(recordingSeconds)}</span>
            </>
          )}
        </div>

        <div className="upload-card" onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); const file = event.dataTransfer.files?.[0]; if (file) setAudioSource(file, file.name); }}>
          <div className="icon-box"><span>📁</span></div>
          <h3>Upload Audio</h3>
          <p>Drop audio here or choose a file</p>
          <button className="primary-button" onClick={() => inputRef.current?.click()}>Choose Audio File</button>
          <input ref={inputRef} type="file" accept="audio/*" hidden onChange={handleFileUpload} />
          <small>WAV • MP3 • FLAC • M4A • OGG • WebM</small>
        </div>
      </section>

      {fileInfo && (
        <section className="denoise-file-info">
          <div>
            <span>File</span>
            <strong>{fileInfo.name}</strong>
          </div>
          <div>
            <span>Size</span>
            <strong>{fileInfo.size}</strong>
          </div>
          <div>
            <span>Duration</span>
            <strong>{fileInfo.duration}</strong>
          </div>
          <div>
            <span>Sample rate</span>
            <strong>{fileInfo.sampleRate}</strong>
          </div>
          <div>
            <span>Channels</span>
            <strong>{fileInfo.channels}</strong>
          </div>
        </section>
      )}

      <section className="denoise-processing-panel">
        <div className="timeline">
          <span>NOISY AUDIO</span>
          <span className="arrow">↓</span>
          <span>ANALYZE</span>
          <span className="arrow">↓</span>
          <span>AI DENOISING</span>
          <span className="arrow">↓</span>
          <span>CLEAN AUDIO</span>
        </div>
        <div className="processing-progress">
          <span style={{ width: `${processingProgress}%` }} />
        </div>
      </section>

      <section className="denoise-result-grid">
        <WaveformCard title="Original" color="#26b7ff" data={originalSamples} caption={fileInfo ? fileInfo.name : 'No selection'} accent="blue" />
        <WaveformCard title="Denoised" color="#23c48e" data={denoisedSamples} caption={denoisedUrl ? 'AI output' : 'Waiting for model output'} accent="green" />
      </section>

      <section className="denoise-actions-row">
        <button className="primary-button" onClick={() => playAudioUrl(originalUrl)} disabled={!originalUrl}>▶ Play Original</button>
        <button className="primary-button accent" onClick={processAudio} disabled={!sourceFile || isProcessing || !backendHealth.model_loaded}>✨ Clean Audio with AI</button>
        <button className="secondary-button" onClick={clearAudio}>🗑 Clear</button>
      </section>

      {originalUrl && (
        <section className="denoise-players">
          <div className="player-card">
            <h3>Original Audio</h3>
            <audio controls src={originalUrl} />
          </div>
          {denoisedUrl && (
            <div className="player-card">
              <h3>AI Denoised Audio</h3>
              <audio controls src={denoisedUrl} />
              <a className="download-button" href={denoisedUrl} download="denoised_audio.wav">Download Denoised Audio</a>
            </div>
          )}
        </section>
      )}

      <section className="metrics-panel">
        <div className="metrics-header">
          <span className="eyebrow small">METRICS</span>
          <h3>Processing information</h3>
        </div>
        <div className="metric-grid">
          <div>
            <span>Input SNR</span>
            <strong>{metrics.input}</strong>
          </div>
          <div>
            <span>Output SNR</span>
            <strong>{metrics.output}</strong>
          </div>
          <div>
            <span>Improvement</span>
            <strong>{metrics.improvement}</strong>
          </div>
          <div>
            <span>Processing time</span>
            <strong>{metrics.processingTime}</strong>
          </div>
        </div>
      </section>
    </div>
  )
}
