export function downsample(samples, points = 220) {
  if (!samples?.length) return []
  const step = Math.max(1, Math.floor(samples.length / points))
  return Array.from({ length: Math.min(points, Math.ceil(samples.length / step)) }, (_, index) => {
    const start = index * step
    const end = Math.min(samples.length, start + step)
    let peak = 0
    for (let cursor = start; cursor < end; cursor += 1) peak = Math.max(peak, Math.abs(samples[cursor]))
    return peak
  })
}

export function spectrumFromSamples(samples, bars = 28) {
  const values = downsample(samples, bars).map((value, index) => value * (1 - index / (bars * 2)))
  return values.length ? values : Array.from({ length: bars }, () => 0.2)
}

export function formatDuration(seconds) {
  if (!Number.isFinite(seconds)) return '00:00'
  const minutes = Math.floor(seconds / 60).toString().padStart(2, '0')
  const remainder = Math.floor(seconds % 60).toString().padStart(2, '0')
  return `${minutes}:${remainder}`
}