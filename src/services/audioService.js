export function audioInfo(buffer, file) {
  return {
    duration: buffer.duration,
    sampleRate: buffer.sampleRate,
    channels: buffer.numberOfChannels,
    size: file?.size ?? 0,
  }
}

export function decodeAudioFile(file, context = new AudioContext()) {
  return file.arrayBuffer().then((data) => context.decodeAudioData(data))
}

export function createPlayableBuffer(context, samples, sampleRate) {
  const buffer = context.createBuffer(1, samples.length, sampleRate)
  buffer.copyToChannel(samples, 0)
  return buffer
}

export function playBuffer(buffer, context = new AudioContext()) {
  const source = context.createBufferSource()
  source.buffer = buffer
  source.connect(context.destination)
  source.start()
  return source
}

export async function startMicrophone(onStream) {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  onStream(stream)
  return stream
}