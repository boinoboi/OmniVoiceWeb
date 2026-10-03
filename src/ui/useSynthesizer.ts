import { useCallback, useEffect, useRef, useState } from 'react'
import { concatFloat32, encodeWav } from '../engine/audio'
import type { Device } from '../engine/ort'

export type SynthPhase = 'idle' | 'loading' | 'ready' | 'generating' | 'error'

export interface VoiceReference {
  data: Int32Array
  frames: number
  text?: string
  name?: string
}

export interface SynthView {
  phase: SynthPhase
  stage: string
  ratio: number
  detail: string
  audioUrl: string | null
  frames: number
  milliseconds: number
  error: string | null
  notice: string | null
}

const initial: SynthView = {
  phase: 'idle',
  stage: '',
  ratio: 0,
  detail: '',
  audioUrl: null,
  frames: 0,
  milliseconds: 0,
  error: null,
  notice: null,
}

interface WorkerMessage {
  type: string
  progress?: { stage: string; ratio: number; detail?: string }
  message?: string
  samples?: Float32Array
  sampleRate?: number
  frames?: number
  milliseconds?: number
  index?: number
  total?: number
  codes?: Int32Array
}

interface Pending {
  resolve: (value: WorkerMessage | 'loaded') => void
  reject: (error: Error) => void
}

export function useSynthesizer(device: Device) {
  const workerRef = useRef<Worker | null>(null)
  const pendingRef = useRef<Pending | null>(null)
  const urlRef = useRef<string | null>(null)
  const chunksRef = useRef<Float32Array[]>([])
  const referenceRef = useRef<VoiceReference | null>(null)
  const [reference, setReferenceState] = useState<VoiceReference | null>(null)
  const [view, setView] = useState<SynthView>(initial)

  const publish = useCallback((parts: Float32Array[], sampleRate: number): string => {
    const wav = encodeWav(concatFloat32(parts), sampleRate)
    const url = URL.createObjectURL(new Blob([wav], { type: 'audio/wav' }))
    if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    urlRef.current = url
    return url
  }, [])

  const getWorker = useCallback((): Worker => {
    if (!workerRef.current) {
      const worker = new Worker(new URL('../worker/inference.worker.ts', import.meta.url), {
        type: 'module',
      })
      worker.onerror = (event) => {
        const message = `worker error: ${event.message || 'failed to load'}`
        setView((v) => ({ ...v, phase: 'error', error: message }))
        pendingRef.current?.reject(new Error(message))
      }
      worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
        const message = event.data
        if (message.type === 'progress' && message.progress) {
          setView((v) => ({
            ...v,
            stage: message.progress!.stage,
            ratio: message.progress!.ratio,
            detail: message.progress!.detail ?? '',
          }))
        } else if (message.type === 'chunk') {
          if (message.samples) chunksRef.current.push(message.samples)
          const url = publish(chunksRef.current, message.sampleRate ?? 24000)
          setView((v) => ({
            ...v,
            phase: 'generating',
            audioUrl: url,
            frames: v.frames + (message.frames ?? 0),
            detail:
              message.index !== undefined && message.total !== undefined
                ? `part ${message.index + 1}/${message.total}`
                : v.detail,
          }))
        } else if (message.type === 'notice') {
          setView((v) => ({ ...v, notice: message.message ?? null }))
        } else if (message.type === 'loaded') {
          pendingRef.current?.resolve('loaded')
        } else if (message.type === 'result') {
          pendingRef.current?.resolve(message)
        } else if (message.type === 'error') {
          pendingRef.current?.reject(new Error(message.message ?? 'Worker error'))
        }
      }
      workerRef.current = worker
    }
    return workerRef.current
  }, [publish])

  const send = useCallback(
    (message: unknown): Promise<WorkerMessage | 'loaded'> => {
      const worker = getWorker()
      return new Promise((resolve, reject) => {
        pendingRef.current = { resolve, reject }
        worker.postMessage(message)
      })
    },
    [getWorker],
  )

  const generate = useCallback(
    async (text: string) => {
      if (!text.trim()) return
      chunksRef.current = []
      setView((v) => ({ ...v, phase: 'loading', error: null, notice: null, ratio: 0, detail: '' }))
      try {
        await send({ type: 'load', device })
        setView((v) => ({ ...v, phase: 'generating', stage: 'generate', ratio: 0, detail: '' }))
        const ref = referenceRef.current
        const result = (await send({
          type: 'generate',
          text,
          refCodes: ref?.data,
          refFrames: ref?.frames,
          refText: ref?.text,
        })) as WorkerMessage
        const samples = result.samples ?? concatFloat32(chunksRef.current)
        if (!samples || samples.length === 0) throw new Error('No audio returned')
        const url = publish([samples], result.sampleRate ?? 24000)
        setView((v) => ({
          ...v,
          phase: 'ready',
          audioUrl: url,
          frames: result.frames ?? v.frames,
          milliseconds: result.milliseconds ?? 0,
          ratio: 1,
        }))
      } catch (error) {
        setView((v) => ({
          ...v,
          phase: 'error',
          error: error instanceof Error ? error.message : String(error),
        }))
      }
    },
    [device, send, publish],
  )

  const encodeReference = useCallback(
    async (samples: Float32Array, name?: string, text?: string): Promise<VoiceReference> => {
      setView((v) => ({
        ...v,
        phase: 'loading',
        stage: 'clone',
        ratio: 0,
        detail: 'encoding reference…',
        error: null,
      }))
      try {
        const result = (await send({ type: 'encode', samples, device })) as WorkerMessage
        if (!result.codes) throw new Error('Reference encoding returned no codes')
        const ref: VoiceReference = {
          data: result.codes,
          frames: result.frames ?? 0,
          text: text?.trim() || undefined,
          name,
        }
        referenceRef.current = ref
        setReferenceState(ref)
        setView((v) => ({ ...v, phase: 'ready', stage: '', ratio: 1, detail: '' }))
        return ref
      } catch (error) {
        setView((v) => ({
          ...v,
          phase: 'error',
          error: error instanceof Error ? error.message : String(error),
        }))
        throw error
      }
    },
    [device, send],
  )

  const clearReference = useCallback(() => {
    referenceRef.current = null
    setReferenceState(null)
  }, [])

  useEffect(() => {
    return () => {
      workerRef.current?.terminate()
      if (urlRef.current) URL.revokeObjectURL(urlRef.current)
    }
  }, [])

  return { view, generate, encodeReference, reference, clearReference }
}
