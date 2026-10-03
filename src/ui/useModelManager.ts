import { useCallback, useEffect, useRef, useState } from 'react'
import {
  ModelDownloader,
  clearModelCache,
  inspectCache,
  isCacheSupported,
  pruneCache,
  type CacheStatus,
  type DownloadUpdate,
} from '../engine/cache'
import { bytesForDevice, type Backend, type Profile } from '../engine/manifest'

export interface OverallProgress {
  loaded: number
  total: number
}

export function useModelManager(backend: Backend = 'webgpu') {
  const [profile, setProfile] = useState<Profile>('lite')
  const [statuses, setStatuses] = useState<CacheStatus[]>([])
  const [progress, setProgress] = useState<OverallProgress>({
    loaded: 0,
    total: bytesForDevice(backend, 'lite'),
  })
  const [downloading, setDownloading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const applyStatuses = useCallback((next: CacheStatus[], total: number) => {
    setStatuses(next)
    setProgress({
      loaded: next.filter((s) => s.cached).reduce((sum, s) => sum + s.bytes, 0),
      total,
    })
  }, [])

  const refresh = useCallback(async () => {
    const next = await inspectCache(profile, backend)
    applyStatuses(next, bytesForDevice(backend, profile))
  }, [profile, backend, applyStatuses])

  useEffect(() => {
    let active = true
    void inspectCache(profile, backend).then((next) => {
      if (active) applyStatuses(next, bytesForDevice(backend, profile))
    })
    return () => {
      active = false
    }
  }, [profile, backend, applyStatuses])

  const start = useCallback(async () => {
    setError(null)
    setDownloading(true)
    const controller = new AbortController()
    abortRef.current = controller
    const downloader = new ModelDownloader({
      concurrency: 2,
      signal: controller.signal,
      onUpdate: (update: DownloadUpdate) =>
        setProgress({ loaded: update.overallLoaded, total: update.overallTotal }),
    })
    try {
      await downloader.run(profile, backend)
    } catch (e) {
      if (!(e instanceof DOMException && e.name === 'AbortError')) {
        setError(e instanceof Error ? e.message : String(e))
      }
    } finally {
      setDownloading(false)
      abortRef.current = null
      await refresh()
    }
  }, [profile, backend, refresh])

  const cancel = useCallback(() => abortRef.current?.abort(), [])

  const clear = useCallback(async () => {
    await clearModelCache()
    await refresh()
  }, [refresh])

  const prune = useCallback(async () => {
    await pruneCache(profile)
    await refresh()
  }, [profile, refresh])

  const allReady = statuses.length > 0 && statuses.every((s) => s.cached)

  return {
    profile,
    setProfile,
    statuses,
    progress,
    downloading,
    error,
    start,
    cancel,
    clear,
    prune,
    allReady,
    cacheSupported: isCacheSupported(),
    refresh,
  }
}
