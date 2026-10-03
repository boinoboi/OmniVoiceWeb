import type { ModelFile, Profile } from './manifest'
import { filesForProfile } from './manifest'

export const MODEL_CACHE_NAME = 'omnivoice-models-v1'

const supportsCache = (): boolean =>
  typeof caches !== 'undefined' && typeof location !== 'undefined'

const cacheKey = (path: string): string =>
  `${location.origin}/__omnivoice_models__/${path}`

export const isCacheSupported = supportsCache

export async function openModelCache(): Promise<Cache> {
  return caches.open(MODEL_CACHE_NAME)
}

export async function getModelBlob(path: string): Promise<Blob | undefined> {
  if (!supportsCache()) return undefined
  const cache = await openModelCache()
  const res = await cache.match(cacheKey(path))
  return res ? res.blob() : undefined
}

export async function isCached(path: string): Promise<boolean> {
  if (!supportsCache()) return false
  const cache = await openModelCache()
  return Boolean(await cache.match(cacheKey(path)))
}

export interface CacheStatus {
  file: ModelFile
  cached: boolean
  bytes: number
}

export async function inspectCache(profile: Profile): Promise<CacheStatus[]> {
  const files = filesForProfile(profile)
  if (!supportsCache()) return files.map((file) => ({ file, cached: false, bytes: file.bytes }))
  const cache = await openModelCache()
  const out: CacheStatus[] = []
  for (const file of files) {
    const hit = await cache.match(cacheKey(file.path))
    out.push({ file, cached: Boolean(hit), bytes: file.bytes })
  }
  return out
}

export async function clearModelCache(): Promise<void> {
  if (!supportsCache()) return
  await caches.delete(MODEL_CACHE_NAME)
}

const CACHE_PREFIX = (): string => `${location.origin}/__omnivoice_models__/`

export function filesToPrune(profile: Profile, cachedPaths: Iterable<string>): string[] {
  const keep = new Set(filesForProfile(profile).map((file) => file.path))
  const victims: string[] = []
  for (const path of cachedPaths) if (!keep.has(path)) victims.push(path)
  return victims
}

export async function listCachedPaths(): Promise<string[]> {
  if (!supportsCache()) return []
  const cache = await openModelCache()
  const prefix = CACHE_PREFIX()
  const keys = await cache.keys()
  return keys
    .map((key) => key.url)
    .filter((url) => url.startsWith(prefix))
    .map((url) => url.slice(prefix.length))
}

export async function deleteCached(path: string): Promise<void> {
  if (!supportsCache()) return
  const cache = await openModelCache()
  await cache.delete(cacheKey(path))
}

export async function pruneCache(profile: Profile): Promise<string[]> {
  if (!supportsCache()) return []
  const victims = filesToPrune(profile, await listCachedPaths())
  for (const path of victims) await deleteCached(path)
  return victims
}

export async function cachedBytes(): Promise<number> {
  if (!supportsCache()) return 0
  const cache = await openModelCache()
  const keys = await cache.keys()
  let total = 0
  for (const key of keys) {
    const response = await cache.match(key)
    if (!response) continue
    const header = Number(response.headers.get('content-length'))
    total += Number.isFinite(header) && header > 0 ? header : (await response.clone().blob()).size
  }
  return total
}

async function putBlob(path: string, blob: Blob): Promise<void> {
  const cache = await openModelCache()
  await cache.put(
    cacheKey(path),
    new Response(blob, {
      headers: {
        'content-type': blob.type || 'application/octet-stream',
        'content-length': String(blob.size),
      },
    }),
  )
}

export interface DownloadUpdate {
  file: ModelFile
  fileLoaded: number
  fileTotal: number
  overallLoaded: number
  overallTotal: number
  phase: 'start' | 'progress' | 'file-done' | 'skipped' | 'done'
}

export interface DownloadOptions {
  concurrency?: number
  signal?: AbortSignal
  onUpdate?: (update: DownloadUpdate) => void
}

export class ModelDownloader {
  private loadedByPath = new Map<string, number>()
  private readonly options: DownloadOptions

  constructor(options: DownloadOptions = {}) {
    this.options = options
  }

  async run(profile: Profile): Promise<void> {
    await this.runFiles(filesForProfile(profile))
  }

  async runFiles(files: ModelFile[]): Promise<void> {
    const total = files.reduce((sum, f) => sum + f.bytes, 0)
    const concurrency = Math.max(1, this.options.concurrency ?? 2)
    let cursor = 0
    const emit = (u: DownloadUpdate) => this.options.onUpdate?.(u)

    for (const file of files) {
      if (await isCached(file.path)) {
        this.loadedByPath.set(file.path, file.bytes)
        emit({
          file,
          fileLoaded: file.bytes,
          fileTotal: file.bytes,
          overallLoaded: this.overall(total),
          overallTotal: total,
          phase: 'skipped',
        })
      }
    }

    const worker = async (): Promise<void> => {
      while (cursor < files.length) {
        const file = files[cursor++]
        if (this.options.signal?.aborted) throw new DOMException('Aborted', 'AbortError')
        await this.fetchOne(file, total, emit)
      }
    }

    await Promise.all(Array.from({ length: concurrency }, worker))
    emit({
      file: files[files.length - 1],
      fileLoaded: 0,
      fileTotal: 0,
      overallLoaded: total,
      overallTotal: total,
      phase: 'done',
    })
  }

  private overall(total: number): number {
    let sum = 0
    for (const value of this.loadedByPath.values()) sum += value
    return Math.min(sum, total)
  }

  private async fetchOne(
    file: ModelFile,
    total: number,
    emit: (u: DownloadUpdate) => void,
  ): Promise<void> {
    if (await isCached(file.path)) return
    const res = await fetch(file.url, { signal: this.options.signal })
    if (!res.ok) throw new Error(`HTTP ${res.status} when downloading ${file.path}`)
    const headerTotal = Number(res.headers.get('content-length'))
    const fileTotal = Number.isFinite(headerTotal) && headerTotal > 0 ? headerTotal : file.bytes

    const chunks: BlobPart[] = []
    let fileLoaded = 0
    this.loadedByPath.set(file.path, 0)
    emit({
      file,
      fileLoaded: 0,
      fileTotal,
      overallLoaded: this.overall(total),
      overallTotal: total,
      phase: 'start',
    })

    const reader = res.body?.getReader()
    if (reader) {
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        if (value) {
          chunks.push(value as unknown as BlobPart)
          fileLoaded += value.byteLength
          this.loadedByPath.set(file.path, fileLoaded)
          emit({
            file,
            fileLoaded,
            fileTotal,
            overallLoaded: this.overall(total),
            overallTotal: total,
            phase: 'progress',
          })
        }
      }
    } else {
      const buffer = await res.arrayBuffer()
      chunks.push(buffer)
      fileLoaded = buffer.byteLength
    }

    const blob = new Blob(chunks, { type: 'application/octet-stream' })
    await putBlob(file.path, blob)
    this.loadedByPath.set(file.path, file.bytes)
    emit({
      file,
      fileLoaded: file.bytes,
      fileTotal,
      overallLoaded: this.overall(total),
      overallTotal: total,
      phase: 'file-done',
    })
  }
}

export async function ensureCached(
  files: ModelFile[],
  options: DownloadOptions = {},
): Promise<void> {
  await new ModelDownloader(options).runFiles(files)
}

export interface StorageEstimate {
  usage: number
  quota: number
}

export async function estimateStorage(): Promise<StorageEstimate | undefined> {
  if (typeof navigator === 'undefined' || !navigator.storage?.estimate) return undefined
  const { usage = 0, quota = 0 } = await navigator.storage.estimate()
  return { usage, quota }
}
