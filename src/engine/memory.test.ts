import { afterEach, describe, expect, it, vi } from 'vitest'
import { IdleReleaser } from './memory'

describe('IdleReleaser', () => {
  afterEach(() => {
    vi.useRealTimers()
  })

  it('releases after the idle timeout', async () => {
    vi.useFakeTimers()
    const onRelease = vi.fn()
    const releaser = new IdleReleaser({ timeoutMs: 1000, onRelease })
    releaser.touch()
    expect(releaser.active).toBe(true)
    await vi.advanceTimersByTimeAsync(999)
    expect(onRelease).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(1)
    expect(onRelease).toHaveBeenCalledTimes(1)
    expect(releaser.active).toBe(false)
  })

  it('resets the timer on touch', async () => {
    vi.useFakeTimers()
    const onRelease = vi.fn()
    const releaser = new IdleReleaser({ timeoutMs: 1000, onRelease })
    releaser.touch()
    await vi.advanceTimersByTimeAsync(800)
    releaser.touch()
    await vi.advanceTimersByTimeAsync(800)
    expect(onRelease).not.toHaveBeenCalled()
    await vi.advanceTimersByTimeAsync(200)
    expect(onRelease).toHaveBeenCalledTimes(1)
  })

  it('cancel prevents release', async () => {
    vi.useFakeTimers()
    const onRelease = vi.fn()
    const releaser = new IdleReleaser({ timeoutMs: 1000, onRelease })
    releaser.touch()
    releaser.cancel()
    await vi.advanceTimersByTimeAsync(5000)
    expect(onRelease).not.toHaveBeenCalled()
  })
})
