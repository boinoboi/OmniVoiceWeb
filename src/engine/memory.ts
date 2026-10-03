export interface IdleReleaserOptions {
  timeoutMs: number
  onRelease: () => Promise<void> | void
}

export class IdleReleaser {
  private timer: ReturnType<typeof setTimeout> | undefined
  private readonly timeoutMs: number
  private readonly onRelease: () => Promise<void> | void
  private releasing = false

  constructor(options: IdleReleaserOptions) {
    this.timeoutMs = Math.max(0, options.timeoutMs)
    this.onRelease = options.onRelease
  }

  touch(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = setTimeout(() => {
      this.timer = undefined
      void this.release()
    }, this.timeoutMs)
  }

  private async release(): Promise<void> {
    if (this.releasing) return
    this.releasing = true
    try {
      await this.onRelease()
    } finally {
      this.releasing = false
    }
  }

  cancel(): void {
    if (this.timer) clearTimeout(this.timer)
    this.timer = undefined
  }

  get active(): boolean {
    return this.timer !== undefined
  }
}
