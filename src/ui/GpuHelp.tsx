import { useState, type ReactNode } from 'react'
import type { BrowserName } from '../engine/capabilities'

const CHROMIUM = new Set<BrowserName>(['chrome', 'edge', 'opera', 'samsung'])

function CopyRow({ label, value }: { label: string; value: string }) {
  const [copied, setCopied] = useState(false)
  const copy = async (): Promise<void> => {
    try {
      await navigator.clipboard.writeText(value)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      setCopied(false)
    }
  }
  return (
    <div className="copy-row">
      <code className="mono" title={label}>
        {value}
      </code>
      <button type="button" className="btn ghost tiny" onClick={() => void copy()}>
        {copied ? 'Copied' : 'Copy'}
      </button>
    </div>
  )
}

function chromiumSteps(): ReactNode {
  return (
    <>
      <p className="muted small">
        On desktop, WebGPU is on by default (Chrome/Edge 113+). On Android it's enabled only for
        Qualcomm/ARM GPUs; other GPUs (Exynos/Xclipse, AMD, some Samsung Internet) are blocked. To
        force-enable it, open the two flags below, set both to <strong>Enabled</strong>, then tap
        <strong> Relaunch</strong>. (Browsers can't open <span className="mono">chrome://</span> links
        from a web page, so copy these into the address bar.)
      </p>
      <CopyRow label="unsafe webgpu" value="chrome://flags/#enable-unsafe-webgpu" />
      <CopyRow label="ignore gpu blocklist" value="chrome://flags/#ignore-gpu-blocklist" />
      <p className="muted small">
        Samsung Internet may not expose WebGPU at all — use <strong>Chrome</strong> there.
      </p>
    </>
  )
}

function firefoxSteps(): ReactNode {
  return (
    <>
      <p className="muted small">
        Desktop Firefox has WebGPU on by default on recent versions; if not, open{' '}
        <span className="mono">about:config</span> and set <span className="mono">dom.webgpu.enabled</span>{' '}
        to <strong>true</strong>. On Android, use <strong>Firefox Nightly</strong> with the same pref.
      </p>
      <CopyRow label="config" value="about:config" />
      <CopyRow label="pref" value="dom.webgpu.enabled" />
    </>
  )
}

function safariSteps(): ReactNode {
  return (
    <p className="muted small">
      Safari exposes WebGPU on <strong>macOS 15 (Sequoia)</strong> and <strong>iOS/iPadOS 18+</strong>,
      on by default — no flags. On older versions, update the OS.
    </p>
  )
}

export function GpuHelp({ browser }: { browser: BrowserName }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="gpu-help">
      <button type="button" className="btn ghost tiny" onClick={() => setOpen((v) => !v)}>
        {open ? 'Hide' : 'How to enable WebGPU'}
      </button>
      {open && (
        <div className="gpu-help-body">
          {CHROMIUM.has(browser) && chromiumSteps()}
          {browser === 'firefox' && firefoxSteps()}
          {browser === 'safari' && safariSteps()}
          {browser === 'other' && (
            <p className="muted small">Use Chrome, Edge or Safari 18+ for on-device WebGPU.</p>
          )}
          <p className="muted small">
            Verify your device at{' '}
            <a className="link" href="https://webgpureport.org" target="_blank" rel="noreferrer">
              webgpureport.org
            </a>
            . If WebGPU can't be enabled, the app still runs on CPU (slower, lower quality).
          </p>
        </div>
      )}
    </div>
  )
}
