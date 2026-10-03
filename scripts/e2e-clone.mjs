import { existsSync, readFileSync, writeFileSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const url = process.argv[2] ?? 'http://localhost:4173/'
const executablePath = process.env.CHROME_PATH ?? '/usr/bin/google-chrome-stable'
const timeoutMs = Number(process.env.E2E_TIMEOUT ?? 1_800_000)
const refPath = process.env.CLONE_REF ?? 'tools/refs/reference.wav'
const refText = process.env.CLONE_REF_TEXT_FILE
  ? readFileSync(process.env.CLONE_REF_TEXT_FILE, 'utf8').trim()
  : 'Hello there, this is a short cloned voice test.'
const text = process.env.CLONE_TEXT ?? 'Hello there, this is a short cloned voice test.'
const outPath = process.env.CLONE_OUT ?? 'tools/golden/browser_clone.wav'

if (!existsSync(refPath)) throw new Error(`reference not found: ${refPath}`)

const browser = await puppeteer.launch({
  executablePath,
  headless: process.env.E2E_HEADFUL !== '1',
  protocolTimeout: timeoutMs + 120_000,
  userDataDir: process.env.E2E_PROFILE,
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
})
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 1000 })

const errors = []
const BENIGN = ['VerifyEachNodeIsAssignedToAnEp', 'not assigned to the preferred execution providers', 'Rerunning with verbose']
page.on('pageerror', (e) => {
  errors.push(`pageerror: ${e.message}`)
  console.log('PAGEERROR', e.message)
})
page.on('console', (m) => {
  if (m.type() !== 'error') return
  const t = m.text()
  if (!BENIGN.some((n) => t.includes(n))) console.log('CONSOLE ERR', t)
})

console.log(`loading ${url}`)
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })

const adapter = await page.evaluate(async () => {
  if (!navigator.gpu) return 'no navigator.gpu'
  const a = await navigator.gpu.requestAdapter()
  return a ? `${a.info?.vendor} | ${a.info?.architecture}` : 'no adapter'
})
console.log('webgpu adapter:', adapter)
if (adapter.startsWith('no ')) {
  console.error('E2E FAILED: no WebGPU adapter')
  await browser.close()
  process.exit(1)
}

await page.waitForSelector('.textarea', { timeout: 30_000 })
await page.click('.textarea')
await page.keyboard.down('Control')
await page.keyboard.press('KeyA')
await page.keyboard.up('Control')
await page.keyboard.press('Backspace')
await page.type('.textarea', text)

await page.click('.cloner summary')

if (process.env.CLONE_VOICE) {
  console.log('selecting precomputed voice:', process.env.CLONE_VOICE)
  const clicked = await page.evaluate((name) => {
    const chip = [...document.querySelectorAll('.cloner .chip')].find((c) =>
      c.textContent?.includes(name),
    )
    chip?.click()
    return Boolean(chip)
  }, process.env.CLONE_VOICE)
  if (!clicked) throw new Error(`voice chip "${process.env.CLONE_VOICE}" not found`)
} else {
  console.log('typing transcript + uploading reference', refPath)
  await page.click('.cloner input.textarea')
  await page.type('.cloner input.textarea', refText)
  await page.click('.consent input[type="checkbox"]')
  const fileInput = await page.waitForSelector('.cloner input[type="file"]', { timeout: 15_000 })
  await fileInput.uploadFile(refPath)
}

const poll = setInterval(async () => {
  try {
    const status = await page.evaluate(() => {
      const synth = [...document.querySelectorAll('.card')].find(
        (c) => c.querySelector('h2')?.textContent === 'Synthesize',
      )
      const busy = synth?.querySelector('.actions .muted')?.textContent ?? ''
      const cloner = synth?.querySelectorAll('.cloner .muted')
      const last = cloner?.[cloner.length - 1]?.textContent ?? ''
      const note = synth?.querySelector('.notice')?.textContent ?? ''
      return `${busy || last} ${note}`.trim()
    })
    console.log('  …', status)
  } catch {
    /* busy */
  }
}, 10_000)

console.log('waiting for reference encoding…')
await page.waitForFunction(
  () => (document.querySelector('.cloner')?.textContent ?? '').includes('Active voice:'),
  { timeout: timeoutMs, polling: 2000 },
)
clearInterval(poll)
console.log('reference encoded; generating')
console.log(
  await page.evaluate(() => {
    const synth = [...document.querySelectorAll('.card')].find(
      (c) => c.querySelector('h2')?.textContent === 'Synthesize',
    )
    return synth?.querySelector('.cloner p.muted')?.textContent ?? ''
  }),
)

await page.evaluate(() => {
  const buttons = [...document.querySelectorAll('button.btn.primary')]
  buttons.find((b) => b.textContent?.includes('Generate'))?.click()
})

const pollGen = setInterval(async () => {
  try {
    const stage = await page.evaluate(() => {
      const synth = [...document.querySelectorAll('.card')].find(
        (c) => c.querySelector('h2')?.textContent === 'Synthesize',
      )
      const bar = synth?.querySelector('.progress span')
      const busy = synth?.querySelector('.actions .muted')?.textContent ?? ''
      const note = synth?.querySelector('.notice')?.textContent ?? ''
      return `bar=${bar?.getAttribute('style') ?? '?'} busy="${busy}" ${note}`
    })
    console.log('  … gen', stage)
  } catch {
    /* busy */
  }
}, 10_000)

await page.waitForSelector('audio[src^="blob:"]', { timeout: timeoutMs })
clearInterval(pollGen)

const stats = await page.evaluate(async () => {
  const audio = document.querySelector('audio')
  const buffer = await (await fetch(audio.src)).arrayBuffer()
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  return { bytes: buffer.byteLength, base64: btoa(binary) }
})
writeFileSync(outPath, Buffer.from(stats.base64, 'base64'))
console.log(`saved ${outPath} (${stats.bytes} bytes, ~${((stats.bytes - 44) / 2 / 24000).toFixed(2)}s)`)

await browser.close()
if (stats.bytes < 40_000) {
  console.error('E2E FAILED: audio too short')
  process.exit(1)
}
if (errors.length) {
  console.error('E2E FAILED: page errors')
  process.exit(1)
}
console.log('E2E PASSED: browser cloning produced audio')
