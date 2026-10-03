import { writeFileSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const url = process.argv[2] ?? 'http://localhost:4173/'
const executablePath = process.env.CHROME_PATH ?? '/usr/bin/google-chrome-stable'
const timeoutMs = Number(process.env.E2E_TIMEOUT ?? 1_200_000)
const useGpu = process.env.E2E_GPU === '1'
const useRealGpu = process.env.E2E_REALGPU === '1'
const outPath = process.env.E2E_OUT
const testText = process.env.E2E_TEXT ?? 'Hello world.'

const args = useRealGpu
  ? ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan']
  : useGpu
    ? ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=WebGPU', '--use-webgpu-adapter=swiftshader']
    : ['--no-sandbox', '--disable-gpu']

const browser = await puppeteer.launch({
  executablePath,
  headless: true,
  protocolTimeout: timeoutMs + 120_000,
  args,
})
const page = await browser.newPage()
await page.setViewport({ width: 1280, height: 900 })

const errors = []
page.on('pageerror', (e) => {
  errors.push(`pageerror: ${e.message}`)
  console.log('PAGEERROR', e.message)
})
page.on('console', (m) => {
  if (m.type() === 'error') {
    errors.push(`console: ${m.text()}`)
    console.log('CONSOLE ERR', m.text())
  }
})

console.log(`loading ${url}`)
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })

await page.waitForSelector('.textarea', { timeout: 30_000 })
await page.click('.textarea')
await page.keyboard.down('Control')
await page.keyboard.press('KeyA')
await page.keyboard.up('Control')
await page.keyboard.press('Backspace')
await page.type('.textarea', testText)

console.log('clicking Generate (downloads models, then runs the pipeline)')
await page.evaluate(() => {
  const buttons = [...document.querySelectorAll('button.btn.primary')]
  const generate = buttons.find((b) => b.textContent?.includes('Generate'))
  generate?.click()
})

const poll = setInterval(async () => {
  try {
    const stage = await page.evaluate(() => {
      const progress = document.querySelector('.card .progress span')
      const status = [...document.querySelectorAll('.actions .muted')].map((e) => e.textContent).join(' ')
      return `bar=${progress?.getAttribute('style') ?? '?'} ${status}`
    })
    console.log('  …', stage)
  } catch {
    /* page busy */
  }
}, 15_000)

await page.waitForSelector('audio[src^="blob:"]', { timeout: timeoutMs })
clearInterval(poll)
console.log('audio element appeared')

const stats = await page.evaluate(async () => {
  const audio = document.querySelector('audio')
  const response = await fetch(audio.src)
  const buffer = await response.arrayBuffer()
  const bytes = new Uint8Array(buffer)
  let binary = ''
  for (let i = 0; i < bytes.length; i += 0x8000) {
    binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000))
  }
  return { bytes: buffer.byteLength, duration: (buffer.byteLength - 44) / 2 / 24000, base64: btoa(binary) }
})

if (outPath) {
  writeFileSync(outPath, Buffer.from(stats.base64, 'base64'))
  console.log(`saved ${outPath}`)
}
console.log(`audio: ${stats.bytes} bytes, ~${stats.duration.toFixed(2)} s @ 24 kHz`)

await browser.close()

if (stats.bytes < 40_000 || stats.duration < 1) {
  console.error('E2E FAILED: audio too short or empty')
  process.exit(1)
}
if (errors.length) {
  console.error('E2E FAILED: console errors present')
  process.exit(1)
}
console.log('E2E PASSED: produced non-trivial audio through the full browser pipeline')
