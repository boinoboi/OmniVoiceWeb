import puppeteer from 'puppeteer-core'

const url = process.argv[2] ?? 'https://boinoboi.github.io/OmniVoiceWeb/'
const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome-stable',
  headless: process.env.E2E_HEADFUL !== '1',
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan'],
})
const page = await browser.newPage()
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })
await new Promise((r) => setTimeout(r, 3000))
const info = await page.evaluate(async () => {
  const hasGpu = !!navigator.gpu
  let adapter = null
  let err = null
  try {
    const a = await navigator.gpu?.requestAdapter()
    adapter = a ? `${a.info?.vendor ?? '?'} | ${a.info?.architecture ?? '?'}` : null
  } catch (e) {
    err = String(e)
  }
  const badge = [...document.querySelectorAll('.badge')].map((e) => e.textContent).join(' | ')
  return { url: location.href, secure: isSecureContext, hasGpu, adapter, err, badge }
})
console.log(JSON.stringify(info, null, 2))
await browser.close()
