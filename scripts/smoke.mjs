import puppeteer from 'puppeteer-core'

const url = process.argv[2] ?? 'http://localhost:4173/'
const executablePath = process.env.CHROME_PATH ?? '/usr/bin/google-chrome-stable'

const browser = await puppeteer.launch({
  executablePath,
  headless: true,
  args: ['--no-sandbox', '--disable-gpu'],
})

const failures = []

async function check(name, viewport) {
  const page = await browser.newPage()
  await page.setViewport(viewport)
  const errors = []
  page.on('pageerror', (e) => errors.push(`pageerror: ${e.message}`))
  page.on('console', (m) => {
    if (m.type() === 'error') errors.push(`console: ${m.text()}`)
  })

  await page.goto(url, { waitUntil: 'networkidle0', timeout: 60_000 })
  await page.waitForSelector('.filelist li', { timeout: 20_000 }).catch(() => {})

  const result = await page.evaluate(() => {
    const text = (sel) => document.querySelector(sel)?.textContent ?? ''
    return {
      title: document.title,
      codeSwitching: text('.pill'),
      profileButtons: document.querySelectorAll('.segmented button').length,
      liteGroups: document.querySelectorAll('.filelist li').length,
      deviceBadge: text('.card .badge'),
      hasWebGPU: typeof navigator.gpu !== 'undefined',
      hasCacheApi: typeof caches !== 'undefined',
      horizontalOverflow: document.documentElement.scrollWidth - window.innerWidth,
    }
  })

  await page.click('.segmented button:nth-child(2)')
  await new Promise((resolve) => setTimeout(resolve, 400))
  const fullGroups = await page.$$eval('.filelist li', (els) => els.length)
  result.fullGroups = fullGroups

  const checks = {
    title: result.title.includes('OmniVoice'),
    codeSwitching: result.codeSwitching.includes('Code-switching'),
    profiles: result.profileButtons === 2,
    liteGroups: result.liteGroups === 3,
    fullGroups: result.fullGroups === 4,
    noOverflow: result.horizontalOverflow <= 1,
    noErrors: errors.length === 0,
  }

  console.log(`\n[${name}] ${JSON.stringify(result, null, 2)}`)
  for (const [key, ok] of Object.entries(checks)) {
    console.log(`  ${ok ? 'PASS' : 'FAIL'} ${key}`)
    if (!ok) failures.push(`${name}:${key}`)
  }
  if (errors.length) console.log('  errors:', errors.slice(0, 6))

  await page.close()
}

await check('desktop', { width: 1280, height: 800 })
await check('iphone', { width: 390, height: 844, isMobile: true, hasTouch: true, deviceScaleFactor: 3 })
await check('ipad', { width: 820, height: 1180, isMobile: true, hasTouch: true, deviceScaleFactor: 2 })

await browser.close()

if (failures.length) {
  console.error(`\nSMOKE FAILED: ${failures.join(', ')}`)
  process.exit(1)
}
console.log('\nSMOKE PASSED')
