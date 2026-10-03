import { mkdirSync } from 'node:fs'
import puppeteer from 'puppeteer-core'

const url = process.argv[2] ?? 'http://localhost:4173/'
const outDir = process.env.SHOT_DIR ?? '/tmp/opencode/shots'
mkdirSync(outDir, { recursive: true })

const viewports = [
  { name: 'desktop-1440', width: 1440, height: 900, dpr: 1 },
  { name: 'laptop-1280', width: 1280, height: 800, dpr: 1 },
  { name: 'laptop-1366', width: 1366, height: 768, dpr: 1 },
  { name: 'tablet-820', width: 820, height: 1180, dpr: 2 },
  { name: 'phone-390', width: 390, height: 844, dpr: 3 },
  { name: 'phone-360', width: 360, height: 740, dpr: 2 },
]

const browser = await puppeteer.launch({
  executablePath: process.env.CHROME_PATH ?? '/usr/bin/google-chrome-stable',
  headless: true,
  args: ['--no-sandbox', '--disable-gpu', '--hide-scrollbars'],
})

for (const vp of viewports) {
  const page = await browser.newPage()
  await page.setViewport({ width: vp.width, height: vp.height, deviceScaleFactor: vp.dpr, isMobile: vp.width < 600, hasTouch: vp.width < 600 })
  await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60_000 })
  await page.waitForSelector('.card', { timeout: 30_000 })
  await new Promise((r) => setTimeout(r, 400))
  await page.screenshot({ path: `${outDir}/${vp.name}.png`, fullPage: true })

  const cloner = await page.$('.cloner summary')
  if (cloner) {
    await cloner.click()
    await new Promise((r) => setTimeout(r, 200))
    await page.screenshot({ path: `${outDir}/${vp.name}-cloner.png`, fullPage: true })
  }

  const metrics = await page.evaluate(() => ({
    scrollW: document.documentElement.scrollWidth,
    clientW: document.documentElement.clientWidth,
    scrollH: document.documentElement.scrollHeight,
  }))
  const overflow = metrics.scrollW > metrics.clientW + 1
  console.log(`${vp.name.padEnd(14)} ${vp.width}x${vp.height} scrollW=${metrics.scrollW} clientW=${metrics.clientW} h=${metrics.scrollH} ${overflow ? 'HORIZONTAL-OVERFLOW' : 'ok'}`)
  if (overflow) {
    const offenders = await page.evaluate(() => {
      const w = document.documentElement.clientWidth
      const out = []
      for (const el of Array.from(document.querySelectorAll('*'))) {
        const r = el.getBoundingClientRect()
        if (r.right > w + 1 || r.left < -1) {
          out.push(`${el.tagName}.${el.className} w=${Math.round(r.width)} r=${Math.round(r.right)}`)
        }
      }
      return out.slice(0, 10)
    })
    console.log('  offenders:', offenders.join(' | '))
  }
  await page.close()
}

await browser.close()
console.log(`screenshots -> ${outDir}`)
