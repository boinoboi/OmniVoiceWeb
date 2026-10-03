import puppeteer from 'puppeteer-core'

const executablePath = process.env.CHROME_PATH ?? '/usr/bin/google-chrome-stable'

const combos = [
  { name: 'headless + vulkan + gpu', headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-angle=vulkan', '--enable-gpu', '--ignore-gpu-blocklist'] },
  { name: 'headless + use-vulkan=native', headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan', '--use-vulkan=native', '--enable-gpu', '--ignore-gpu-blocklist'] },
  { name: 'headless + swiftshader', headless: true, args: ['--no-sandbox', '--enable-unsafe-webgpu', '--use-webgpu-adapter=swiftshader'] },
  { name: 'headful + unsafe-webgpu', headless: false, args: ['--no-sandbox', '--enable-unsafe-webgpu'] },
]

for (const combo of combos) {
  let browser
  try {
    browser = await puppeteer.launch({ executablePath, headless: combo.headless, args: combo.args, userDataDir: process.env.E2E_PROFILE })
    const page = await browser.newPage()
    await page.goto('http://localhost:4173/', { waitUntil: 'domcontentloaded', timeout: 15_000 })
    const info = await page.evaluate(async () => {
      if (!navigator.gpu) return 'no navigator.gpu'
      const adapter = await navigator.gpu.requestAdapter()
      if (!adapter) return 'no adapter'
      const i = adapter.info ?? {}
      return `${i.vendor ?? '?'} | ${i.architecture ?? '?'} | ${i.description ?? '?'}`
    })
    console.log(`${combo.name.padEnd(32)} -> ${info}`)
  } catch (error) {
    console.log(`${combo.name.padEnd(32)} -> ERROR ${error.message.split('\n')[0]}`)
  } finally {
    await browser?.close().catch(() => {})
  }
}
