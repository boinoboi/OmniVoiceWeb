import puppeteer from 'puppeteer-core'

const b = await puppeteer.launch({
  executablePath: '/usr/bin/google-chrome-stable',
  headless: false,
  args: ['--no-sandbox', '--enable-unsafe-webgpu', '--enable-features=Vulkan'],
})
const p = await b.newPage()
await p.goto('http://localhost:4173/')
const r = await p.evaluate(async () => {
  const a = await navigator.gpu.requestAdapter()
  if (!a) return 'no adapter'
  return { info: a.info, features: [...a.features], f16: a.features.has('shader-f16') }
})
console.log(JSON.stringify(r, null, 2))
await b.close()
