// One-off: drive the real app in headless Chrome, run a genuine graph, and
// record assets/demo.gif + assets/demo.png. Requires the server running on :3000
// with a working OPENAI_API_KEY. Run: node scripts/make-demo.mjs
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import puppeteer from "puppeteer-core";
import { PNG } from "pngjs";
import gifenc from "gifenc";
const { GIFEncoder, quantize, applyPalette } = gifenc;

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.join(__dirname, "..");
const ASSETS = path.join(ROOT, "assets");
fs.mkdirSync(ASSETS, { recursive: true });

const CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const URL = process.env.DEMO_URL || "http://localhost:3000";
const QUERY = "What does LangSmith pricing include?";
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: "new",
  args: ["--no-sandbox", "--force-device-scale-factor=1"],
  defaultViewport: { width: 1180, height: 720 },
});

try {
  const page = await browser.newPage();
  await page.goto(URL, { waitUntil: "networkidle2" });
  await page.evaluate(async () => { await loadConfig(); });

  // Set up the Playground for a real RetrievalGraph run.
  await page.evaluate((q) => {
    changeTab("playground");
    selectGraph("RetrievalGraph");
    state.playgroundInput = q;
    document.getElementById("playground-input-box").value = q;
    render();
    window.scrollTo(0, 0);
  }, QUERY);
  await sleep(400);

  const frames = [];
  const grab = async () => Buffer.from(await page.screenshot({ type: "png" }));

  frames.push(await grab()); // idle, query typed

  // Fire the real run (don't await — we want to capture it in progress).
  await page.evaluate(() => { executeGraphPlayground(); });

  for (let i = 0; i < 16; i++) {
    await sleep(420);
    frames.push(await grab());
    const running = await page.evaluate(() => state.playgroundIsRunning);
    if (!running && i > 2) break;
  }
  await sleep(600);
  for (let i = 0; i < 3; i++) { frames.push(await grab()); await sleep(500); } // hold on final result

  // Crisp final PNG at 2x for the README hero.
  await page.setViewport({ width: 1180, height: 720, deviceScaleFactor: 2 });
  await page.evaluate(() => window.scrollTo(0, 0));
  await sleep(300);
  await page.screenshot({ path: path.join(ASSETS, "demo.png"), type: "png" });
  console.log("wrote assets/demo.png");

  // Encode the frames into an animated GIF.
  const enc = GIFEncoder();
  for (const buf of frames) {
    const { width, height, data } = PNG.sync.read(buf); // RGBA
    const palette = quantize(data, 256);
    const index = applyPalette(data, palette);
    enc.writeFrame(index, width, height, { palette, delay: 480 });
  }
  enc.finish();
  fs.writeFileSync(path.join(ASSETS, "demo.gif"), enc.bytes());
  const kb = (fs.statSync(path.join(ASSETS, "demo.gif")).size / 1024).toFixed(0);
  console.log(`wrote assets/demo.gif (${frames.length} frames, ${kb} KB)`);
} finally {
  await browser.close();
}
