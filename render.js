// usage: node render.js [comma-separated times | all] [outdir] [workers]
const puppeteer = require('puppeteer-core'), fs = require('fs'), path = require('path');
(async () => {
  const arg = process.argv[2] || 'all', out = process.argv[3] || 'frames', workers = +(process.argv[4] || 4);
  const list = arg === 'all' ? [...Array(900).keys()] : arg.split(',').map(t => Math.round(+t * 60));
  fs.mkdirSync(out, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', headless: true,
    args: ['--allow-file-access-from-files', '--force-color-profile=srgb'] });
  let next = 0, done = 0; const t0 = Date.now();
  await Promise.all([...Array(Math.min(workers, list.length))].map(async () => {
    const p = await browser.newPage(); await p.setViewport({ width: 1920, height: 1080 });
    p.on('pageerror', e => console.log('ERR', e.message)); p.on('console', m => console.log('PAGE', m.text()));
    await p.goto('file://' + path.resolve(__dirname, 'index.html')); await p.evaluate(() => window.ready);
    while (next < list.length) {
      const f = list[next++];
      const d = await p.evaluate(f => { renderFrame(f / 60); return document.getElementById('c').toDataURL('image/png'); }, f);
      fs.writeFileSync(path.join(out, `f_${String(f).padStart(4, '0')}.png`), Buffer.from(d.split(',')[1], 'base64'));
      if (++done % 100 === 0) console.log(done, 'frames', ((Date.now() - t0) / 1000).toFixed(1) + 's');
    }
  }));
  await browser.close(); console.log('done', done, ((Date.now() - t0) / 1000).toFixed(1) + 's');
})();
