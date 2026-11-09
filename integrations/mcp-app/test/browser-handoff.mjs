import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
const { chromium } = await import(process.env.PLAYWRIGHT_MODULE ?? 'playwright');
const browser = await chromium.launch({ headless: true, ...(process.env.CHROME_PATH ? { executablePath: process.env.CHROME_PATH } : {}) });
const output = process.env.REVIEW_DIR ?? '../../../.impeccable/review';
await mkdir(output, { recursive: true });
const page = await browser.newPage({ viewport: { width: 390, height: 844 }, colorScheme: 'light' });
page.setDefaultTimeout(10000);
const errors = [];
page.on('pageerror', (error) => errors.push(error.message));
await page.goto('http://127.0.0.1:62747/dev/preview.html');
const frame = page.frameLocator('iframe');
async function capture(name) {
  await frame.locator('html').evaluate(() => document.fonts.ready);
  await frame.locator('.app-shell').screenshot({ path: `${output}/${name}.png`, animations: 'disabled' });
}
async function chooseRevision(version) {
  await frame.getByRole('button', { name: version === 'v0' ? 'Previous revision' : 'Next revision', exact: true }).click();
  await frame.locator(`.revision-slide[data-version="${version}"][aria-hidden="false"]`).waitFor();
}
try {
  await frame.getByAltText('Vector artwork for v1', { exact: true }).waitFor();
  assert.equal(await frame.getByRole('img', { name: 'vectorEyes', exact: true }).evaluate(async (image) => {
    await image.decode();
    return image.src.startsWith('data:image/svg+xml');
  }), true, 'The vectorEyes lockup must load as a self-contained SVG');
  assert.equal(await frame.getByRole('button', { name: 'Source & difference', exact: true }).count(), 0);
  assert.equal(await frame.getByText('Rerun settings & passes', { exact: true }).count(), 0);
  assert.equal(await frame.locator('.revision-thumbnail, .revision-strip').count(), 0);
  assert.equal(await frame.locator('.revision-slide[aria-hidden="false"] img').count(), 1);
  assert.equal(await frame.getByAltText('Vector artwork for v1', { exact: true }).evaluate((image) => image.src.startsWith('data:image/svg+xml')), true);
  assert.equal(await frame.locator('html').evaluate((node) => node.scrollWidth > node.clientWidth), false);
  await frame.locator('.revision-carousel').hover();
  await page.mouse.wheel(-390, 0);
  await frame.getByAltText('Vector artwork for v0', { exact: true }).waitFor();
  await frame.locator('.revision-carousel').focus();
  await frame.locator('.revision-carousel').press('ArrowRight');
  await frame.getByAltText('Vector artwork for v1', { exact: true }).waitFor();
  assert.equal(await frame.locator('.revision-carousel').evaluate((node) => {
    const viewport = node.getBoundingClientRect();
    return [...node.querySelectorAll('.revision-slide')].filter((slide) => {
      const box = slide.getBoundingClientRect();
      return box.left < viewport.right - 1 && box.right > viewport.left + 1;
    }).length;
  }), 1, 'Exactly one full-size revision occupies the viewport after navigation');
  await capture('mobile-handoff');

  await frame.getByRole('button', { name: 'Review & edit', exact: true }).click();
  await frame.getByRole('button', { name: 'Segments & features', exact: true }).click();
  await frame.getByAltText('path_map for v1', { exact: true }).waitFor();
  await frame.getByRole('textbox', { name: 'What would you like to change?' }).fill('Make the red tip lower side one smooth curve.');
  await frame.getByText('Diagnostics · v1', { exact: true }).click();
  await frame.locator('.artifact-uri').filter({ hasText: 'Parent v0' }).waitFor();
  await capture('mobile-review');

  await chooseRevision('v0');
  await frame.getByAltText('path_map for v0', { exact: true }).waitFor();
  assert.equal(await frame.getByRole('textbox', { name: 'What would you like to change?' }).inputValue(), '');
  assert.equal(await frame.getByText('Diagnostics · v1', { exact: true }).count(), 0);
  await frame.getByRole('textbox', { name: 'What would you like to change?' }).fill('A different draft for v0.');
  await chooseRevision('v1');
  await frame.getByAltText('path_map for v1', { exact: true }).waitFor();
  assert.equal(await frame.getByRole('textbox', { name: 'What would you like to change?' }).inputValue(), 'Make the red tip lower side one smooth curve.');
  await frame.getByRole('button', { name: 'Ask agent to revise', exact: true }).click();
  await frame.getByRole('button', { name: 'Request sent', exact: true }).waitFor();
  const message = await page.evaluate(() => window.previewEvents.find((item) => item.message));
  assert.match(JSON.stringify(message), /starting from v1/);
  assert.equal(await frame.getByText('Creating next revision…', { exact: true }).count(), 0, 'A queued edit is not confirmed generation');

  // Export is always the selected revision SVG, even while viewing a segment map.
  const downloadEvent = page.waitForEvent('download');
  await frame.getByRole('button', { name: 'Export SVG', exact: true }).click();
  const download = await downloadEvent;
  assert.match(download.suggestedFilename(), /-v1-svg\.svg$/);
  assert.match(await readFile(await download.path(), 'utf8'), /<svg/);

  await frame.getByText('Settings used · v1', { exact: true }).click();
  await frame.getByRole('heading', { name: 'Source trace settings', exact: true }).waitFor();
  await frame.getByText('Rerun settings & passes', { exact: true }).click();
  await frame.getByRole('heading', { name: 'Rerun from the source', exact: true }).waitFor();
  await capture('mobile-settings');
  await frame.getByRole('button', { name: 'Close review', exact: true }).click();
  await frame.getByAltText('Vector artwork for v1', { exact: true }).waitFor();
  assert.equal(await frame.getByText('Diagnostics · v1', { exact: true }).count(), 0);

  await page.evaluate(() => window.previewBridge.sendToolInput({ arguments: { plan: { base_version: 'v1', ops: [] } } }));
  await frame.getByText('Creating next revision…', { exact: true }).waitFor();
  await frame.getByAltText('Vector artwork for v1', { exact: true }).waitFor();
  await capture('mobile-pending');

  await page.setViewportSize({ width: 1440, height: 1000 });
  await frame.getByAltText('Vector artwork for v1', { exact: true }).waitFor();
  await capture('desktop-handoff');
  await frame.getByRole('button', { name: 'Review & edit', exact: true }).click();
  await frame.getByRole('button', { name: 'Source & difference', exact: true }).click();
  await frame.getByAltText('review_panel for v1', { exact: true }).waitFor();
  await frame.getByLabel('Review zoom').selectOption('2');
  assert.equal(await frame.locator('.revision-slide[aria-hidden="false"] .preview-stage').evaluate((node) => node.scrollWidth > node.clientWidth), true);
  await frame.getByLabel('Review zoom').selectOption('1');
  await capture('desktop-review');
  await page.emulateMedia({ colorScheme: 'dark', reducedMotion: 'reduce' });
  await page.setViewportSize({ width: 390, height: 844 });
  assert.equal(await frame.locator('.dot-grid i').first().evaluate((node) => getComputedStyle(node).animationName), 'none');
  await capture('mobile-dark-reduced-motion');

  await page.goto('http://127.0.0.1:62747/dev/preview.html?scenario=generating');
  await frame.getByRole('heading', { name: 'Creating your vector artwork' }).waitFor();
  await frame.getByText('Creating SVG…', { exact: true }).waitFor();
  await capture('mobile-generating');
  await page.goto('http://127.0.0.1:62747/dev/preview.html?scenario=error');
  await frame.getByRole('button', { name: 'Retry artifact', exact: true }).waitFor();
  await capture('error');
  await page.goto('http://127.0.0.1:62747/dev/preview.html?scenario=empty');
  await frame.getByRole('heading', { name: 'Your next image, in vector.' }).waitFor();
  assert.deepEqual(errors, []);
  console.log('Browser checks passed: single-picture SVG carousel, mobile/desktop handoff, parent-linked revisions, scoped evidence, retained drafts, versioned edits, SVG-only export, zoom, truthful generation, dark/reduced-motion, error, empty.');
} finally { await browser.close(); }
