import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);

  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the FrameLine local viewer", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);

  const html = await response.text();
  assert.match(html, /<title>FrameLine — Local Frame-Accurate Video Viewer<\/title>/i);
  assert.match(html, /FrameLine/);
  assert.match(html, /Annotate sampled episodes/);
  assert.match(html, /LEHOME DATASET ANNOTATION/);
  assert.match(html, /Files stay on this device/);
  assert.doesNotMatch(html, /codex-preview|Your site is taking shape|react-loading-skeleton/i);
});

test("source includes exact metadata, keyboard shuttle, and pointer scrubbing", async () => {
  const [page, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /from "mp4box"/);
  assert.match(page, /videoTrack\.nb_samples/);
  assert.match(page, /ArrowLeft/);
  assert.match(page, /ArrowRight/);
  assert.match(page, /setPointerCapture/);
  assert.match(page, /URL\.createObjectURL/);
  assert.match(page, /requestVideoFrameCallback/);
  assert.match(packageJson, /"name": "frameline-local-video-viewer"/);
  assert.doesNotMatch(packageJson, /lucide-react|react-loading-skeleton/);
  await assert.rejects(access(new URL("../app/_sites-preview", import.meta.url)));
});

test("unsupported codecs fall back to the on-device converter", async () => {
  const [page, helper, packageJson] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../scripts/run-local.mjs", import.meta.url), "utf8"),
    readFile(new URL("../package.json", import.meta.url), "utf8"),
  ]);

  assert.match(page, /canPlayType/);
  assert.match(page, /prepareCompatibleVideo/);
  assert.match(page, /video\.play\(\)\.catch/);
  assert.match(page, /onError=\{onVideoError\}/);
  assert.match(helper, /"libx264"/);
  assert.match(helper, /"Accept-Ranges": "bytes"/);
  assert.match(packageJson, /node scripts\/run-local\.mjs dev/);
});

test("held arrow shuttle is not cancelled by current-frame rerenders", async () => {
  const page = await readFile(new URL("../app/page.tsx", import.meta.url), "utf8");

  assert.match(page, /currentFrameRef\.current >= maxFrame/);
  assert.match(page, /window\.setTimeout\(repeat, 320\)/);
  assert.match(page, /if \(!event\.repeat\) startHold/);
  assert.doesNotMatch(
    page,
    /\[currentFrame, isPreparing, maxFrame, prepareCompatibleVideo, seekFrame\]/,
  );
});

test("single-lane annotation imports labels and exports contiguous checkpoints", async () => {
  const [page, sample, css] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../public/sample-segments.json", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
  ]);
  const sampleData = JSON.parse(sample);

  assert.ok(sampleData.labels.length > 0);
  assert.match(page, /Load JSON/);
  assert.match(page, /Use sample labels/);
  assert.match(page, /SINGLE ANNOTATION TIMELINE/);
  assert.match(page, /end_frame_exclusive/);
  assert.match(page, /coverage_is_contiguous: true/);
  assert.match(page, /showSaveFilePicker/);
  assert.match(page, /Save checkpoints\.json/);
  assert.match(page, /commitCurrentSegment/);
  assert.match(page, /event\.key !== "Enter"/);
  assert.match(page, /event\.key === "ArrowUp" \|\| event\.key === "ArrowDown"/);
  assert.match(page, /selectSegment\(activeSegmentIndex/);
  assert.doesNotMatch(page, /disabled=\{unavailable\}/);
  assert.match(page, /undoLastAnnotation/);
  assert.match(page, /event\.ctrlKey \|\| event\.metaKey/);
  assert.match(page, /annotationHistoryRef\.current\.pop/);
  assert.match(page, /annotationRedoRef\.current\.pop/);
  assert.match(page, /redoLastAnnotation/);
  assert.match(page, /key === "y"/);
  assert.match(page, /switchTimelineMode/);
  assert.match(page, /key === "s" \? "scrub" : key === "a" \? "annotate" : "progress"/);
  assert.match(page, /buildSegmentProgress/);
  assert.match(page, /toggleProgressDirection/);
  assert.match(page, /piecewise_linear_direction_toggles/);
  assert.match(page, /progressCanvasRef/);
  assert.match(page, /progress: buildSegmentProgress/);
  assert.match(page, /Progress Edit/);
  assert.match(page, /commitProgressPoint/);
  assert.match(page, /pendingProgressPoint/);
  assert.match(page, /piecewise_linear_control_points/);
  assert.match(page, /linear_pieces/);
  assert.match(page, /progress-control-dot/);
  assert.match(page, /clickedProgressLane/);
  assert.match(page, /setInteractionMode\("progress"\)/);
  assert.match(page, /event\.shiftKey \? drag\.originFrame/);
  assert.match(page, /event\.ctrlKey \? drag\.originProgress/);
  assert.match(page, /event\.clientX - bounds\.left/);
  assert.match(page, /currentProgress = currentModel\.per_frame/);
  assert.match(page, /event\.clientX - drag\.originClientX/);
  assert.match(page, /event\.clientY - drag\.originClientY/);
  assert.match(page, /cursorFrame = currentFrameRef\.current/);
  assert.match(page, /if \(interactionMode === "progress"\) setPendingProgressPoint\(null\)/);
  assert.match(page, /Progress point prepared at cursor frame/);
  assert.match(css, /\.playhead-label \{[^}]*color: var\(--red\)/);
  assert.doesNotMatch(page, /<strong>Frame \{hoverFrame\}<\/strong>/);
  assert.doesNotMatch(css, /\.hover-playhead > div/);
  assert.match(css, /\.progress-track/);
  assert.match(css, /\.progress-control-dot/);
  assert.match(page, /Press Enter to save and continue/);
  assert.match(page, /setAnnotationPreviewEnd\(requestedEnd\)/);
  assert.doesNotMatch(page, /applySegmentEnd\(annotationDragRef\.current\.segmentIndex/);
  assert.match(page, /annotationSegments\[activeSegmentIndex - 1\]\.end_frame_exclusive/);
  assert.match(page, /inspector-segments/);
  assert.match(page, /videoDisplayScale/);
  assert.match(page, /Video size/);
  assert.match(page, /<\/aside>\s*<aside className="segments-panel">/);
  assert.match(page, /<aside className="annotation-panel">[\s\S]*ANNOTATION WORKFLOW[\s\S]*<aside className="segments-panel">/);
  assert.match(css, /\.segment-selector\.vertical/);
  assert.match(css, /\.display-scale-control/);
  assert.match(css, /\.annotation-panel \.annotation-console/);
  assert.match(css, /height: 100vh;\s*overflow: hidden/);
  assert.match(css, /grid-template-columns: minmax\(0, 1fr\) 220px 270px 290px/);
});

test("dataset mode samples unannotated garment episodes and saves per-episode checkpoints", async () => {
  const [page, helper, extractor, labels, readme] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../scripts/run-local.mjs", import.meta.url), "utf8"),
    readFile(new URL("../scripts/extract-episode-video.py", import.meta.url), "utf8"),
    readFile(new URL("../public/garment-segment-labels.json", import.meta.url), "utf8"),
    readFile(new URL("../README.md", import.meta.url), "utf8"),
  ]);
  const templates = JSON.parse(labels).templates;

  assert.deepEqual(templates.map((template) => template.category_id), [
    "pant",
    "shorts",
    "top_long_sleeve",
    "top_short_sleeve",
  ]);
  assert.match(page, /Start 25% sampling session/);
  assert.match(page, /dataset-session-tracker/);
  assert.match(page, /Next episode/);
  assert.match(page, /Save episode checkpoint/);
  assert.match(page, /\/dataset\/checkpoint/);
  assert.match(helper, /Math\.round\(allEpisodeIndices\.length \* 0\.25\)/);
  assert.match(helper, /!existsSync\(checkpointPath\(episodeIndex\)\)/);
  assert.match(helper, /annotations.*temporal_checkpoints/);
  assert.match(helper, /frameline_sessions/);
  assert.match(helper, /episode_\$\{String\(episodeIndex\)\.padStart\(6, "0"\)\}/);
  assert.match(extractor, /observation\.images\.top_rgb/);
  assert.match(extractor, /image2pipe/);
  assert.match(readme, /1,045 episodes in total/);
});
