import {
  createReadStream,
  createWriteStream,
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  renameSync,
  statSync,
  writeFileSync,
} from "node:fs";
import { randomInt, randomUUID } from "node:crypto";
import { createServer } from "node:http";
import { tmpdir } from "node:os";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawn } from "node:child_process";

const mode = process.argv[2] === "start" ? "start" : "dev";
const projectDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const mediaDirectory = join(tmpdir(), "frameline-local-media");
const datasetMediaDirectory = join(mediaDirectory, "dataset");
const sessionsDirectory = resolve(process.env.FRAMELINE_SESSION_ROOT || join(projectDirectory, ".frameline_sessions"));
const extractScript = join(projectDirectory, "scripts", "extract-episode-video.py");
const generatingVideos = new Map();
const episodeMetadataCache = new Map();
const preparedEpisodes = new Map();

const CATEGORY_DEFINITIONS = [
  { id: "shorts", label: "Shorts" },
  { id: "top_long_sleeve", label: "Top Long sleeve" },
  { id: "top_short_sleeve", label: "Top Short sleeve" },
];

const DATASET_DEFINITIONS = [
  {
    id: "human",
    label: "Human",
    root: resolve(process.env.FRAMELINE_HUMAN_DATASET_ROOT || process.env.FRAMELINE_DATASET_ROOT || "D:\\pretrain_lehome_all_garment_data_z180"),
    fpsFallback: 23,
    categories: [
      { id: "shorts", label: "Shorts", start: 1018, endExclusive: 2039 },
      { id: "top_long_sleeve", label: "Top Long sleeve", start: 2039, endExclusive: 2875 },
      { id: "top_short_sleeve", label: "Top Short sleeve", start: 2875, endExclusive: 4180 },
    ],
  },
  {
    id: "sim",
    label: "Sim",
    root: resolve(process.env.FRAMELINE_SIM_DATASET_ROOT || "E:\\Lehome-Dataset\\lehome_round_2_dataset\\sim_dataset\\robot_sim_ft_lehome_all_garment_data_z180"),
    fpsFallback: 30,
    categories: [
      { id: "shorts", label: "Shorts", start: 250, endExclusive: 500 },
      { id: "top_long_sleeve", label: "Top Long sleeve", start: 500, endExclusive: 750 },
      { id: "top_short_sleeve", label: "Top Short sleeve", start: 750, endExclusive: 1000 },
    ],
  },
];

const DATASET_ORDER = DATASET_DEFINITIONS.map((dataset) => dataset.id);

mkdirSync(mediaDirectory, { recursive: true });
mkdirSync(datasetMediaDirectory, { recursive: true });
mkdirSync(sessionsDirectory, { recursive: true });

function cors(response) {
  response.setHeader("Access-Control-Allow-Origin", "http://localhost:3000");
  response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  response.setHeader("Access-Control-Allow-Headers", "Content-Type, X-File-Name");
}

function json(response, status, body) {
  cors(response);
  response.writeHead(status, { "Content-Type": "application/json", "Cache-Control": "no-store" });
  response.end(JSON.stringify(body));
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function writeJsonAtomic(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  const temporaryPath = `${path}.${randomUUID()}.tmp`;
  writeFileSync(temporaryPath, `${JSON.stringify(value, null, 2)}\n`, "utf8");
  renameSync(temporaryPath, path);
}

function readRequestJson(request, maximumBytes = 25 * 1024 * 1024) {
  return new Promise((resolveBody, rejectBody) => {
    const chunks = [];
    let bytes = 0;
    request.on("data", (chunk) => {
      bytes += chunk.length;
      if (bytes > maximumBytes) {
        rejectBody(new Error("Request is too large."));
        request.destroy();
        return;
      }
      chunks.push(chunk);
    });
    request.on("end", () => {
      try {
        resolveBody(JSON.parse(Buffer.concat(chunks).toString("utf8") || "{}"));
      } catch {
        rejectBody(new Error("Request body is not valid JSON."));
      }
    });
    request.on("error", rejectBody);
  });
}

function datasetById(datasetId) {
  const dataset = DATASET_DEFINITIONS.find((candidate) => candidate.id === datasetId);
  if (!dataset) throw new Error(`Unknown dataset '${datasetId}'.`);
  return dataset;
}

function categoryForDataset(dataset, categoryId) {
  const category = dataset.categories.find((candidate) => candidate.id === categoryId);
  if (!category) throw new Error(`Dataset '${dataset.id}' does not define category '${categoryId}'.`);
  return category;
}

function datasetAnnotationsDirectory(dataset) {
  return join(dataset.root, "annotations");
}

function datasetCheckpointsDirectory(dataset) {
  return join(datasetAnnotationsDirectory(dataset), "temporal_checkpoints");
}

function episodeStem(episodeIndex) {
  return `episode_${String(episodeIndex).padStart(6, "0")}`;
}

function episodeChunk(episodeIndex) {
  return `chunk-${String(Math.floor(episodeIndex / 1000)).padStart(3, "0")}`;
}

function parquetPath(dataset, episodeIndex) {
  return join(dataset.root, "data", episodeChunk(episodeIndex), `${episodeStem(episodeIndex)}.parquet`);
}

function checkpointPath(dataset, episodeIndex) {
  return join(datasetCheckpointsDirectory(dataset), episodeChunk(episodeIndex), episodeStem(episodeIndex), "checkpoints.json");
}

function sessionPath(sessionId) {
  if (!/^[a-f0-9-]+$/.test(sessionId)) throw new Error("Invalid session identifier.");
  return join(sessionsDirectory, `${sessionId}.json`);
}

function shuffle(values) {
  const result = [...values];
  for (let index = result.length - 1; index > 0; index -= 1) {
    const replacement = randomInt(index + 1);
    [result[index], result[replacement]] = [result[replacement], result[index]];
  }
  return result;
}

function readDatasetInfo(dataset) {
  const infoPath = join(dataset.root, "meta", "info.json");
  if (!existsSync(infoPath)) throw new Error(`Dataset metadata was not found at ${infoPath}.`);
  return readJson(infoPath);
}

function publicDatasetInfo(dataset) {
  const info = readDatasetInfo(dataset);
  const totalEpisodes = Number(info.total_episodes ?? 0);
  return {
    id: dataset.id,
    label: dataset.label,
    root: dataset.root,
    total_episodes: totalEpisodes,
    fps: Number(info.fps || dataset.fpsFallback),
    checkpoint_directory: datasetCheckpointsDirectory(dataset),
    categories: dataset.categories.map((definition) => ({
      ...definition,
      total_episodes: Math.max(0, Math.min(definition.endExclusive, totalEpisodes) - definition.start),
    })),
  };
}

function categoryProgress(category) {
  const completedSet = new Set(category.completed_episode_indices);
  const remaining = category.sampled_episode_indices.filter((episodeIndex) => !completedSet.has(episodeIndex));
  return {
    id: category.id,
    label: category.label,
    total_episodes: category.total_episodes,
    eligible_at_sampling: category.eligible_at_sampling,
    target_sample_count: category.target_sample_count,
    sampled_count: category.sampled_episode_indices.length,
    completed_count: category.completed_episode_indices.length,
    remaining_count: remaining.length,
  };
}

function datasetSessionProgress(datasetSession) {
  const categories = datasetSession.categories.map(categoryProgress);
  return {
    id: datasetSession.id,
    label: datasetSession.label,
    root: datasetSession.root,
    total_episodes: datasetSession.total_episodes,
    categories,
    sampled_count: categories.reduce((sum, category) => sum + category.sampled_count, 0),
    completed_count: categories.reduce((sum, category) => sum + category.completed_count, 0),
  };
}

function combinedCategoryProgress(session) {
  return CATEGORY_DEFINITIONS.map((definition) => {
    const perDataset = session.datasets
      .map((datasetSession) => datasetSession.categories.find((category) => category.id === definition.id))
      .filter(Boolean)
      .map(categoryProgress);
    return {
      id: definition.id,
      label: definition.label,
      total_episodes: perDataset.reduce((sum, category) => sum + category.total_episodes, 0),
      eligible_at_sampling: perDataset.reduce((sum, category) => sum + category.eligible_at_sampling, 0),
      target_sample_count: perDataset.reduce((sum, category) => sum + category.target_sample_count, 0),
      sampled_count: perDataset.reduce((sum, category) => sum + category.sampled_count, 0),
      completed_count: perDataset.reduce((sum, category) => sum + category.completed_count, 0),
      remaining_count: perDataset.reduce((sum, category) => sum + category.remaining_count, 0),
    };
  });
}

function remainingEpisodeIndices(datasetSession, categoryId) {
  const category = datasetSession.categories.find((candidate) => candidate.id === categoryId);
  if (!category) return [];
  const completed = new Set(category.completed_episode_indices);
  return category.sampled_episode_indices.filter((episodeIndex) => !completed.has(episodeIndex));
}

function activeSelection(session) {
  for (const category of CATEGORY_DEFINITIONS) {
    const preferredDatasetId = session.next_dataset_id_by_category?.[category.id] || DATASET_ORDER[0];
    const datasetIds = [preferredDatasetId, ...DATASET_ORDER.filter((datasetId) => datasetId !== preferredDatasetId)];
    for (const datasetId of datasetIds) {
      const datasetSession = session.datasets.find((candidate) => candidate.id === datasetId);
      if (!datasetSession) continue;
      const remaining = remainingEpisodeIndices(datasetSession, category.id);
      if (remaining.length) {
        return {
          dataset_id: datasetSession.id,
          dataset_label: datasetSession.label,
          category_id: category.id,
          category_label: category.label,
          episode_index: remaining[0],
        };
      }
    }
  }
  return null;
}

function publicSession(session) {
  const datasets = session.datasets.map(datasetSessionProgress);
  const categories = combinedCategoryProgress(session);
  const active = activeSelection(session);
  return {
    session_id: session.session_id,
    created_at: session.created_at,
    sample_fraction: session.sample_fraction,
    status: active ? "active" : "complete",
    datasets,
    categories,
    active_dataset_id: active?.dataset_id ?? null,
    active_dataset_label: active?.dataset_label ?? null,
    active_category_id: active?.category_id ?? null,
    active_category_label: active?.category_label ?? null,
    active_episode_index: active?.episode_index ?? null,
    sampled_count: datasets.reduce((sum, dataset) => sum + dataset.sampled_count, 0),
    completed_count: datasets.reduce((sum, dataset) => sum + dataset.completed_count, 0),
  };
}

function refreshSession(session) {
  if (!session || session.schema_version !== "2.0" || !Array.isArray(session.datasets)) {
    throw new Error("This is not a supported paired-dataset sampling session.");
  }
  for (const datasetSession of session.datasets) {
    const dataset = datasetById(datasetSession.id);
    for (const category of datasetSession.categories) {
      category.completed_episode_indices = category.sampled_episode_indices.filter((episodeIndex) => existsSync(checkpointPath(dataset, episodeIndex)));
    }
  }
  session.updated_at = new Date().toISOString();
  session.status = publicSession(session).status;
  return session;
}

function loadSession(sessionId) {
  const path = sessionPath(sessionId);
  if (!existsSync(path)) throw new Error("Sampling session was not found.");
  const session = refreshSession(readJson(path));
  writeJsonAtomic(path, session);
  return session;
}

function latestActiveSession() {
  if (!existsSync(sessionsDirectory)) return null;
  const paths = readdirSync(sessionsDirectory)
    .filter((name) => /^[a-f0-9-]+\.json$/.test(name))
    .map((name) => join(sessionsDirectory, name))
    .sort((left, right) => statSync(right).mtimeMs - statSync(left).mtimeMs);
  for (const path of paths) {
    try {
      const session = refreshSession(readJson(path));
      writeJsonAtomic(path, session);
      if (session.status === "active") return publicSession(session);
    } catch {
      // Ignore malformed or historical single-dataset session files.
    }
  }
  return null;
}

function createSamplingSession() {
  const sessionId = randomUUID();
  const datasets = DATASET_DEFINITIONS.map((dataset) => {
    const info = readDatasetInfo(dataset);
    const totalEpisodes = Number(info.total_episodes ?? 0);
    const categories = dataset.categories.map((definition) => {
      const endExclusive = Math.min(definition.endExclusive, totalEpisodes);
      const allEpisodeIndices = Array.from(
        { length: Math.max(0, endExclusive - definition.start) },
        (_, offset) => definition.start + offset,
      );
      const eligible = allEpisodeIndices.filter((episodeIndex) => existsSync(parquetPath(dataset, episodeIndex)) && !existsSync(checkpointPath(dataset, episodeIndex)));
      const targetSampleCount = Math.round(allEpisodeIndices.length * 0.25);
      return {
        ...definition,
        total_episodes: allEpisodeIndices.length,
        eligible_at_sampling: eligible.length,
        target_sample_count: targetSampleCount,
        sampled_episode_indices: shuffle(eligible).slice(0, Math.min(targetSampleCount, eligible.length)),
        completed_episode_indices: [],
      };
    });
    return {
      id: dataset.id,
      label: dataset.label,
      root: dataset.root,
      total_episodes: totalEpisodes,
      categories,
    };
  });
  const session = {
    schema_version: "2.0",
    session_id: sessionId,
    dataset_roots: Object.fromEntries(DATASET_DEFINITIONS.map((dataset) => [dataset.id, dataset.root])),
    dataset_order: DATASET_ORDER,
    category_order: CATEGORY_DEFINITIONS.map((category) => category.id),
    next_dataset_id_by_category: Object.fromEntries(CATEGORY_DEFINITIONS.map((category) => [category.id, DATASET_ORDER[0]])),
    sample_fraction: 0.25,
    created_at: new Date().toISOString(),
    updated_at: new Date().toISOString(),
    status: "active",
    datasets,
  };
  writeJsonAtomic(sessionPath(sessionId), session);
  return session;
}

function flipNextDatasetForCategory(session, categoryId, justCompletedDatasetId) {
  if (!session.next_dataset_id_by_category) session.next_dataset_id_by_category = {};
  const preferred = DATASET_ORDER.find((datasetId) => datasetId !== justCompletedDatasetId) || justCompletedDatasetId;
  const preferredDatasetSession = session.datasets.find((datasetSession) => datasetSession.id === preferred);
  if (preferredDatasetSession && remainingEpisodeIndices(preferredDatasetSession, categoryId).length) {
    session.next_dataset_id_by_category[categoryId] = preferred;
    return;
  }
  session.next_dataset_id_by_category[categoryId] = justCompletedDatasetId;
}

function validateCheckpointPayload(payload, expectedFrames) {
  if (!payload || typeof payload !== "object" || !Array.isArray(payload.segments) || !payload.segments.length) {
    throw new Error("Checkpoint JSON must contain at least one segment.");
  }
  let expectedStart = 0;
  for (const segment of payload.segments) {
    const start = Number(segment.start_frame);
    const end = Number(segment.end_frame_exclusive);
    if (!Number.isInteger(start) || !Number.isInteger(end) || start !== expectedStart || end <= start) {
      throw new Error("Checkpoint segments must be ordered, contiguous, and non-empty.");
    }
    expectedStart = end;
  }
  if (expectedStart !== expectedFrames) {
    throw new Error(`Checkpoint coverage must end at frame ${expectedFrames}.`);
  }
}

function serveMp4(request, response, outputPath) {
  if (!existsSync(outputPath)) {
    json(response, 404, { error: "Video not found." });
    return;
  }
  cors(response);
  const size = statSync(outputPath).size;
  const range = request.headers.range;
  if (range) {
    const match = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (!match) {
      response.writeHead(416, { "Content-Range": `bytes */${size}` });
      response.end();
      return;
    }
    const start = Number(match[1]);
    const end = match[2] ? Math.min(Number(match[2]), size - 1) : size - 1;
    response.writeHead(206, {
      "Content-Type": "video/mp4",
      "Accept-Ranges": "bytes",
      "Content-Range": `bytes ${start}-${end}/${size}`,
      "Content-Length": end - start + 1,
      "Cache-Control": "no-store",
    });
    createReadStream(outputPath, { start, end }).pipe(response);
    return;
  }
  response.writeHead(200, {
    "Content-Type": "video/mp4",
    "Accept-Ranges": "bytes",
    "Content-Length": size,
    "Cache-Control": "no-store",
  });
  createReadStream(outputPath).pipe(response);
}

function generateDatasetVideo(dataset, episodeIndex, fps) {
  const outputDirectory = join(datasetMediaDirectory, dataset.id);
  mkdirSync(outputDirectory, { recursive: true });
  const outputPath = join(outputDirectory, `${episodeStem(episodeIndex)}.mp4`);
  const generationKey = `${dataset.id}:${episodeIndex}`;
  if (existsSync(outputPath) && statSync(outputPath).size > 0) return Promise.resolve(outputPath);
  if (generatingVideos.has(generationKey)) return generatingVideos.get(generationKey);
  const generation = new Promise((resolveVideo, rejectVideo) => {
    const child = spawn("python", [extractScript, parquetPath(dataset, episodeIndex), outputPath, String(fps)], { windowsHide: true });
    let errorOutput = "";
    child.stdout.on("data", () => {});
    child.stderr.on("data", (chunk) => { errorOutput += chunk.toString(); });
    child.on("error", (error) => rejectVideo(new Error(`Could not start the episode extractor: ${error.message}`)));
    child.on("close", (code) => {
      if (code === 0 && existsSync(outputPath)) resolveVideo(outputPath);
      else rejectVideo(new Error(errorOutput.trim() || `Episode extraction failed with code ${code}.`));
    });
  }).finally(() => generatingVideos.delete(generationKey));
  generatingVideos.set(generationKey, generation);
  return generation;
}

function inspectEpisode(dataset, episodeIndex) {
  const cacheKey = `${dataset.id}:${episodeIndex}`;
  if (episodeMetadataCache.has(cacheKey)) return Promise.resolve(episodeMetadataCache.get(cacheKey));
  const parquet = parquetPath(dataset, episodeIndex);
  const metadataProcess = spawn("python", [extractScript, "--inspect", parquet], { windowsHide: true });
  let metadataOutput = "";
  let metadataError = "";
  metadataProcess.stdout.on("data", (chunk) => { metadataOutput += chunk.toString(); });
  metadataProcess.stderr.on("data", (chunk) => { metadataError += chunk.toString(); });
  return new Promise((resolveMetadata, rejectMetadata) => {
    metadataProcess.on("error", rejectMetadata);
    metadataProcess.on("close", (code) => {
      if (code === 0) {
        const metadata = JSON.parse(metadataOutput);
        episodeMetadataCache.set(cacheKey, metadata);
        resolveMetadata(metadata);
      }
      else rejectMetadata(new Error(metadataError || "Could not inspect episode."));
    });
  });
}

async function buildEpisodeResponsePayload(sessionSummary) {
  if (sessionSummary.active_episode_index === null || !sessionSummary.active_dataset_id || !sessionSummary.active_category_id) {
    return { session: sessionSummary, complete: true };
  }
  const dataset = datasetById(sessionSummary.active_dataset_id);
  const category = categoryForDataset(dataset, sessionSummary.active_category_id);
  const info = readDatasetInfo(dataset);
  const fps = Number(info.fps || dataset.fpsFallback);
  const outputPath = await generateDatasetVideo(dataset, sessionSummary.active_episode_index, fps);
  const metadata = await inspectEpisode(dataset, sessionSummary.active_episode_index);
  const parquet = parquetPath(dataset, sessionSummary.active_episode_index);
  return {
    session: sessionSummary,
    complete: false,
    episode: {
      dataset_id: dataset.id,
      dataset_label: dataset.label,
      dataset_root: dataset.root,
      episode_index: sessionSummary.active_episode_index,
      episode_stem: episodeStem(sessionSummary.active_episode_index),
      category_id: category.id,
      category_label: category.label,
      parquet_path: parquet,
      checkpoint_path: checkpointPath(dataset, sessionSummary.active_episode_index),
      video_url: `http://127.0.0.1:3001/dataset-media/${dataset.id}/${episodeStem(sessionSummary.active_episode_index)}.mp4`,
      video_size: statSync(outputPath).size,
      frame_count: metadata.frame_count,
      width: metadata.width,
      height: metadata.height,
      fps,
    },
  };
}

function preparedEpisodeKey(sessionSummary) {
  if (!sessionSummary || sessionSummary.active_episode_index === null || !sessionSummary.active_dataset_id || !sessionSummary.active_category_id) {
    return null;
  }
  return `${sessionSummary.session_id}:${sessionSummary.active_dataset_id}:${sessionSummary.active_category_id}:${sessionSummary.active_episode_index}`;
}

function prefetchPreparedEpisode(sessionSummary) {
  const key = preparedEpisodeKey(sessionSummary);
  if (!key) return null;
  const existing = preparedEpisodes.get(key);
  if (existing?.status === "ready") return existing;
  if (existing?.status === "pending") {
    const ageMs = Date.now() - (existing.started_at_ms || Date.parse(existing.started_at || "") || Date.now());
    if (ageMs < 120000) return existing;
    preparedEpisodes.delete(key);
  }
  if (existing?.status === "failed") {
    const ageMs = Date.now() - (existing.finished_at_ms || Date.parse(existing.finished_at || "") || Date.now());
    if (ageMs < 5000) return existing;
    preparedEpisodes.delete(key);
  }
  const entry = {
    status: "pending",
    value: null,
    error: null,
    started_at: new Date().toISOString(),
    started_at_ms: Date.now(),
    finished_at: null,
    finished_at_ms: null,
    promise: null,
  };
  entry.promise = buildEpisodeResponsePayload(sessionSummary)
    .then((payload) => {
      entry.status = "ready";
      entry.value = payload;
      return payload;
    })
    .catch((error) => {
      entry.status = "failed";
      entry.error = error instanceof Error ? error.message : String(error);
      entry.finished_at = new Date().toISOString();
      entry.finished_at_ms = Date.now();
      console.error(`FrameLine prepared episode failed for ${key}: ${entry.error}`);
      throw error;
    });
  preparedEpisodes.set(key, entry);
  return entry;
}

function getPreparedEpisodeIfReady(sessionSummary) {
  const key = preparedEpisodeKey(sessionSummary);
  if (!key) return null;
  const entry = preparedEpisodes.get(key);
  if (entry?.status === "ready") return entry.value;
  return null;
}

function cloneSessionForPreview(session) {
  return JSON.parse(JSON.stringify(session));
}

function markEpisodeCompletedInSession(session, datasetId, categoryId, episodeIndex) {
  const datasetSession = session.datasets.find((candidate) => candidate.id === datasetId);
  if (!datasetSession) return null;
  const category = datasetSession.categories.find((candidate) => candidate.id === categoryId);
  if (!category) return null;
  const alreadyCompleted = category.completed_episode_indices.includes(episodeIndex);
  if (!category.completed_episode_indices.includes(episodeIndex)) {
    category.completed_episode_indices.push(episodeIndex);
  }
  if (!alreadyCompleted) flipNextDatasetForCategory(session, categoryId, datasetId);
  session.updated_at = new Date().toISOString();
  session.status = publicSession(session).status;
  return session;
}

function nextSessionSummaryAfterCompletion(session, datasetId, categoryId, episodeIndex) {
  const preview = cloneSessionForPreview(session);
  if (!markEpisodeCompletedInSession(preview, datasetId, categoryId, episodeIndex)) return null;
  return publicSession(preview);
}

function prefetchEpisodeForSummary(sessionSummary) {
  const entry = prefetchPreparedEpisode(sessionSummary);
  if (entry) void entry.promise.catch(() => {});
}

function buildFutureSummaries(session, completedEpisodes, count) {
  const preview = cloneSessionForPreview(session);
  for (const completion of completedEpisodes) {
    markEpisodeCompletedInSession(
      preview,
      String(completion.dataset_id || ""),
      String(completion.category_id || ""),
      Number(completion.episode_index),
    );
  }

  const summaries = [];
  for (let index = 0; index < count; index += 1) {
    const summary = publicSession(preview);
    if (summary.status === "complete" || summary.active_episode_index === null || !summary.active_dataset_id || !summary.active_category_id) {
      break;
    }
    summaries.push(summary);
    markEpisodeCompletedInSession(
      preview,
      summary.active_dataset_id,
      summary.active_category_id,
      summary.active_episode_index,
    );
  }
  return summaries;
}

const helper = createServer(async (request, response) => {
  const url = new URL(request.url || "/", "http://127.0.0.1:3001");
  if (request.method === "OPTIONS") {
    cors(response);
    response.writeHead(204);
    response.end();
    return;
  }

  const convertedMediaMatch = /^\/media\/([a-f0-9-]+)\.mp4$/.exec(url.pathname);
  if (request.method === "GET" && convertedMediaMatch) {
    serveMp4(request, response, join(mediaDirectory, `${convertedMediaMatch[1]}.mp4`));
    return;
  }

  const datasetMediaMatch = /^\/dataset-media\/([a-z_]+)\/(episode_\d{6})\.mp4$/.exec(url.pathname);
  if (request.method === "GET" && datasetMediaMatch) {
    const dataset = datasetById(datasetMediaMatch[1]);
    serveMp4(request, response, join(datasetMediaDirectory, dataset.id, `${datasetMediaMatch[2]}.mp4`));
    return;
  }

  if (request.method === "GET" && url.pathname === "/dataset/info") {
    try {
      const datasets = DATASET_DEFINITIONS.map(publicDatasetInfo);
      json(response, 200, {
        available: true,
        root: datasets.map((dataset) => `${dataset.label}: ${dataset.root}`).join(" | "),
        datasets,
        total_episodes: datasets.reduce((sum, dataset) => sum + dataset.total_episodes, 0),
        fps: datasets[0]?.fps,
        categories: CATEGORY_DEFINITIONS.map((definition) => ({
          ...definition,
          total_episodes: datasets.reduce((sum, dataset) => {
            const category = dataset.categories.find((candidate) => candidate.id === definition.id);
            return sum + Number(category?.total_episodes || 0);
          }, 0),
        })),
        active_session: latestActiveSession(),
        checkpoint_directory: DATASET_DEFINITIONS.map((dataset) => `${dataset.label}: ${datasetCheckpointsDirectory(dataset)}`).join(" | "),
      });
    } catch (error) {
      json(response, 404, {
        available: false,
        root: DATASET_DEFINITIONS.map((dataset) => `${dataset.label}: ${dataset.root}`).join(" | "),
        error: error instanceof Error ? error.message : "Dataset is unavailable.",
      });
    }
    return;
  }

  if (request.method === "POST" && url.pathname === "/dataset/session") {
    try {
      const existing = latestActiveSession();
      const session = existing ? loadSession(existing.session_id) : createSamplingSession();
      json(response, 200, { session: publicSession(session), resumed: Boolean(existing) });
    } catch (error) {
      json(response, 500, { error: error instanceof Error ? error.message : "Could not create the sampling session." });
    }
    return;
  }

  if (request.method === "GET" && url.pathname === "/dataset/episode") {
    try {
      const sessionId = url.searchParams.get("session_id") || "";
      const session = loadSession(sessionId);
      const summary = publicSession(session);
      const payload = getPreparedEpisodeIfReady(summary) || await buildEpisodeResponsePayload(summary);
      if (!payload.complete && payload.episode) {
        const nextSummary = nextSessionSummaryAfterCompletion(
          session,
          payload.episode.dataset_id,
          payload.episode.category_id,
          payload.episode.episode_index,
        );
        prefetchEpisodeForSummary(nextSummary);
      }
      json(response, 200, payload);
    } catch (error) {
      json(response, 500, { error: error instanceof Error ? error.message : "Could not load the next dataset episode." });
    }
    return;
  }

  if (request.method === "GET" && url.pathname === "/dataset/prepared-next") {
    try {
      const sessionId = url.searchParams.get("session_id") || "";
      const datasetId = url.searchParams.get("dataset_id") || "";
      const categoryId = url.searchParams.get("category_id") || "";
      const episodeIndex = Number(url.searchParams.get("episode_index"));
      const session = loadSession(sessionId);
      const nextSummary = nextSessionSummaryAfterCompletion(session, datasetId, categoryId, episodeIndex);
      if (!nextSummary || nextSummary.status === "complete" || nextSummary.active_episode_index === null) {
        json(response, 200, { session: publicSession(session), ready: true, complete: true });
        return;
      }
      const entry = prefetchPreparedEpisode(nextSummary);
      if (entry?.status === "pending") void entry.promise.catch(() => {});
      if (entry?.status === "ready" && entry.value) {
        const payload = entry.value;
        if (!payload.complete && payload.episode) {
          const previewSession = cloneSessionForPreview(session);
          const previewDatasetSession = previewSession.datasets.find((candidate) => candidate.id === datasetId);
          const previewCategory = previewDatasetSession?.categories.find((candidate) => candidate.id === categoryId);
          if (previewCategory && !previewCategory.completed_episode_indices.includes(episodeIndex)) {
            previewCategory.completed_episode_indices.push(episodeIndex);
          }
          flipNextDatasetForCategory(previewSession, categoryId, datasetId);
          const afterQueuedSummary = nextSessionSummaryAfterCompletion(
            previewSession,
            payload.episode.dataset_id,
            payload.episode.category_id,
            payload.episode.episode_index,
          );
          prefetchEpisodeForSummary(afterQueuedSummary);
        }
        json(response, 200, { ready: true, ...payload });
        return;
      }
      json(response, 200, {
        session: nextSummary,
        ready: false,
        complete: false,
        preparing: true,
        active_dataset_id: nextSummary.active_dataset_id,
        active_episode_index: nextSummary.active_episode_index,
      });
    } catch (error) {
      json(response, 500, { error: error instanceof Error ? error.message : "Could not prepare the next queued episode." });
    }
    return;
  }

  if (request.method === "POST" && url.pathname === "/dataset/prepared-queue") {
    try {
      const body = await readRequestJson(request);
      const session = loadSession(String(body.session_id || ""));
      const count = Math.max(1, Math.min(20, Number(body.count || 5)));
      const completedEpisodes = Array.isArray(body.completed_episodes) ? body.completed_episodes : [];
      const futureSummaries = buildFutureSummaries(session, completedEpisodes, count);
      const entries = futureSummaries.map((summary) => {
        const entry = prefetchPreparedEpisode(summary);
        if (entry?.status === "pending") void entry.promise.catch(() => {});
        return { summary, entry };
      });

      const ready = [];
      for (const { entry } of entries) {
        if (entry?.status === "ready" && entry.value && !entry.value.complete && entry.value.episode) {
          ready.push(entry.value);
          continue;
        }
        break;
      }

      const firstPending = entries.find(({ entry }) => entry?.status !== "ready");
      const failed = entries
        .filter(({ entry }) => entry?.status === "failed")
        .map(({ summary, entry }) => ({
          dataset_id: summary.active_dataset_id,
          episode_index: summary.active_episode_index,
          error: entry.error,
        }));
      json(response, 200, {
        session: publicSession(session),
        requested_count: count,
        future_count: futureSummaries.length,
        ready_count: ready.length,
        preparing_count: Math.max(0, futureSummaries.length - ready.length),
        failed_count: failed.length,
        failed,
        status: ready.length >= Math.min(count, futureSummaries.length) ? "ready" : "preparing",
        first_pending: firstPending?.summary ? {
          dataset_id: firstPending.summary.active_dataset_id,
          episode_index: firstPending.summary.active_episode_index,
        } : null,
        episodes: ready,
      });
    } catch (error) {
      json(response, 500, { error: error instanceof Error ? error.message : "Could not prepare the queued episodes." });
    }
    return;
  }

  if (request.method === "POST" && url.pathname === "/dataset/checkpoint") {
    try {
      const body = await readRequestJson(request);
      const session = loadSession(String(body.session_id || ""));
      const dataset = datasetById(String(body.dataset_id || ""));
      const datasetSession = session.datasets.find((candidate) => candidate.id === dataset.id);
      if (!datasetSession) throw new Error("This dataset is not part of the current sampling session.");
      const episodeIndex = Number(body.episode_index);
      const category = datasetSession.categories.find((candidate) => candidate.sampled_episode_indices.includes(episodeIndex));
      if (!category) throw new Error("This episode is not part of the current sampling session.");
      const expectedFrameCount = Number.isInteger(Number(body.expected_frame_count))
        ? Number(body.expected_frame_count)
        : (await inspectEpisode(dataset, episodeIndex)).frame_count;
      validateCheckpointPayload(body.checkpoints, expectedFrameCount);
      const output = {
        ...body.checkpoints,
        dataset_annotation: {
          dataset_id: dataset.id,
          dataset_label: dataset.label,
          dataset_root: dataset.root,
          episode_index: episodeIndex,
          episode_file: parquetPath(dataset, episodeIndex),
          garment_category: category.id,
          sampling_session_id: session.session_id,
          saved_at: new Date().toISOString(),
        },
      };
      const outputPath = checkpointPath(dataset, episodeIndex);
      writeJsonAtomic(outputPath, output);

      const refreshed = refreshSession(session);
      flipNextDatasetForCategory(refreshed, category.id, dataset.id);
      writeJsonAtomic(sessionPath(refreshed.session_id), refreshed);
      const summary = publicSession(refreshed);
      if (body.return_next) {
        const payload = getPreparedEpisodeIfReady(summary);
        if (payload && !payload.complete && payload.episode) {
          const nextSummary = nextSessionSummaryAfterCompletion(
            refreshed,
            payload.episode.dataset_id,
            payload.episode.category_id,
            payload.episode.episode_index,
          );
          prefetchEpisodeForSummary(nextSummary);
          json(response, 200, { saved: true, path: outputPath, ...payload });
        } else if (summary.status === "complete" || summary.active_episode_index === null) {
          json(response, 200, { saved: true, path: outputPath, session: summary, complete: true });
        } else {
          prefetchEpisodeForSummary(summary);
          json(response, 200, {
            saved: true,
            path: outputPath,
            session: summary,
            complete: false,
            preparing_next: true,
          });
        }
      } else {
        prefetchEpisodeForSummary(summary);
        json(response, 200, { saved: true, path: outputPath, session: summary });
      }
    } catch (error) {
      json(response, 400, { error: error instanceof Error ? error.message : "Could not save the checkpoint JSON." });
    }
    return;
  }

  if (request.method === "POST" && url.pathname === "/transcode") {
    const id = randomUUID();
    const inputPath = join(mediaDirectory, `${id}.source`);
    const outputPath = join(mediaDirectory, `${id}.mp4`);
    const upload = createWriteStream(inputPath);
    request.pipe(upload);
    upload.on("error", () => json(response, 500, { error: "Could not read the dropped file." }));
    upload.on("finish", () => {
      const ffmpeg = spawn(
        "ffmpeg",
        [
          "-hide_banner", "-loglevel", "error", "-y", "-i", inputPath,
          "-map", "0:v:0", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
          "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", outputPath,
        ],
        { windowsHide: true },
      );
      let errorOutput = "";
      ffmpeg.stderr.on("data", (chunk) => { errorOutput += chunk.toString(); });
      ffmpeg.on("error", () => json(response, 500, { error: "FFmpeg is not available. Install FFmpeg and restart FrameLine." }));
      ffmpeg.on("close", (code) => {
        if (code !== 0) json(response, 500, { error: errorOutput.trim() || "Video conversion failed." });
        else json(response, 200, { url: `http://127.0.0.1:3001/media/${id}.mp4` });
      });
    });
    return;
  }

  json(response, 404, { error: "Not found." });
});

helper.listen(3001, "127.0.0.1", () => {
  console.log("FrameLine local helper ready on http://127.0.0.1:3001");
  for (const dataset of DATASET_DEFINITIONS) {
    console.log(`FrameLine ${dataset.label} dataset root: ${dataset.root}`);
  }
});

const command = process.platform === "win32" ? "npx.cmd" : "npx";
const site = spawn(command, ["vinext", mode], {
  stdio: "inherit",
  shell: process.platform === "win32",
});

function stop(exitCode = 0) {
  helper.close(() => process.exit(exitCode));
  if (!site.killed) site.kill();
}

site.on("exit", (code) => stop(code ?? 0));
process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));
