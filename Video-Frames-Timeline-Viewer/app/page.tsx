"use client";

import {
  ChangeEvent,
  DragEvent,
  PointerEvent as ReactPointerEvent,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { createFile } from "mp4box";

type ParsedVideo = {
  fps: number;
  totalFrames: number;
  width: number;
  height: number;
  codec: string;
};

type TranscodeResponse = {
  url?: string;
  error?: string;
};

type DatasetCategoryInfo = {
  id: string;
  label: string;
  start: number;
  endExclusive: number;
  total_episodes: number;
};

type DatasetCategoryProgress = {
  id: string;
  label: string;
  total_episodes: number;
  eligible_at_sampling: number;
  target_sample_count: number;
  sampled_count: number;
  completed_count: number;
  remaining_count: number;
};

type DatasetProgress = {
  id: string;
  label: string;
  root: string;
  total_episodes: number;
  categories: DatasetCategoryProgress[];
  sampled_count: number;
  completed_count: number;
};

type DatasetSourceInfo = {
  id: string;
  label: string;
  root: string;
  total_episodes: number;
  fps?: number;
  categories: DatasetCategoryInfo[];
};

type DatasetSession = {
  session_id: string;
  created_at: string;
  sample_fraction: number;
  status: "active" | "complete";
  datasets: DatasetProgress[];
  categories: DatasetCategoryProgress[];
  active_dataset_id: string | null;
  active_dataset_label: string | null;
  active_category_id: string | null;
  active_category_label: string | null;
  active_episode_index: number | null;
  sampled_count: number;
  completed_count: number;
};

type DatasetInfo = {
  available: boolean;
  root: string;
  total_episodes?: number;
  fps?: number;
  datasets?: DatasetSourceInfo[];
  categories?: DatasetCategoryInfo[];
  active_session?: DatasetSession | null;
  checkpoint_directory?: string;
  error?: string;
};

type DatasetEpisode = {
  dataset_id: string;
  dataset_label: string;
  dataset_root: string;
  episode_index: number;
  episode_stem: string;
  category_id: string;
  category_label: string;
  parquet_path: string;
  checkpoint_path: string;
  video_url: string;
  video_size: number;
  frame_count: number;
  width: number;
  height: number;
  fps: number;
};

type SaveStatus = {
  state: "idle" | "saving" | "saved" | "failed";
  message: string;
  path?: string;
};

type PreparedQueueItem = {
  session: DatasetSession;
  episode: DatasetEpisode;
  browser_cached?: boolean;
};

type PreparedQueueResponse = {
  session?: DatasetSession;
  requested_count?: number;
  future_count?: number;
  ready_count?: number;
  preparing_count?: number;
  failed_count?: number;
  failed?: Array<{ dataset_id?: string; episode_index?: number; error?: string }>;
  status?: "ready" | "preparing";
  episodes?: PreparedQueueItem[];
  error?: string;
};

type GarmentLabelTemplate = AnnotationDocument & {
  category_id?: string;
};

type AnnotationSegment = {
  segment_id: number;
  label: string;
  start_frame: number;
  end_frame_exclusive: number;
  end_frame_inclusive: number;
  num_frames: number;
  notes?: string;
};

type SegmentProgress = {
  method: "piecewise_linear_direction_toggles" | "piecewise_linear_control_points";
  range: [0, 1];
  initial_direction: "increasing";
  slope_per_frame: number | null;
  control_points: Array<{ frame: number; progress: number }>;
  linear_pieces: Array<{
    start_frame: number;
    end_frame_inclusive: number;
    start_progress: number;
    end_progress: number;
    slope_per_frame: number;
  }>;
  direction_changes: Array<{
    frame: number;
    progress: number;
    direction_after: "increasing" | "decreasing";
  }>;
  end_progress: number;
  per_frame: Array<{ frame: number; progress: number }>;
};

type AnnotationDocument = Record<string, unknown> & {
  labels?: unknown;
  segments?: unknown;
};

type AnnotationHistoryEntry = {
  segments: AnnotationSegment[];
  activeSegmentIndex: number;
  progressToggles: Record<number, number[]>;
  progressControlPoints: Record<number, Array<{ frame: number; progress: number }>>;
};

type LocalSavePicker = (options: {
  suggestedName: string;
  types: Array<{ description: string; accept: Record<string, string[]> }>;
}) => Promise<{ createWritable: () => Promise<{ write: (value: Blob) => Promise<void>; close: () => Promise<void> }> }>;

const ACCEPTED_EXTENSIONS = [".mp4", ".m4v", ".mov", ".webm"];
const SEGMENT_COLORS = ["#6ee7ff", "#a78bfa", "#fb7185", "#fbbf24", "#34d399", "#60a5fa", "#f472b6", "#a3e635", "#fb923c", "#22d3ee"];
const LOCAL_HELPER = "http://127.0.0.1:3001";
const ANNOTATION_CATEGORY_IDS = new Set(["shorts", "top_long_sleeve", "top_short_sleeve"]);
const PREPARED_QUEUE_TARGET = 5;

function clamp(value: number, minimum: number, maximum: number) {
  return Math.min(maximum, Math.max(minimum, value));
}

function preparedQueueKey(item: PreparedQueueItem) {
  return `${item.episode.dataset_id}:${item.episode.category_id}:${item.episode.episode_index}`;
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatClock(seconds: number) {
  if (!Number.isFinite(seconds)) return "00:00:00.000";
  const safe = Math.max(0, seconds);
  const hours = Math.floor(safe / 3600);
  const minutes = Math.floor((safe % 3600) / 60);
  const wholeSeconds = Math.floor(safe % 60);
  const milliseconds = Math.floor((safe - Math.floor(safe)) * 1000);
  return [hours, minutes, wholeSeconds]
    .map((part) => String(part).padStart(2, "0"))
    .join(":") + `.${String(milliseconds).padStart(3, "0")}`;
}

function formatTimecode(frame: number, fps: number) {
  const nominalFps = Math.max(1, Math.round(fps));
  const safeFrame = Math.max(0, Math.round(frame));
  const frames = safeFrame % nominalFps;
  const totalSeconds = Math.floor(safeFrame / nominalFps);
  const seconds = totalSeconds % 60;
  const minutes = Math.floor(totalSeconds / 60) % 60;
  const hours = Math.floor(totalSeconds / 3600);
  return [hours, minutes, seconds, frames]
    .map((part) => String(part).padStart(2, "0"))
    .join(":");
}

function makeSegment(index: number, label: string, start: number, end: number, notes?: string): AnnotationSegment {
  return {
    segment_id: index,
    label,
    start_frame: start,
    end_frame_exclusive: end,
    end_frame_inclusive: end - 1,
    num_frames: end - start,
    ...(notes ? { notes } : {}),
  };
}

function buildSegmentProgress(
  segment: AnnotationSegment,
  rawToggleFrames: number[],
  rawControlPoints: Array<{ frame: number; progress: number }> = [],
): SegmentProgress {
  const controlPointMap = new Map<number, number>();
  rawControlPoints.forEach((point) => {
    if (Number.isInteger(point.frame) && point.frame > segment.start_frame && point.frame < segment.end_frame_inclusive) {
      controlPointMap.set(point.frame, clamp(Number(point.progress), 0, 1));
    }
  });
  const interiorControlPoints = [...controlPointMap.entries()]
    .map(([frame, progress]) => ({ frame, progress: Number(progress.toFixed(6)) }))
    .sort((left, right) => left.frame - right.frame);

  if (interiorControlPoints.length) {
    const controlPoints = [
      { frame: segment.start_frame, progress: 0 },
      ...interiorControlPoints,
      { frame: segment.end_frame_inclusive, progress: 1 },
    ];
    const linearPieces = controlPoints.slice(0, -1).map((point, index) => {
      const next = controlPoints[index + 1];
      return {
        start_frame: point.frame,
        end_frame_inclusive: next.frame,
        start_progress: point.progress,
        end_progress: next.progress,
        slope_per_frame: Number(((next.progress - point.progress) / Math.max(1, next.frame - point.frame)).toFixed(9)),
      };
    });
    const perFrame: SegmentProgress["per_frame"] = [];
    linearPieces.forEach((piece, pieceIndex) => {
      const lastFrame = pieceIndex === linearPieces.length - 1 ? piece.end_frame_inclusive : piece.end_frame_inclusive - 1;
      for (let frame = piece.start_frame; frame <= lastFrame; frame += 1) {
        const ratio = (frame - piece.start_frame) / Math.max(1, piece.end_frame_inclusive - piece.start_frame);
        const progress = piece.start_progress + ratio * (piece.end_progress - piece.start_progress);
        perFrame.push({ frame, progress: Number(progress.toFixed(6)) });
      }
    });
    return {
      method: "piecewise_linear_control_points",
      range: [0, 1],
      initial_direction: "increasing",
      slope_per_frame: null,
      control_points: controlPoints,
      linear_pieces: linearPieces,
      direction_changes: interiorControlPoints.map((point, index) => ({
        frame: point.frame,
        progress: point.progress,
        direction_after: linearPieces[index + 1].slope_per_frame >= 0 ? "increasing" : "decreasing",
      })),
      end_progress: 1,
      per_frame: perFrame,
    };
  }

  const toggleFrames = [...new Set(rawToggleFrames)]
    .filter((frame) => Number.isInteger(frame) && frame >= segment.start_frame && frame < segment.end_frame_inclusive)
    .sort((left, right) => left - right);
  const toggleSet = new Set(toggleFrames);
  const slope = segment.num_frames > 1 ? 1 / (segment.num_frames - 1) : 0;
  const perFrame: Array<{ frame: number; progress: number }> = [];
  const directionChanges: SegmentProgress["direction_changes"] = [];
  let value = 0;
  let direction: 1 | -1 = 1;

  for (let frame = segment.start_frame; frame <= segment.end_frame_inclusive; frame += 1) {
    perFrame.push({ frame, progress: Number(value.toFixed(6)) });
    if (toggleSet.has(frame)) {
      direction *= -1;
      directionChanges.push({
        frame,
        progress: Number(value.toFixed(6)),
        direction_after: direction === 1 ? "increasing" : "decreasing",
      });
    }
    value = clamp(value + direction * slope, 0, 1);
  }

  const toggleAnchors = [
    { frame: segment.start_frame, progress: 0 },
    ...directionChanges.map(({ frame, progress }) => ({ frame, progress })),
    { frame: segment.end_frame_inclusive, progress: perFrame.at(-1)?.progress ?? 0 },
  ].filter((point, index, points) => index === 0 || point.frame !== points[index - 1].frame);
  const linearPieces = toggleAnchors.slice(0, -1).map((point, index) => {
    const next = toggleAnchors[index + 1];
    return {
      start_frame: point.frame,
      end_frame_inclusive: next.frame,
      start_progress: point.progress,
      end_progress: next.progress,
      slope_per_frame: Number(((next.progress - point.progress) / Math.max(1, next.frame - point.frame)).toFixed(9)),
    };
  });

  return {
    method: "piecewise_linear_direction_toggles",
    range: [0, 1],
    initial_direction: "increasing",
    slope_per_frame: Number(slope.toFixed(9)),
    control_points: toggleAnchors,
    linear_pieces: linearPieces,
    direction_changes: directionChanges,
    end_progress: perFrame.at(-1)?.progress ?? 0,
    per_frame: perFrame,
  };
}

async function parseMp4(file: File): Promise<ParsedVideo | null> {
  if (!file.name.toLowerCase().match(/\.(mp4|m4v|mov)$/)) return null;
  const parser = createFile();
  const buffer = (await file.arrayBuffer()) as ArrayBuffer & { fileStart: number };
  buffer.fileStart = 0;

  return new Promise((resolve, reject) => {
    parser.onError = (error) => reject(new Error(String(error)));
    parser.onReady = (info) => {
      const videoTrack = info.videoTracks?.[0] ?? info.tracks.find((track) => Boolean(track.video));
      if (!videoTrack || !videoTrack.nb_samples || !videoTrack.duration) {
        resolve(null);
        return;
      }
      const durationSeconds = videoTrack.duration / videoTrack.timescale;
      resolve({
        fps: videoTrack.nb_samples / durationSeconds,
        totalFrames: videoTrack.nb_samples,
        width: videoTrack.video?.width ?? videoTrack.track_width,
        height: videoTrack.video?.height ?? videoTrack.track_height,
        codec: videoTrack.codec,
      });
    };
    parser.appendBuffer(buffer);
    parser.flush();
  });
}

export default function Home() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const viewerRef = useRef<HTMLElement>(null);
  const progressCanvasRef = useRef<HTMLCanvasElement>(null);
  const progressTrackRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const annotationInputRef = useRef<HTMLInputElement>(null);
  const currentFrameRef = useRef(0);
  const holdTimerRef = useRef<number | null>(null);
  const activeArrowRef = useRef<string | null>(null);
  const sourceFileRef = useRef<File | null>(null);
  const transcodeInFlightRef = useRef(false);
  const annotationDragRef = useRef<{ segmentIndex: number; startFrame: number } | null>(null);
  const progressDragRef = useRef<{
    segmentIndex: number;
    originFrame: number;
    originProgress: number;
    originClientX: number;
    originClientY: number;
    moved: boolean;
  } | null>(null);
  const annotationHistoryRef = useRef<AnnotationHistoryEntry[]>([]);
  const annotationRedoRef = useRef<AnnotationHistoryEntry[]>([]);
  const queuedEpisodePollRef = useRef<number | null>(null);
  const preparedEpisodeQueueRef = useRef<PreparedQueueItem[]>([]);
  const preparedVideoCacheRef = useRef<Map<string, Promise<PreparedQueueItem>>>(new Map());
  const pendingCompletedEpisodesRef = useRef<Array<{ dataset_id: string; category_id: string; episode_index: number }>>([]);
  const queueRequestIdRef = useRef(0);

  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [fileSize, setFileSize] = useState(0);
  const [duration, setDuration] = useState(0);
  const [fps, setFps] = useState(30);
  const [totalFrames, setTotalFrames] = useState(0);
  const [currentFrame, setCurrentFrame] = useState(0);
  const [currentTime, setCurrentTime] = useState(0);
  const [resolution, setResolution] = useState("—");
  const [codec, setCodec] = useState("—");
  const [isPlaying, setIsPlaying] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [isScrubbing, setIsScrubbing] = useState(false);
  const [hoverFrame, setHoverFrame] = useState<number | null>(null);
  const [metadataStatus, setMetadataStatus] = useState("Drop a local video to begin");
  const [holdSpeed, setHoldSpeed] = useState(1);
  const [isPreparing, setIsPreparing] = useState(false);
  const [loadError, setLoadError] = useState("");
  const [videoDisplayScale, setVideoDisplayScale] = useState(75);
  const [annotationLabels, setAnnotationLabels] = useState<string[]>([]);
  const [annotationSegments, setAnnotationSegments] = useState<AnnotationSegment[]>([]);
  const [annotationBase, setAnnotationBase] = useState<AnnotationDocument | null>(null);
  const [activeSegmentIndex, setActiveSegmentIndex] = useState(0);
  const [interactionMode, setInteractionMode] = useState<"scrub" | "annotate" | "progress">("scrub");
  const [annotationStatus, setAnnotationStatus] = useState("Load a segment list or checkpoint JSON to begin annotation");
  const [annotationPreviewEnd, setAnnotationPreviewEnd] = useState<number | null>(null);
  const [progressToggles, setProgressToggles] = useState<Record<number, number[]>>({});
  const [progressControlPoints, setProgressControlPoints] = useState<Record<number, Array<{ frame: number; progress: number }>>>({});
  const [pendingProgressPoint, setPendingProgressPoint] = useState<{ segmentIndex: number; frame: number; progress: number; originalFrame?: number } | null>(null);
  const [datasetInfo, setDatasetInfo] = useState<DatasetInfo | null>(null);
  const [datasetSession, setDatasetSession] = useState<DatasetSession | null>(null);
  const [datasetEpisode, setDatasetEpisode] = useState<DatasetEpisode | null>(null);
  const [datasetBusy, setDatasetBusy] = useState(false);
  const [datasetMessage, setDatasetMessage] = useState("Checking the configured human and sim LeRobot datasets…");
  const [datasetCheckpointSaved, setDatasetCheckpointSaved] = useState(false);
  const [previousSaveStatus, setPreviousSaveStatus] = useState<SaveStatus>({ state: "idle", message: "No episode saved yet" });
  const [queuedEpisodeStatus, setQueuedEpisodeStatus] = useState("Next episode queue idle");
  const [preparedEpisodeQueue, setPreparedEpisodeQueue] = useState<PreparedQueueItem[]>([]);

  const maxFrame = Math.max(0, totalFrames - 1);
  const progress = maxFrame > 0 ? currentFrame / maxFrame : 0;

  useEffect(() => {
    currentFrameRef.current = currentFrame;
  }, [currentFrame]);

  useEffect(() => () => {
    if (queuedEpisodePollRef.current !== null) window.clearTimeout(queuedEpisodePollRef.current);
  }, []);

  useEffect(() => {
    preparedEpisodeQueueRef.current = preparedEpisodeQueue;
  }, [preparedEpisodeQueue]);

  const timelineTicks = useMemo(() => {
    if (!totalFrames) return [];
    return Array.from({ length: 9 }, (_, index) => {
      const ratio = index / 8;
      const frame = Math.round(ratio * maxFrame);
      return { ratio, frame, time: frame / fps };
    });
  }, [fps, maxFrame, totalFrames]);

  useEffect(() => {
    let cancelled = false;
    void fetch(`${LOCAL_HELPER}/dataset/info`)
      .then(async (response) => {
        const result = await response.json() as DatasetInfo;
        if (!response.ok || !result.available) throw new Error(result.error || "The configured dataset is unavailable.");
        if (cancelled) return;
        setDatasetInfo(result);
        setDatasetSession(result.active_session ?? null);
        setDatasetMessage(result.active_session
          ? "An unfinished annotation session is ready to resume."
          : "Datasets ready. Start a session to sample 25% of Shorts, Top Long sleeve, and Top Short sleeve.");
      })
      .catch((error) => {
        if (cancelled) return;
        setDatasetInfo({ available: false, root: "Human: D:\\pretrain_lehome_all_garment_data_z180 | Sim: E:\\Lehome-Dataset\\lehome_round_2_dataset\\sim_dataset\\robot_sim_ft_lehome_all_garment_data_z180", error: error instanceof Error ? error.message : "Dataset helper unavailable." });
        setDatasetMessage("Start FrameLine with npm run dev to enable dataset mode.");
      });
    return () => { cancelled = true; };
  }, []);

  const seekFrame = useCallback(
    (nextFrame: number) => {
      const video = videoRef.current;
      if (!video || !totalFrames) return;
      const target = clamp(Math.round(nextFrame), 0, maxFrame);
      video.pause();
      const targetTime = Math.min(duration || Number.POSITIVE_INFINITY, (target + 0.001) / fps);
      video.currentTime = targetTime;
      setCurrentFrame(target);
      setCurrentTime(target / fps);
      setIsPlaying(false);
    },
    [duration, fps, maxFrame, totalFrames],
  );

  const prepareCompatibleVideo = useCallback(async (file: File) => {
    if (transcodeInFlightRef.current) return;
    transcodeInFlightRef.current = true;
    setIsPreparing(true);
    setLoadError("");
    setMetadataStatus("Converting video for browser playback…");

    try {
      const response = await fetch("http://127.0.0.1:3001/transcode", {
        method: "POST",
        headers: {
          "Content-Type": "application/octet-stream",
          "X-File-Name": encodeURIComponent(file.name),
        },
        body: file,
      });
      const result = (await response.json()) as TranscodeResponse;
      if (!response.ok || !result.url) {
        throw new Error(result.error || "The local converter could not prepare this video.");
      }
      setVideoUrl(result.url);
      setMetadataStatus("Converted locally for browser playback");
    } catch (error) {
      const detail = error instanceof Error ? error.message : "Unknown conversion error";
      setLoadError(`This video's codec is not supported directly by the browser, and local conversion failed. ${detail}`);
      setMetadataStatus("Video could not be opened");
      setVideoUrl(null);
    } finally {
      transcodeInFlightRef.current = false;
      setIsPreparing(false);
    }
  }, []);

  const togglePlayback = useCallback(() => {
    const video = videoRef.current;
    if (!video || isPreparing) return;
    if (video.paused) {
      if (currentFrameRef.current >= maxFrame) seekFrame(0);
      void video.play().catch(() => {
        const sourceFile = sourceFileRef.current;
        if (sourceFile) void prepareCompatibleVideo(sourceFile);
      });
    } else {
      video.pause();
    }
  }, [isPreparing, maxFrame, prepareCompatibleVideo, seekFrame]);

  const clearArrowHold = useCallback(() => {
    if (holdTimerRef.current !== null) window.clearTimeout(holdTimerRef.current);
    holdTimerRef.current = null;
    activeArrowRef.current = null;
    setHoldSpeed(1);
  }, []);

  useEffect(() => {
    const startHold = (direction: -1 | 1, key: string) => {
      if (activeArrowRef.current) return;
      activeArrowRef.current = key;
      seekFrame(currentFrameRef.current + direction);
      const started = performance.now();

      const repeat = () => {
        const elapsed = performance.now() - started;
        const step = elapsed > 2600 ? 12 : elapsed > 1700 ? 6 : elapsed > 900 ? 3 : 1;
        setHoldSpeed(step);
        seekFrame(currentFrameRef.current + direction * step);
        holdTimerRef.current = window.setTimeout(repeat, elapsed > 900 ? 48 : 78);
      };
      holdTimerRef.current = window.setTimeout(repeat, 320);
    };

    const onKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement;
      if (["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
        event.preventDefault();
        if (!event.repeat) startHold(event.key === "ArrowLeft" ? -1 : 1, event.key);
      } else if (event.code === "Space" && interactionMode !== "annotate") {
        event.preventDefault();
        if (!event.repeat) togglePlayback();
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.key === activeArrowRef.current) clearArrowHold();
    };

    window.addEventListener("keydown", onKeyDown);
    window.addEventListener("keyup", onKeyUp);
    window.addEventListener("blur", clearArrowHold);
    return () => {
      window.removeEventListener("keydown", onKeyDown);
      window.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("blur", clearArrowHold);
      clearArrowHold();
    };
  }, [clearArrowHold, interactionMode, seekFrame, togglePlayback]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || !("requestVideoFrameCallback" in video)) return;
    let callbackId = 0;
    const update = (_now: DOMHighResTimeStamp, metadata: VideoFrameCallbackMetadata) => {
      const mediaTime = metadata.mediaTime;
      setCurrentTime(mediaTime);
      setCurrentFrame(clamp(Math.floor(mediaTime * fps + 0.001), 0, maxFrame));
      callbackId = video.requestVideoFrameCallback(update);
    };
    callbackId = video.requestVideoFrameCallback(update);
    return () => video.cancelVideoFrameCallback(callbackId);
  }, [fps, maxFrame, videoUrl]);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
    };
  }, [videoUrl]);

  const loadFile = useCallback(
    async (file: File) => {
      const lowerName = file.name.toLowerCase();
      if (!file.type.startsWith("video/") && !ACCEPTED_EXTENSIONS.some((ext) => lowerName.endsWith(ext))) {
        setMetadataStatus("That file is not a supported video");
        return;
      }

      sourceFileRef.current = file;
      setLoadError("");
      setIsPreparing(true);
      setMetadataStatus("Reading video frame metadata…");
      setIsPlaying(false);
      setCurrentFrame(0);
      setCurrentTime(0);
      setDuration(0);
      setFps(30);
      setTotalFrames(0);
      setResolution("—");
      setCodec("—");
      setFileName(file.name);
      setFileSize(file.size);
      setAnnotationLabels([]);
      setAnnotationSegments([]);
      setAnnotationBase(null);
      setActiveSegmentIndex(0);
      setInteractionMode("scrub");
      setAnnotationStatus("Load a segment list or checkpoint JSON to begin annotation");
      setAnnotationPreviewEnd(null);
      setProgressToggles({});
      setProgressControlPoints({});
      setPendingProgressPoint(null);
      annotationHistoryRef.current = [];
      annotationRedoRef.current = [];
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);

      try {
        const parsed = await parseMp4(file);
        if (parsed) {
          const detectedFps = Number(parsed.fps.toFixed(6));
          setFps(detectedFps);
          setTotalFrames(parsed.totalFrames);
          setResolution(`${parsed.width} × ${parsed.height}`);
          setCodec(parsed.codec.toUpperCase());
          setMetadataStatus(`Exact MP4 metadata · ${detectedFps.toFixed(3)} FPS`);

          const mimeType = file.type || "video/mp4";
          const codecSupport = document
            .createElement("video")
            .canPlayType(`${mimeType}; codecs="${parsed.codec}"`);
          if (!codecSupport) {
            await prepareCompatibleVideo(file);
            return;
          }
        } else {
          setMetadataStatus("Set FPS manually for exact frame indexing");
        }
      } catch {
        setMetadataStatus("Could not parse frame metadata · set FPS manually");
      }

      setVideoUrl(URL.createObjectURL(file));
      setIsPreparing(false);
    },
    [prepareCompatibleVideo, videoUrl],
  );

  const onLoadedMetadata = () => {
    const video = videoRef.current;
    if (!video) return;
    setDuration(video.duration);
    if (resolution === "—") setResolution(`${video.videoWidth} × ${video.videoHeight}`);
    if (!totalFrames) setTotalFrames(Math.max(1, Math.round(video.duration * fps)));
  };

  const onTimeUpdate = () => {
    const video = videoRef.current;
    if (!video) return;
    setCurrentTime(video.currentTime);
    setCurrentFrame(clamp(Math.floor(video.currentTime * fps + 0.001), 0, maxFrame));
  };

  const onVideoError = () => {
    const sourceFile = sourceFileRef.current;
    if (sourceFile && !transcodeInFlightRef.current) {
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);
      void prepareCompatibleVideo(sourceFile);
    }
  };

  const chooseFile = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void loadFile(file);
    event.target.value = "";
  };

  const onDrop = (event: DragEvent) => {
    event.preventDefault();
    setIsDraggingFile(false);
    const file = event.dataTransfer.files?.[0];
    if (file) void loadFile(file);
  };

  const loadAnnotationDocument = useCallback((data: AnnotationDocument, sourceName: string) => {
    const sourceSegments = Array.isArray(data.segments) ? data.segments as Array<Record<string, unknown>> : [];
    const sourceLabels = Array.isArray(data.labels)
      ? data.labels.filter((label): label is string => typeof label === "string" && label.trim().length > 0)
      : sourceSegments.map((segment) => String(segment.label ?? "")).filter(Boolean);
    if (!sourceLabels.length) {
      setAnnotationStatus(`${sourceName} does not contain a non-empty labels list`);
      return;
    }

    let importedSegments: AnnotationSegment[] = [];
    const importedProgressToggles: Record<number, number[]> = {};
    const importedProgressControlPoints: Record<number, Array<{ frame: number; progress: number }>> = {};
    if (sourceSegments.length) {
      try {
        let expectedStart = 0;
        importedSegments = sourceSegments.map((segment, index) => {
          const start = Number(segment.start_frame);
          const end = Number(segment.end_frame_exclusive);
          if (!Number.isInteger(start) || !Number.isInteger(end) || start !== expectedStart || end <= start) {
            throw new Error(`segment ${index} is not contiguous`);
          }
          const progress = typeof segment.progress === "object" && segment.progress
            ? segment.progress as Record<string, unknown>
            : null;
          const directionChanges = progress?.method !== "piecewise_linear_control_points" && progress && Array.isArray(progress.direction_changes)
            ? progress.direction_changes as Array<Record<string, unknown>>
            : [];
          const toggleFrames = directionChanges
            .map((change) => Number(change.frame))
            .filter((frame) => Number.isInteger(frame) && frame >= start && frame < end - 1);
          if (toggleFrames.length) importedProgressToggles[index] = toggleFrames;
          const sourceControlPoints = progress?.method === "piecewise_linear_control_points" && Array.isArray(progress.control_points)
            ? progress.control_points as Array<Record<string, unknown>>
            : [];
          const controlPoints = sourceControlPoints
            .map((point) => ({ frame: Number(point.frame), progress: Number(point.progress) }))
            .filter((point) => Number.isInteger(point.frame)
              && Number.isFinite(point.progress)
              && point.frame > start
              && point.frame < end - 1)
            .map((point) => ({ ...point, progress: clamp(point.progress, 0, 1) }));
          if (controlPoints.length) importedProgressControlPoints[index] = controlPoints;
          expectedStart = end;
          return makeSegment(index, sourceLabels[index] ?? String(segment.label), start, end, typeof segment.notes === "string" ? segment.notes : undefined);
        });
        if (importedSegments.length !== sourceLabels.length || expectedStart !== totalFrames) {
          throw new Error(`segment coverage must end at frame ${totalFrames}`);
        }
      } catch (error) {
        importedSegments = [];
        const detail = error instanceof Error ? error.message : "invalid segment boundaries";
        setAnnotationStatus(`Loaded ${sourceLabels.length} labels; existing boundaries were ignored because ${detail}`);
      }
    }

    setAnnotationLabels(sourceLabels);
    setAnnotationSegments(importedSegments);
    setAnnotationBase(data);
    setActiveSegmentIndex(importedSegments.length ? 0 : 0);
    setInteractionMode("annotate");
    setAnnotationPreviewEnd(null);
    setProgressToggles(importedSegments.length ? importedProgressToggles : {});
    setProgressControlPoints(importedSegments.length ? importedProgressControlPoints : {});
    setPendingProgressPoint(null);
    annotationHistoryRef.current = [];
    annotationRedoRef.current = [];
    if (importedSegments.length) {
      setAnnotationStatus(`Loaded ${sourceLabels.length} labels and ${importedSegments.length} contiguous segments from ${sourceName}`);
    } else if (!sourceSegments.length) {
      setAnnotationStatus(`Loaded ${sourceLabels.length} labels from ${sourceName}. Set segment 1, then press Enter to save it.`);
    }
  }, [totalFrames]);

  const chooseAnnotationJson = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;
    try {
      const data = JSON.parse(await file.text()) as AnnotationDocument;
      loadAnnotationDocument(data, file.name);
    } catch {
      setAnnotationStatus("Could not read that JSON file");
    }
  };

  const loadSampleLabels = async () => {
    try {
      const response = await fetch("/sample-segments.json");
      if (!response.ok) throw new Error();
      loadAnnotationDocument(await response.json() as AnnotationDocument, "sample-segments.json");
    } catch {
      setAnnotationStatus("Could not load the sample segment list");
    }
  };

  const applyDatasetEpisode = useCallback(async (episode: DatasetEpisode) => {
    if (!ANNOTATION_CATEGORY_IDS.has(episode.category_id)) {
      throw new Error(`This viewer is configured for Shorts, Top Long sleeve, and Top Short sleeve only. The helper returned '${episode.category_label}', so restart npm run dev to load the updated paired-dataset helper.`);
    }
    sourceFileRef.current = null;
    transcodeInFlightRef.current = false;
    if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
    setVideoUrl(null);
    setDatasetEpisode(episode);
    setDatasetCheckpointSaved(false);
    const episodeDatasetLabel = episode.dataset_label ?? episode.dataset_id ?? "dataset";
    setFileName(`${episodeDatasetLabel.toLowerCase()}_${episode.episode_stem}_top_rgb.mp4`);
    setFileSize(episode.video_size);
    setDuration(episode.frame_count / episode.fps);
    setFps(episode.fps);
    setTotalFrames(episode.frame_count);
    setCurrentFrame(0);
    setCurrentTime(0);
    setResolution(`${episode.width} Ã— ${episode.height}`);
    setCodec("H264");
    setIsPlaying(false);
    setMetadataStatus(`${episode.category_label} Â· dataset episode ${episode.episode_index}`);
    setAnnotationLabels([]);
    setAnnotationSegments([]);
    setAnnotationBase(null);
    setActiveSegmentIndex(0);
    setInteractionMode("scrub");
    setAnnotationStatus("Loading the category checkpoint vocabularyâ€¦");
    setAnnotationPreviewEnd(null);
    setProgressToggles({});
    setProgressControlPoints({});
    setPendingProgressPoint(null);
    annotationHistoryRef.current = [];
    annotationRedoRef.current = [];

    setMetadataStatus(`${episodeDatasetLabel} - ${episode.category_label} - dataset episode ${episode.episode_index}`);
    const templateResponse = await fetch("/garment-segment-labels.json");
    if (!templateResponse.ok) throw new Error("Could not load garment-segment-labels.json.");
    const templateDocument = await templateResponse.json() as { templates?: GarmentLabelTemplate[] };
    const template = templateDocument.templates?.find((candidate) => candidate.category_id === episode.category_id);
    if (!template) throw new Error(`No label template is configured for ${episode.category_label}.`);
    loadAnnotationDocument(template, `${episode.category_id} template`);
    setVideoUrl(episode.video_url);
    setDatasetMessage(`${episodeDatasetLabel} ${episode.category_label}: annotating episode ${String(episode.episode_index).padStart(6, "0")}.`);
    setIsPreparing(false);
  }, [loadAnnotationDocument, videoUrl]);

  const cachePreparedEpisodeVideo = useCallback((item: PreparedQueueItem) => {
    if (item.browser_cached || item.episode.video_url.startsWith("blob:")) {
      return Promise.resolve(item);
    }
    const key = preparedQueueKey(item);
    const existing = preparedVideoCacheRef.current.get(key);
    if (existing) return existing;

    const request = fetch(item.episode.video_url, { cache: "no-store" })
      .then(async (response) => {
        if (!response.ok) throw new Error(`Could not cache queued video (${response.status}).`);
        const videoBlob = await response.blob();
        return {
          ...item,
          browser_cached: true,
          episode: {
            ...item.episode,
            video_url: URL.createObjectURL(videoBlob),
            video_size: videoBlob.size,
          },
        };
      })
      .catch((error) => {
        preparedVideoCacheRef.current.delete(key);
        throw error;
      });
    preparedVideoCacheRef.current.set(key, request);
    return request;
  }, []);

  const refillPreparedEpisodeQueue = useCallback((session: DatasetSession | null, episode: DatasetEpisode | null) => {
    if (queuedEpisodePollRef.current !== null) window.clearTimeout(queuedEpisodePollRef.current);
    queuedEpisodePollRef.current = null;
    const requestId = queueRequestIdRef.current + 1;
    queueRequestIdRef.current = requestId;
    if (!session || !episode || session.status === "complete") {
      preparedEpisodeQueueRef.current = [];
      setPreparedEpisodeQueue([]);
      setQueuedEpisodeStatus("No next episodes to queue");
      return;
    }
    setQueuedEpisodeStatus(`Preparing next ${PREPARED_QUEUE_TARGET} episodes in background...`);
    const poll = async () => {
      try {
        const response = await fetch(`${LOCAL_HELPER}/dataset/prepared-queue`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: session.session_id,
            count: PREPARED_QUEUE_TARGET,
            completed_episodes: [
              ...pendingCompletedEpisodesRef.current,
              {
                dataset_id: episode.dataset_id,
                category_id: episode.category_id,
                episode_index: episode.episode_index,
              },
            ],
          }),
        });
        const result = await response.json() as PreparedQueueResponse;
        if (!response.ok) throw new Error(result.error || "Could not prepare the episode queue.");
        if ((result.failed_count ?? 0) > 0) {
          const failed = result.failed?.[0];
          throw new Error(failed?.error || "One queued episode failed to prepare.");
        }
        if (requestId !== queueRequestIdRef.current) return;
        const serverReadyEpisodes = result.episodes ?? [];
        if (serverReadyEpisodes.length > 0) {
          setQueuedEpisodeStatus(`Caching ${serverReadyEpisodes.length} prepared video${serverReadyEpisodes.length === 1 ? "" : "s"} in memory...`);
        }
        const readyEpisodes = await Promise.all(serverReadyEpisodes.map(cachePreparedEpisodeVideo));
        if (requestId !== queueRequestIdRef.current) return;
        const mergedQueue = [...preparedEpisodeQueueRef.current];
        const knownKeys = new Set(mergedQueue.map(preparedQueueKey));
        for (const item of readyEpisodes) {
          const key = preparedQueueKey(item);
          if (!knownKeys.has(key)) {
            mergedQueue.push(item);
            knownKeys.add(key);
          }
          if (mergedQueue.length >= PREPARED_QUEUE_TARGET) break;
        }
        preparedEpisodeQueueRef.current = mergedQueue;
        setPreparedEpisodeQueue(mergedQueue);
        if (mergedQueue.length > 0) {
          const firstEpisode = mergedQueue[0].episode;
          setQueuedEpisodeStatus(`${mergedQueue.length}/${PREPARED_QUEUE_TARGET} ready; next ${firstEpisode.dataset_label} episode ${String(firstEpisode.episode_index).padStart(6, "0")}`);
        } else {
          setQueuedEpisodeStatus(`Preparing queue: 0/${PREPARED_QUEUE_TARGET} ready`);
        }
        if (result.future_count === 0) {
          if (mergedQueue.length === 0) setQueuedEpisodeStatus("No next episodes after this one");
          return;
        }
        if (mergedQueue.length >= Math.min(PREPARED_QUEUE_TARGET, result.future_count ?? PREPARED_QUEUE_TARGET)) return;
        queuedEpisodePollRef.current = window.setTimeout(poll, 1200);
      } catch (error) {
        if (requestId !== queueRequestIdRef.current) return;
        const message = error instanceof Error ? error.message : "Episode queue failed.";
        setQueuedEpisodeStatus(message);
        queuedEpisodePollRef.current = window.setTimeout(poll, 2500);
      }
    };
    void poll();
  }, [cachePreparedEpisodeVideo]);

  const loadNextDatasetEpisode = useCallback(async (sessionOverride?: DatasetSession) => {
    const session = sessionOverride ?? datasetSession;
    if (!session) return;
    setDatasetBusy(true);
    setDatasetCheckpointSaved(false);
    setDatasetMessage("Preparing the next sampled episode from its Parquet frames…");
    setIsPreparing(true);
    setLoadError("");
    try {
      const response = await fetch(`${LOCAL_HELPER}/dataset/episode?session_id=${encodeURIComponent(session.session_id)}`);
      const result = await response.json() as { session?: DatasetSession; episode?: DatasetEpisode; complete?: boolean; error?: string };
      if (!response.ok || !result.session) throw new Error(result.error || "Could not load the next sampled episode.");
      setDatasetSession(result.session);
      if (result.complete || !result.episode) {
        setDatasetEpisode(null);
        setVideoUrl(null);
        setDatasetMessage("Sampling session complete. Every selected episode has a saved checkpoint JSON.");
        setIsPreparing(false);
        return;
      }

      await applyDatasetEpisode(result.episode);
      pendingCompletedEpisodesRef.current = [];
      refillPreparedEpisodeQueue(result.session, result.episode);
      return;

      const episode = result.episode;
      if (!ANNOTATION_CATEGORY_IDS.has(episode.category_id)) {
        throw new Error(`This viewer is configured for Shorts, Top Long sleeve, and Top Short sleeve only. The helper returned '${episode.category_label}', so restart npm run dev to load the updated paired-dataset helper.`);
      }
      sourceFileRef.current = null;
      transcodeInFlightRef.current = false;
      if (videoUrl?.startsWith("blob:")) URL.revokeObjectURL(videoUrl);
      setVideoUrl(null);
      setDatasetEpisode(episode);
      const episodeDatasetLabel = episode.dataset_label ?? episode.dataset_id ?? "dataset";
      setFileName(`${episodeDatasetLabel.toLowerCase()}_${episode.episode_stem}_top_rgb.mp4`);
      setFileSize(episode.video_size);
      setDuration(episode.frame_count / episode.fps);
      setFps(episode.fps);
      setTotalFrames(episode.frame_count);
      setCurrentFrame(0);
      setCurrentTime(0);
      setResolution(`${episode.width} × ${episode.height}`);
      setCodec("H264");
      setIsPlaying(false);
      setMetadataStatus(`${episode.category_label} · dataset episode ${episode.episode_index}`);
      setAnnotationLabels([]);
      setAnnotationSegments([]);
      setAnnotationBase(null);
      setActiveSegmentIndex(0);
      setInteractionMode("scrub");
      setAnnotationStatus("Loading the category checkpoint vocabulary…");
      setAnnotationPreviewEnd(null);
      setProgressToggles({});
      setProgressControlPoints({});
      setPendingProgressPoint(null);
      annotationHistoryRef.current = [];
      annotationRedoRef.current = [];

      setMetadataStatus(`${episodeDatasetLabel} - ${episode.category_label} - dataset episode ${episode.episode_index}`);
      const templateResponse = await fetch("/garment-segment-labels.json");
      if (!templateResponse.ok) throw new Error("Could not load garment-segment-labels.json.");
      const templateDocument = await templateResponse.json() as { templates?: GarmentLabelTemplate[] };
      const template = templateDocument.templates?.find((candidate) => candidate.category_id === episode.category_id);
      if (!template) throw new Error(`No label template is configured for ${episode.category_label}.`);
      loadAnnotationDocument(template, `${episode.category_id} template`);
      setVideoUrl(episode.video_url);
      setDatasetMessage(`${episodeDatasetLabel} ${episode.category_label}: annotating episode ${String(episode.episode_index).padStart(6, "0")}.`);
      setIsPreparing(false);
    } catch (error) {
      const message = error instanceof Error ? error.message : "Could not prepare the sampled episode.";
      setDatasetMessage(message);
      setLoadError(message);
      setIsPreparing(false);
    } finally {
      setDatasetBusy(false);
    }
  }, [applyDatasetEpisode, datasetSession, loadAnnotationDocument, refillPreparedEpisodeQueue, videoUrl]);

  const startOrResumeDatasetSession = async () => {
    setDatasetBusy(true);
    setDatasetMessage(datasetSession ? "Resuming the existing sampling session…" : "Randomly sampling unannotated episodes…");
    try {
      const response = await fetch(`${LOCAL_HELPER}/dataset/session`, { method: "POST" });
      const result = await response.json() as { session?: DatasetSession; error?: string };
      if (!response.ok || !result.session) throw new Error(result.error || "Could not start the dataset session.");
      setDatasetSession(result.session);
      await loadNextDatasetEpisode(result.session);
    } catch (error) {
      setDatasetMessage(error instanceof Error ? error.message : "Could not start the dataset session.");
      setDatasetBusy(false);
    }
  };

  const applySegmentEnd = useCallback((segmentIndex: number, requestedEnd: number) => {
    if (!annotationLabels.length || segmentIndex < 0 || segmentIndex >= annotationLabels.length) return;
    setAnnotationSegments((current) => {
      if (segmentIndex > current.length) {
        setAnnotationStatus("Annotate the preceding segment first");
        return current;
      }
      const start = segmentIndex === 0 ? 0 : current[segmentIndex - 1].end_frame_exclusive;
      const remainingSegments = annotationLabels.length - segmentIndex - 1;
      const roomLimit = totalFrames - remainingSegments;
      const nextSegment = current[segmentIndex + 1];
      const nextLimit = nextSegment ? nextSegment.end_frame_exclusive - 1 : roomLimit;
      const maximum = Math.min(roomLimit, nextLimit);
      if (maximum < start + 1) {
        setAnnotationStatus("There is no frame room for this boundary; adjust a later segment first");
        return current;
      }
      const end = clamp(Math.round(requestedEnd), start + 1, maximum);
      const next = [...current];
      const existing = current[segmentIndex];
      next[segmentIndex] = makeSegment(segmentIndex, annotationLabels[segmentIndex], start, end, existing?.notes);
      if (nextSegment) {
        next[segmentIndex + 1] = makeSegment(
          segmentIndex + 1,
          annotationLabels[segmentIndex + 1],
          end,
          nextSegment.end_frame_exclusive,
          nextSegment.notes,
        );
      }
      setAnnotationStatus(`Saved segment ${segmentIndex + 1}: [${start}, ${end}) · ${end - start} frames`);
      return next;
    });
  }, [annotationLabels, totalFrames]);

  const captureAnnotationState = useCallback((): AnnotationHistoryEntry => ({
      segments: annotationSegments.map((segment) => ({ ...segment })),
      activeSegmentIndex,
      progressToggles: Object.fromEntries(Object.entries(progressToggles).map(([index, frames]) => [index, [...frames]])),
      progressControlPoints: Object.fromEntries(Object.entries(progressControlPoints).map(([index, points]) => [index, points.map((point) => ({ ...point }))])),
    }), [activeSegmentIndex, annotationSegments, progressControlPoints, progressToggles]);

  const rememberAnnotationState = useCallback(() => {
    annotationHistoryRef.current.push(captureAnnotationState());
    if (annotationHistoryRef.current.length > 100) annotationHistoryRef.current.shift();
    annotationRedoRef.current = [];
  }, [captureAnnotationState]);

  const commitCurrentSegment = useCallback(() => {
    if (!annotationLabels.length || interactionMode !== "annotate") return;
    if (annotationPreviewEnd === null) {
      setAnnotationStatus(`Set the end of segment ${activeSegmentIndex + 1}, then press Enter to save it`);
      return;
    }
    if (activeSegmentIndex > annotationSegments.length) {
      setAnnotationStatus("Annotate the preceding segment first");
      return;
    }

    const start = activeSegmentIndex === 0 ? 0 : annotationSegments[activeSegmentIndex - 1].end_frame_exclusive;
    const remainingSegments = annotationLabels.length - activeSegmentIndex - 1;
    const roomLimit = totalFrames - remainingSegments;
    const nextSegment = annotationSegments[activeSegmentIndex + 1];
    const maximum = Math.min(roomLimit, nextSegment ? nextSegment.end_frame_exclusive - 1 : roomLimit);
    if (maximum < start + 1) {
      setAnnotationStatus("There is no frame room for this boundary; adjust a later segment first");
      return;
    }

    const committedEnd = clamp(Math.round(annotationPreviewEnd), start + 1, maximum);
    rememberAnnotationState();
    applySegmentEnd(activeSegmentIndex, committedEnd);
    setProgressToggles((current) => {
      const next = { ...current };
      next[activeSegmentIndex] = (next[activeSegmentIndex] ?? []).filter((frame) => frame >= start && frame < committedEnd - 1);
      if (nextSegment) {
        next[activeSegmentIndex + 1] = (next[activeSegmentIndex + 1] ?? []).filter(
          (frame) => frame >= committedEnd && frame < nextSegment.end_frame_exclusive - 1,
        );
      }
      return next;
    });
    setProgressControlPoints((current) => {
      const next = { ...current };
      next[activeSegmentIndex] = (next[activeSegmentIndex] ?? []).filter((point) => point.frame > start && point.frame < committedEnd - 1);
      if (nextSegment) {
        next[activeSegmentIndex + 1] = (next[activeSegmentIndex + 1] ?? []).filter(
          (point) => point.frame > committedEnd && point.frame < nextSegment.end_frame_exclusive - 1,
        );
      }
      return next;
    });
    setAnnotationPreviewEnd(null);
    if (activeSegmentIndex < annotationLabels.length - 1) {
      setActiveSegmentIndex(activeSegmentIndex + 1);
      seekFrame(committedEnd);
    }
  }, [activeSegmentIndex, annotationLabels, annotationPreviewEnd, annotationSegments, applySegmentEnd, interactionMode, rememberAnnotationState, seekFrame, totalFrames]);

  const selectSegment = useCallback((requestedIndex: number) => {
    if (!annotationLabels.length) return;
    const index = clamp(requestedIndex, 0, annotationLabels.length - 1);
    const segment = annotationSegments[index];
    setActiveSegmentIndex(index);
    setAnnotationPreviewEnd(null);
    setPendingProgressPoint(null);

    if (segment) {
      seekFrame(segment.start_frame);
      setAnnotationStatus(`Selected segment ${index + 1}: [${segment.start_frame}, ${segment.end_frame_exclusive}). Adjust it, then press Enter to save.`);
      return;
    }
    if (index === annotationSegments.length) {
      const start = index === 0 ? 0 : annotationSegments[index - 1].end_frame_exclusive;
      seekFrame(start);
      setAnnotationStatus(`Selected segment ${index + 1}: ${annotationLabels[index]}. Set its end, then press Enter to save.`);
      return;
    }
    setAnnotationStatus(`Selected segment ${index + 1}: ${annotationLabels[index]}. Save the preceding segments before setting its boundary.`);
  }, [annotationLabels, annotationSegments, seekFrame]);

  const undoLastAnnotation = useCallback(() => {
    if (pendingProgressPoint) {
      setPendingProgressPoint(null);
      setAnnotationStatus("Undid the pending progress control point");
      return;
    }
    if (annotationPreviewEnd !== null) {
      setAnnotationPreviewEnd(null);
      setAnnotationStatus("Undid the pending boundary change");
      return;
    }
    const previous = annotationHistoryRef.current.pop();
    if (!previous) {
      setAnnotationStatus("Nothing to undo");
      return;
    }
    annotationRedoRef.current.push(captureAnnotationState());
    if (annotationRedoRef.current.length > 100) annotationRedoRef.current.shift();
    setAnnotationSegments(previous.segments);
    setActiveSegmentIndex(previous.activeSegmentIndex);
    setProgressToggles(previous.progressToggles);
    setProgressControlPoints(previous.progressControlPoints);
    setPendingProgressPoint(null);
    const selected = previous.segments[previous.activeSegmentIndex];
    const fallbackFrame = previous.segments.at(-1)?.end_frame_exclusive ?? 0;
    seekFrame(selected?.start_frame ?? fallbackFrame);
    setAnnotationStatus(`Undo restored ${previous.segments.length} saved segment${previous.segments.length === 1 ? "" : "s"}`);
  }, [annotationPreviewEnd, captureAnnotationState, pendingProgressPoint, seekFrame]);

  const redoLastAnnotation = useCallback(() => {
    const next = annotationRedoRef.current.pop();
    if (!next) {
      setAnnotationStatus("Nothing to redo");
      return;
    }
    annotationHistoryRef.current.push(captureAnnotationState());
    if (annotationHistoryRef.current.length > 100) annotationHistoryRef.current.shift();
    setAnnotationSegments(next.segments);
    setActiveSegmentIndex(next.activeSegmentIndex);
    setProgressToggles(next.progressToggles);
    setProgressControlPoints(next.progressControlPoints);
    setAnnotationPreviewEnd(null);
    setPendingProgressPoint(null);
    const selected = next.segments[next.activeSegmentIndex];
    const fallbackFrame = next.segments.at(-1)?.end_frame_exclusive ?? 0;
    seekFrame(selected?.start_frame ?? fallbackFrame);
    setAnnotationStatus(`Redo restored ${next.segments.length} saved segment${next.segments.length === 1 ? "" : "s"}`);
  }, [captureAnnotationState, seekFrame]);

  const toggleProgressDirection = useCallback(() => {
    const segment = annotationSegments[activeSegmentIndex];
    if (!segment) {
      setAnnotationStatus("Save this segment boundary before adding progress reversals");
      return;
    }
    if ((progressControlPoints[activeSegmentIndex] ?? []).length) {
      setAnnotationStatus("This segment uses manual control points; edit them in Progress Edit mode");
      return;
    }
    const frame = currentFrameRef.current;
    if (frame < segment.start_frame || frame >= segment.end_frame_inclusive) {
      setAnnotationStatus(`Move inside segment ${activeSegmentIndex + 1} before reversing progress`);
      return;
    }

    rememberAnnotationState();
    setProgressToggles((current) => {
      const existing = current[activeSegmentIndex] ?? [];
      const alreadyMarked = existing.includes(frame);
      const toggles = alreadyMarked
        ? existing.filter((candidate) => candidate !== frame)
        : [...existing, frame].sort((left, right) => left - right);
      const directionAfter = toggles.filter((candidate) => candidate <= frame).length % 2
        ? "decreasing"
        : "increasing";
      setAnnotationStatus(alreadyMarked
        ? `Removed progress reversal at frame ${frame}; progress is now ${directionAfter}`
        : `Progress reverses at frame ${frame} and is now ${directionAfter}`);
      return { ...current, [activeSegmentIndex]: toggles };
    });
  }, [activeSegmentIndex, annotationSegments, progressControlPoints, rememberAnnotationState]);

  const commitProgressPoint = useCallback(() => {
    if (!pendingProgressPoint || interactionMode !== "progress") {
      setAnnotationStatus("Click and drag the progress lane to prepare a control point, then press Enter");
      return;
    }
    const segment = annotationSegments[pendingProgressPoint.segmentIndex];
    if (!segment) {
      setAnnotationStatus("Select a saved segment before editing progress");
      return;
    }
    rememberAnnotationState();
    const point = {
      frame: clamp(Math.round(pendingProgressPoint.frame), segment.start_frame + 1, segment.end_frame_inclusive - 1),
      progress: Number(clamp(pendingProgressPoint.progress, 0, 1).toFixed(6)),
    };
    setProgressControlPoints((current) => {
      const withoutSameFrame = (current[pendingProgressPoint.segmentIndex] ?? []).filter(
        (candidate) => candidate.frame !== point.frame && candidate.frame !== pendingProgressPoint.originalFrame,
      );
      return {
        ...current,
        [pendingProgressPoint.segmentIndex]: [...withoutSameFrame, point].sort((left, right) => left.frame - right.frame),
      };
    });
    setProgressToggles((current) => ({ ...current, [pendingProgressPoint.segmentIndex]: [] }));
    setPendingProgressPoint(null);
    seekFrame(point.frame);
    setAnnotationStatus(`Saved progress point at frame ${point.frame}, P=${point.progress.toFixed(3)}. Add another point or switch modes.`);
  }, [annotationSegments, interactionMode, pendingProgressPoint, rememberAnnotationState, seekFrame]);

  const switchTimelineMode = useCallback((mode: "scrub" | "annotate" | "progress") => {
    setPendingProgressPoint(null);
    if (mode === "scrub") {
      setInteractionMode("scrub");
      setAnnotationStatus("Scrub mode: timeline actions navigate without changing annotations");
      return;
    }
    if (mode === "annotate") {
      if (!annotationLabels.length) {
        setAnnotationStatus("Load a segment list before entering Annotate mode");
        return;
      }
      setInteractionMode("annotate");
      setAnnotationStatus(`Annotate mode: segment ${activeSegmentIndex + 1} is selected`);
      return;
    }
    const targetIndex = annotationSegments[activeSegmentIndex] ? activeSegmentIndex : annotationSegments.length - 1;
    if (targetIndex < 0 || !annotationSegments[targetIndex]) {
      setAnnotationStatus("Save at least one segment before entering Progress Edit mode");
      return;
    }
    setActiveSegmentIndex(targetIndex);
    setInteractionMode("progress");
    if (targetIndex !== activeSegmentIndex) seekFrame(annotationSegments[targetIndex].start_frame);
    setAnnotationStatus(`Progress Edit mode: segment ${targetIndex + 1} is selected`);
  }, [activeSegmentIndex, annotationLabels.length, annotationSegments, seekFrame]);

  useEffect(() => {
    const handleSegmentKeys = (event: KeyboardEvent) => {
      if (event.repeat) return;
      const target = event.target as HTMLElement;
      const isTyping = ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName) || target.isContentEditable;
      const key = event.key.toLowerCase();
      if (!isTyping && (event.ctrlKey || event.metaKey) && key === "z" && !event.shiftKey) {
        event.preventDefault();
        undoLastAnnotation();
        return;
      }
      if (!isTyping && ((event.ctrlKey || event.metaKey) && key === "y" || (event.ctrlKey || event.metaKey) && event.shiftKey && key === "z")) {
        event.preventDefault();
        redoLastAnnotation();
        return;
      }
      if (!isTyping && !event.ctrlKey && !event.metaKey && !event.altKey && ["s", "a", "p"].includes(key)) {
        event.preventDefault();
        switchTimelineMode(key === "s" ? "scrub" : key === "a" ? "annotate" : "progress");
        return;
      }
      if (interactionMode === "scrub") return;
      if (event.code === "Space" && interactionMode === "annotate" && !isTyping) {
        event.preventDefault();
        toggleProgressDirection();
        return;
      }
      if ((event.key === "ArrowUp" || event.key === "ArrowDown") && !isTyping) {
        event.preventDefault();
        selectSegment(activeSegmentIndex + (event.key === "ArrowUp" ? -1 : 1));
        return;
      }
      if (event.key !== "Enter") return;
      if (interactionMode === "progress") {
        if (target.tagName === "BUTTON" && pendingProgressPoint === null) return;
        event.preventDefault();
        commitProgressPoint();
        return;
      }
      if (target.tagName === "BUTTON" && annotationPreviewEnd === null) return;
      event.preventDefault();
      commitCurrentSegment();
    };
    window.addEventListener("keydown", handleSegmentKeys);
    return () => window.removeEventListener("keydown", handleSegmentKeys);
  }, [activeSegmentIndex, annotationPreviewEnd, commitCurrentSegment, commitProgressPoint, interactionMode, pendingProgressPoint, redoLastAnnotation, selectSegment, switchTimelineMode, toggleProgressDirection, undoLastAnnotation]);

  const clearAnnotations = () => {
    if (annotationSegments.length || annotationPreviewEnd !== null || Object.keys(progressToggles).length || Object.keys(progressControlPoints).length) rememberAnnotationState();
    setAnnotationSegments([]);
    setProgressToggles({});
    setProgressControlPoints({});
    setPendingProgressPoint(null);
    setActiveSegmentIndex(0);
    setAnnotationPreviewEnd(null);
    setAnnotationStatus(annotationLabels.length ? "Boundaries cleared. Set segment 1, then press Enter to save it." : "Load a segment list or checkpoint JSON to begin annotation");
  };

  const annotationComplete = annotationLabels.length > 0
    && annotationSegments.length === annotationLabels.length
    && annotationSegments[0]?.start_frame === 0
    && annotationSegments.at(-1)?.end_frame_exclusive === totalFrames
    && annotationSegments.every((segment, index) => index === 0 || annotationSegments[index - 1].end_frame_exclusive === segment.start_frame);

  const saveCheckpoints = async () => {
    if (!annotationComplete) {
      setAnnotationStatus("Finish every segment through the final video frame before saving");
      return;
    }
    const base: AnnotationDocument = { ...(annotationBase ?? {}) };
    delete base.boundaries;
    delete base.method;
    delete base.reference;
    delete base.validation;
    delete base.diagnostics_dir;
    const segmentsWithProgress = annotationSegments.map((segment, index) => ({
      ...segment,
      progress: buildSegmentProgress(segment, progressToggles[index] ?? [], progressControlPoints[index] ?? []),
    }));
    const payload = {
      ...base,
      schema_version: typeof base.schema_version === "string" ? base.schema_version : "1.0",
      template_description: "Manual contiguous temporal segment annotations created with FrameLine.",
      template_status: "manually_annotated",
      task: typeof base.task === "object" && base.task ? base.task : {
        name: "temporal video annotation",
        description: "Sequential, non-overlapping semantic segments.",
        garment_type: "to_be_annotated",
        annotation_scope: `${fileName} timeline`,
      },
      video_stem: fileName.replace(/\.[^/.]+$/, ""),
      video_file: typeof base.video_file === "string" ? base.video_file : fileName,
      frame_indexing: {
        base: 0,
        start_frame: 0,
        end_frame_exclusive: totalFrames,
        end_frame_inclusive: totalFrames - 1,
        note: "Use zero-based decoded-video frame indices. Each segment is [start_frame, end_frame_exclusive).",
        annotation_sampling: {
          type: "full_rate_every_frame",
          original_frame_stride: 1,
          original_frame_offset: 0,
          annotation_frame_k_maps_to_original_frame: "k",
        },
      },
      summary: {
        num_labels: annotationLabels.length,
        num_unique_labels: new Set(annotationLabels).size,
        num_annotated_segments: annotationSegments.length,
        first_frame: 0,
        last_frame_inclusive: totalFrames - 1,
        num_labeled_frames: totalFrames,
        coverage_is_contiguous: true,
        coverage_is_complete: true,
      },
      annotation_guidance: {
        segments: "Segments are sequential and non-overlapping. end_frame_exclusive equals the next segment's start_frame.",
        coverage: `Every decoded frame from 0 through ${totalFrames - 1} is labeled exactly once.`,
        progress: "Progress starts at 0, changes by a constant normalized slope per frame, reverses at marked frames, and is clamped to [0, 1].",
      },
      segment_schema: {
        segment_id: "zero-based integer in temporal order",
        label: "task-specific string included in labels",
        start_frame: "inclusive zero-based frame index",
        end_frame_exclusive: "exclusive zero-based frame boundary",
        end_frame_inclusive: "end_frame_exclusive - 1",
        num_frames: "end_frame_exclusive - start_frame",
        progress: "piecewise-linear per-frame progress with direction-change keyframes",
      },
      labels: annotationLabels,
      segments: segmentsWithProgress,
      frames: [],
      video_metadata: {
        frame_count: totalFrames,
        fps,
        width: videoRef.current?.videoWidth ?? null,
        height: videoRef.current?.videoHeight ?? null,
        duration_seconds: duration,
      },
      annotation_tool: {
        name: "FrameLine",
        format: "contiguous_single_timeline",
      },
    };
    if (datasetEpisode && datasetSession) {
      try {
        const savedDatasetEpisode = datasetEpisode;
        const savedEpisodeLabel = `${savedDatasetEpisode.dataset_label} episode ${String(savedDatasetEpisode.episode_index).padStart(6, "0")}`;
        const queuedEpisodes = preparedEpisodeQueueRef.current;
        if (queuedEpisodes.length === 0) {
          setAnnotationStatus(`Next episode queue is not ready yet. Wait until NEXT QUEUE shows at least 1/${PREPARED_QUEUE_TARGET} ready.`);
          setQueuedEpisodeStatus(`Preparing queue: 0/${PREPARED_QUEUE_TARGET} ready`);
          refillPreparedEpisodeQueue(datasetSession, savedDatasetEpisode);
          return;
        }

        const nextItem = queuedEpisodes[0];
        const remainingQueue = queuedEpisodes.slice(1);
        preparedEpisodeQueueRef.current = remainingQueue;
        setPreparedEpisodeQueue(remainingQueue);
        preparedVideoCacheRef.current.delete(preparedQueueKey(nextItem));
        const completion = {
          dataset_id: savedDatasetEpisode.dataset_id,
          category_id: savedDatasetEpisode.category_id,
          episode_index: savedDatasetEpisode.episode_index,
        };
        pendingCompletedEpisodesRef.current = [...pendingCompletedEpisodesRef.current, completion];

        setDatasetBusy(true);
        setAnnotationStatus(`Opening queued next episode. Saving ${savedEpisodeLabel} in background...`);
        setPreviousSaveStatus({ state: "saving", message: `Saving ${savedEpisodeLabel}...` });

        setDatasetSession(nextItem.session);
        await applyDatasetEpisode(nextItem.episode);
        setAnnotationStatus(`Started queued ${nextItem.episode.dataset_label} episode ${String(nextItem.episode.episode_index).padStart(6, "0")}. Saving ${savedEpisodeLabel} in background.`);
        setDatasetBusy(false);
        refillPreparedEpisodeQueue(nextItem.session, nextItem.episode);

        void fetch(`${LOCAL_HELPER}/dataset/checkpoint`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: datasetSession.session_id,
            dataset_id: savedDatasetEpisode.dataset_id,
            episode_index: savedDatasetEpisode.episode_index,
            expected_frame_count: totalFrames,
            return_next: false,
            checkpoints: payload,
          }),
        })
          .then(async (backgroundResponse) => {
            const backgroundResult = await backgroundResponse.json() as { saved?: boolean; path?: string; session?: DatasetSession; error?: string };
            if (!backgroundResponse.ok || !backgroundResult.saved) {
              throw new Error(backgroundResult.error || "Dataset checkpoint save failed.");
            }
            pendingCompletedEpisodesRef.current = pendingCompletedEpisodesRef.current.filter((candidate) => (
              candidate.dataset_id !== completion.dataset_id
              || candidate.category_id !== completion.category_id
              || candidate.episode_index !== completion.episode_index
            ));
            setPreviousSaveStatus({ state: "saved", message: `Saved ${savedEpisodeLabel}`, path: backgroundResult.path });
          })
          .catch((error) => {
            const message = error instanceof Error ? error.message : "Could not save this dataset checkpoint.";
            setPreviousSaveStatus({ state: "failed", message: `${savedEpisodeLabel}: ${message}` });
          });
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not save this dataset checkpoint.";
        setPreviousSaveStatus({ state: "failed", message });
        setAnnotationStatus(message);
        setDatasetBusy(false);
      }
      return;
    }
    const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], { type: "application/json" });
    try {
      const picker = (window as unknown as { showSaveFilePicker?: LocalSavePicker }).showSaveFilePicker;
      if (picker) {
        const handle = await picker.call(window, {
          suggestedName: "checkpoints.json",
          types: [{ description: "Checkpoint JSON", accept: { "application/json": [".json"] } }],
        });
        const writable = await handle.createWritable();
        await writable.write(blob);
        await writable.close();
      } else {
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = "checkpoints.json";
        link.click();
        URL.revokeObjectURL(url);
      }
      setAnnotationStatus("Saved checkpoints.json with complete, gap-free coverage");
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setAnnotationStatus("Could not save checkpoints.json");
    }
  };

  const frameFromPointer = (event: ReactPointerEvent<HTMLDivElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const ratio = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
    return Math.round(ratio * maxFrame);
  };

  const progressPointFromDrag = (
    event: ReactPointerEvent<HTMLDivElement>,
    segment: AnnotationSegment,
    drag: { segmentIndex: number; originFrame: number; originProgress: number; originClientX: number; originClientY: number },
  ) => {
    const track = progressTrackRef.current;
    if (!track) return null;
    const bounds = track.getBoundingClientRect();
    const frameDelta = Math.round(((event.clientX - drag.originClientX) / Math.max(1, bounds.width)) * maxFrame);
    const progressDelta = -(event.clientY - drag.originClientY) / Math.max(1, bounds.height);
    const frame = clamp(event.shiftKey ? drag.originFrame : drag.originFrame + frameDelta, segment.start_frame + 1, segment.end_frame_inclusive - 1);
    const progress = clamp(event.ctrlKey ? drag.originProgress : drag.originProgress + progressDelta, 0, 1);
    return { segmentIndex: segment.segment_id, frame, progress: Number(progress.toFixed(6)) };
  };

  const startTimelineInteraction = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!totalFrames) return;
    event.currentTarget.setPointerCapture(event.pointerId);
    const frame = frameFromPointer(event);
    const trackBounds = progressTrackRef.current?.getBoundingClientRect();
    const clickedProgressLane = Boolean(trackBounds && event.clientY >= trackBounds.top && event.clientY <= trackBounds.bottom);
    if (clickedProgressLane) {
      const cursorFrame = currentFrameRef.current;
      const targetSegmentIndex = annotationSegments.findIndex(
        (segment) => cursorFrame >= segment.start_frame && cursorFrame < segment.end_frame_exclusive,
      );
      const segment = annotationSegments[targetSegmentIndex];
      if (!segment || segment.num_frames < 3) {
        setAnnotationStatus("Move the timeline cursor inside a completed segment before clicking the progress lane");
        return;
      }
      if (cursorFrame <= segment.start_frame || cursorFrame >= segment.end_frame_inclusive) {
        setAnnotationStatus("Move the timeline cursor to an interior frame; segment endpoints are fixed at progress 0 and 1");
        return;
      }
      if (!trackBounds || event.clientY < trackBounds.top || event.clientY > trackBounds.bottom) {
        setAnnotationStatus("Click inside the progress lane to add or move a control point");
        return;
      }
      const currentModel = buildSegmentProgress(
        segment,
        progressToggles[targetSegmentIndex] ?? [],
        progressControlPoints[targetSegmentIndex] ?? [],
      );
      const currentProgress = currentModel.per_frame[cursorFrame - segment.start_frame]?.progress ?? 0;
      const point = { segmentIndex: targetSegmentIndex, frame: cursorFrame, progress: currentProgress };
      const nearestExisting = (progressControlPoints[targetSegmentIndex] ?? [])
        .map((candidate) => ({ candidate, distance: Math.abs(candidate.frame - point.frame) }))
        .sort((left, right) => left.distance - right.distance)[0];
      setActiveSegmentIndex(targetSegmentIndex);
      setInteractionMode("progress");
      progressDragRef.current = {
        segmentIndex: targetSegmentIndex,
        originFrame: point.frame,
        originProgress: point.progress,
        originClientX: event.clientX,
        originClientY: event.clientY,
        moved: false,
      };
      setPendingProgressPoint({
        ...point,
        ...(nearestExisting && nearestExisting.distance <= 2 ? { originalFrame: nearestExisting.candidate.frame } : {}),
      });
      setAnnotationStatus(`Progress point prepared at cursor frame ${point.frame}, P=${point.progress.toFixed(3)}; drag it or press Enter.`);
      return;
    }
    if (interactionMode === "annotate" && annotationLabels.length) {
      if (activeSegmentIndex > annotationSegments.length) {
        setAnnotationStatus("Annotate the preceding segment first");
        return;
      }
      const startFrame = activeSegmentIndex === 0 ? 0 : annotationSegments[activeSegmentIndex - 1].end_frame_exclusive;
      annotationDragRef.current = { segmentIndex: activeSegmentIndex, startFrame };
      setAnnotationPreviewEnd(Math.max(startFrame + 1, frame + 1));
      seekFrame(frame);
      return;
    }
    if (interactionMode === "progress") setPendingProgressPoint(null);
    setIsScrubbing(true);
    seekFrame(frame);
  };

  const moveTimelineInteraction = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (!totalFrames) return;
    const frame = frameFromPointer(event);
    setHoverFrame(frame);
    if (progressDragRef.current) {
      const drag = progressDragRef.current;
      const segment = annotationSegments[drag.segmentIndex];
      if (!segment) return;
      drag.moved = drag.moved || Math.hypot(event.clientX - drag.originClientX, event.clientY - drag.originClientY) > 2;
      if (!drag.moved) return;
      const point = progressPointFromDrag(event, segment, drag);
      if (!point) return;
      setPendingProgressPoint((current) => ({ ...point, ...(current?.originalFrame !== undefined ? { originalFrame: current.originalFrame } : {}) }));
      seekFrame(point.frame);
      return;
    }
    if (annotationDragRef.current) {
      setAnnotationPreviewEnd(Math.max(annotationDragRef.current.startFrame + 1, frame + 1));
      seekFrame(frame);
      return;
    }
    if (isScrubbing) seekFrame(frame);
  };

  const finishTimelineInteraction = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (progressDragRef.current) {
      const drag = progressDragRef.current;
      const segment = annotationSegments[drag.segmentIndex];
      const moved = drag.moved || Math.hypot(event.clientX - drag.originClientX, event.clientY - drag.originClientY) > 2;
      const draggedPoint = moved && segment ? progressPointFromDrag(event, segment, drag) : null;
      const point = draggedPoint ?? {
        segmentIndex: drag.segmentIndex,
        frame: drag.originFrame,
        progress: drag.originProgress,
      };
      if (point) {
        setPendingProgressPoint((current) => ({ ...point, ...(current?.originalFrame !== undefined ? { originalFrame: current.originalFrame } : {}) }));
        setAnnotationStatus(`Progress point ready at frame ${point.frame}, P=${point.progress.toFixed(3)}. Press Enter to save it.`);
      }
    }
    if (annotationDragRef.current) {
      const frame = frameFromPointer(event);
      const requestedEnd = Math.max(annotationDragRef.current.startFrame + 1, frame + 1);
      setAnnotationPreviewEnd(requestedEnd);
      setAnnotationStatus(`Boundary ready for segment ${annotationDragRef.current.segmentIndex + 1}. Press Enter to save and continue.`);
    }
    annotationDragRef.current = null;
    progressDragRef.current = null;
    setIsScrubbing(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const updateFps = (event: ChangeEvent<HTMLInputElement>) => {
    const next = Number(event.target.value);
    if (!Number.isFinite(next) || next <= 0) return;
    setFps(next);
    if (duration) setTotalFrames(Math.max(1, Math.round(duration * next)));
    setMetadataStatus("Manual FPS override");
  };

  const selectedSegment = annotationSegments[activeSegmentIndex];
  const selectedStart = activeSegmentIndex === 0
    ? 0
    : annotationSegments[activeSegmentIndex - 1]?.end_frame_exclusive ?? 0;
  const selectedEnd = annotationPreviewEnd ?? selectedSegment?.end_frame_exclusive ?? Math.min(totalFrames, Math.max(selectedStart + 1, currentFrame + 1));
  const annotatedFrames = annotationSegments.length ? annotationSegments.at(-1)?.end_frame_exclusive ?? 0 : 0;
  const previewStart = selectedStart;
  const progressModels = useMemo(
    () => annotationSegments.map((segment, index) => {
      const savedPoints = progressControlPoints[index] ?? [];
      const effectivePoints = pendingProgressPoint?.segmentIndex === index
        ? [
          ...savedPoints.filter((point) => point.frame !== pendingProgressPoint.originalFrame && point.frame !== pendingProgressPoint.frame),
          { frame: pendingProgressPoint.frame, progress: pendingProgressPoint.progress },
        ]
        : savedPoints;
      return {
        segment,
        index,
        model: buildSegmentProgress(segment, progressToggles[index] ?? [], effectivePoints),
      };
    }),
    [annotationSegments, pendingProgressPoint, progressControlPoints, progressToggles],
  );
  const activeProgress = progressModels.find(({ index }) => index === activeSegmentIndex);
  const activeProgressPoint = activeProgress?.model.per_frame[
    clamp(currentFrame - (activeProgress?.segment.start_frame ?? 0), 0, Math.max(0, (activeProgress?.model.per_frame.length ?? 1) - 1))
  ];
  const activeProgressPiece = activeProgress?.model.linear_pieces.find(
    (piece) => currentFrame >= piece.start_frame && currentFrame <= piece.end_frame_inclusive,
  );
  const activeProgressDirection = (activeProgressPiece?.slope_per_frame ?? 0) < 0 ? "decreasing" : "increasing";
  const progressEditPoint = interactionMode === "progress" && activeProgress
    ? pendingProgressPoint?.segmentIndex === activeSegmentIndex
      ? pendingProgressPoint
      : {
        segmentIndex: activeSegmentIndex,
        frame: clamp(currentFrame, activeProgress.segment.start_frame, activeProgress.segment.end_frame_inclusive),
        progress: activeProgressPoint?.progress ?? 0,
      }
    : null;

  useEffect(() => {
    const canvas = progressCanvasRef.current;
    if (!canvas) return;

    const drawProgress = () => {
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const pixelRatio = window.devicePixelRatio || 1;
      canvas.width = Math.round(bounds.width * pixelRatio);
      canvas.height = Math.round(bounds.height * pixelRatio);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      progressModels.forEach(({ segment, index, model }) => {
        const color = SEGMENT_COLORS[index % SEGMENT_COLORS.length];
        const bandStart = (segment.start_frame / Math.max(1, totalFrames)) * bounds.width;
        const bandEnd = (segment.end_frame_exclusive / Math.max(1, totalFrames)) * bounds.width;
        const frameBoundaryX = (frame: number) => (frame / Math.max(1, totalFrames)) * bounds.width;
        context.globalAlpha = index === activeSegmentIndex ? 0.2 : 0.08;
        context.fillStyle = color;
        context.fillRect(bandStart, 0, Math.max(1, bandEnd - bandStart), bounds.height);

        context.globalAlpha = index === activeSegmentIndex ? 1 : 0.45;
        context.strokeStyle = color;
        context.lineWidth = index === activeSegmentIndex ? 2.5 : 1.25;
        context.lineJoin = "round";
        context.lineCap = "round";
        context.beginPath();
        model.per_frame.forEach((point, pointIndex) => {
          const x = frameBoundaryX(point.frame);
          const y = bounds.height - 4 - point.progress * (bounds.height - 8);
          if (pointIndex === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        const lastPoint = model.per_frame.at(-1);
        if (lastPoint) {
          context.lineTo(
            frameBoundaryX(segment.end_frame_exclusive),
            bounds.height - 4 - lastPoint.progress * (bounds.height - 8),
          );
        }
        context.stroke();

        model.direction_changes.forEach((change) => {
          const x = frameBoundaryX(change.frame);
          const y = bounds.height - 4 - change.progress * (bounds.height - 8);
          context.beginPath();
          context.fillStyle = color;
          context.arc(x, y, index === activeSegmentIndex ? 3.5 : 2.5, 0, Math.PI * 2);
          context.fill();
        });
      });
      context.globalAlpha = 1;
    };

    drawProgress();
    const observer = new ResizeObserver(drawProgress);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [activeSegmentIndex, maxFrame, progressModels, totalFrames]);

  const datasetSessionComplete = datasetSession?.status === "complete";

  return (
    <main
      className={`app-shell ${isDraggingFile ? "is-file-dragging" : ""}`}
      onDragEnter={(event) => {
        event.preventDefault();
        setIsDraggingFile(true);
      }}
      onDragOver={(event) => event.preventDefault()}
      onDragLeave={(event) => {
        if (event.currentTarget === event.target) setIsDraggingFile(false);
      }}
      onDrop={onDrop}
    >
      <input ref={inputRef} type="file" accept="video/*,.mp4,.mov,.m4v,.webm" hidden onChange={chooseFile} />
      <input ref={annotationInputRef} type="file" accept="application/json,.json" hidden onChange={chooseAnnotationJson} />

      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark" aria-hidden="true"><span /><span /></div>
          <div>
            <h1>FrameLine</h1>
            <p>Local, source-frame video inspector</p>
          </div>
        </div>
        {datasetSession && (
          <div className="dataset-session-tracker" aria-label="Dataset annotation session progress">
            <div className="dataset-tracker-summary">
              <span>SESSION</span>
              <strong>{datasetSession.completed_count}/{datasetSession.sampled_count}</strong>
              <small>episodes saved</small>
            </div>
            <div className="dataset-source-trackers">
              {(datasetSession.datasets ?? []).map((dataset) => (
                <div
                  key={dataset.id}
                  className={`dataset-source-tracker ${dataset.id === datasetSession.active_dataset_id ? "is-active" : ""}`}
                >
                  <div className="dataset-source-summary">
                    <span>{dataset.label}</span>
                    <strong>{dataset.completed_count}/{dataset.sampled_count}</strong>
                  </div>
                  <div className="dataset-category-trackers">
                    {(dataset.categories ?? []).map((category) => (
                      <div
                        key={`${dataset.id}-${category.id}`}
                        className={`${dataset.id === datasetSession.active_dataset_id && category.id === datasetSession.active_category_id ? "is-active" : ""} ${category.remaining_count === 0 ? "is-complete" : ""}`}
                      >
                        <span>{category.label}</span>
                        <strong>{category.completed_count}/{category.sampled_count}</strong>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
        {datasetSession && (
          <div className={`previous-save-status is-${previousSaveStatus.state}`} title={previousSaveStatus.path ?? previousSaveStatus.message}>
            <span>PREVIOUS SAVE</span>
            <strong>{previousSaveStatus.message}</strong>
          </div>
        )}
        {datasetSession && (
          <div className={`previous-save-status is-${preparedEpisodeQueue.length > 0 ? "saved" : "saving"}`} title={queuedEpisodeStatus}>
            <span>NEXT QUEUE</span>
            <strong>{preparedEpisodeQueue.length > 0 ? `${preparedEpisodeQueue.length}/${PREPARED_QUEUE_TARGET} ready` : queuedEpisodeStatus}</strong>
          </div>
        )}
        <div className="top-actions">
          <span className="privacy-pill"><span className="status-dot" /> Files stay on this device</span>
          {datasetEpisode && datasetCheckpointSaved && (
            <button className="primary-button next-episode-button" disabled={datasetBusy} onClick={() => void loadNextDatasetEpisode()}>
              Next episode <span className="button-glyph">→</span>
            </button>
          )}
          <button className="secondary-button" onClick={() => inputRef.current?.click()}>
            <span className="button-glyph">＋</span> Open video
          </button>
        </div>
      </header>

      {!videoUrl ? (
        <section
          className={`empty-stage ${isPreparing ? "is-preparing" : ""}`}
        >
          <div className="drop-illustration"><span className="upload-glyph">↑</span></div>
          <p className="eyebrow">LEHOME DATASET ANNOTATION</p>
          <h2>{isPreparing ? "Preparing an episode…" : datasetSessionComplete ? "Sampling session complete" : "Annotate sampled episodes"}</h2>
          <p className="empty-copy">
            {isPreparing
              ? datasetMessage
              : loadError || datasetMessage}
          </p>
          {datasetInfo?.available && !datasetSessionComplete && !isPreparing && (
            <div className="dataset-launch-card">
              <div className="dataset-source-list">
                {datasetInfo.datasets?.map((dataset) => (
                  <div className="dataset-root" key={dataset.id}>
                    <span>{dataset.label} dataset</span>
                    <strong>{dataset.root}</strong>
                  </div>
                )) ?? <div className="dataset-root"><span>Datasets</span><strong>{datasetInfo.root}</strong></div>}
              </div>
              <div className="dataset-category-preview">
                {datasetInfo.categories?.filter((category) => ANNOTATION_CATEGORY_IDS.has(category.id)).map((category) => (
                  <span key={category.id}><strong>{category.total_episodes.toLocaleString()}</strong>{category.label}</span>
                ))}
              </div>
              <button className="primary-button" disabled={datasetBusy} onClick={() => void startOrResumeDatasetSession()}>
                <span className="button-glyph">▶</span>{datasetSession ? "Resume sampled session" : "Start 25% sampling session"}
              </button>
              <small>Only Shorts, Top Long sleeve, and Top Short sleeve episodes without a saved temporal checkpoint are eligible. The session alternates Human then Sim inside each category.</small>
            </div>
          )}
          {!isPreparing && (
            <button className="secondary-button standalone-video-button" onClick={() => inputRef.current?.click()}>
              Open a standalone video instead
            </button>
          )}
          {isPreparing && <div className="conversion-progress" aria-label="Converting video"><span /><span /><span /></div>}
          {datasetSessionComplete && datasetSession && (
            <div className="session-complete-summary">{datasetSession.completed_count} of {datasetSession.sampled_count} sampled episodes saved</div>
          )}
        </section>
      ) : (
        <section className="viewer-workspace" ref={viewerRef}>
          <div className="media-strip">
            <div className="media-name"><span className="video-glyph">▰</span><div><strong>{fileName}</strong><span>{formatBytes(fileSize)}</span></div></div>
            <div className="media-facts">
              {datasetEpisode && <span className="dataset-episode-fact">{datasetEpisode.dataset_label} · {datasetEpisode.category_label} · episode {String(datasetEpisode.episode_index).padStart(6, "0")}</span>}
              <span>{resolution}</span><span>{codec}</span><span>{totalFrames.toLocaleString()} frames</span>
              <span className="metadata-status">{metadataStatus}</span>
            </div>
          </div>

          <div className="stage-and-inspector">
            <div className="video-stage">
              <video
                ref={videoRef}
                src={videoUrl}
                style={{ maxWidth: `${videoDisplayScale}%`, maxHeight: `${videoDisplayScale}%` }}
                playsInline
                onLoadedMetadata={onLoadedMetadata}
                onTimeUpdate={onTimeUpdate}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                onEnded={() => setIsPlaying(false)}
                onError={onVideoError}
                onClick={togglePlayback}
              />
              <div className="frame-overlay">
                <span>FRAME</span>
                <strong>{String(currentFrame).padStart(6, "0")}</strong>
              </div>
              <button className="fullscreen-button" aria-label="Enter fullscreen" onClick={() => void viewerRef.current?.requestFullscreen()}>
                <span aria-hidden="true">⛶</span>
              </button>
            </div>

            <aside className="inspector">
              <p className="eyebrow">CURRENT POSITION</p>
              <div className="big-frame-number">{String(currentFrame).padStart(6, "0")}</div>
              <span className="frame-range">of {String(maxFrame).padStart(6, "0")}</span>
              <div className="readout-grid">
                <div><span>Source timecode</span><strong>{formatTimecode(currentFrame, fps)}</strong></div>
                <div><span>Elapsed time</span><strong>{formatClock(currentTime)}</strong></div>
                <div><span>Duration</span><strong>{formatClock(duration)}</strong></div>
                <label><span>Source FPS</span><input type="number" min="1" max="240" step="0.001" value={fps} onChange={updateFps} /></label>
                <label className="display-scale-control">
                  <span>Video size</span>
                  <div>
                    <input
                      aria-label="Video display size"
                      type="range"
                      min="50"
                      max="100"
                      step="1"
                      value={videoDisplayScale}
                      onChange={(event) => setVideoDisplayScale(Number(event.target.value))}
                    />
                    <strong>{videoDisplayScale}%</strong>
                  </div>
                </label>
              </div>
              <div className="frame-jump">
                <label htmlFor="frame-jump">Jump to frame</label>
                <div><input id="frame-jump" type="number" min="0" max={maxFrame} value={currentFrame} onChange={(event) => seekFrame(Number(event.target.value))} /><button onClick={() => seekFrame(0)} aria-label="Return to first frame">↺</button></div>
              </div>
            </aside>

            <aside className="annotation-panel">
              <div className="annotation-console">
                <div className="annotation-console-top">
                  <div>
                    <p className="eyebrow">ANNOTATION WORKFLOW</p>
                    <strong>{annotationLabels.length ? `${annotationSegments.length}/${annotationLabels.length} segments defined` : "No segment list loaded"}</strong>
                    <span>{annotationStatus}</span>
                  </div>
                  <div className="annotation-actions">
                    <button onClick={() => annotationInputRef.current?.click()}>Load JSON</button>
                    <button onClick={() => void loadSampleLabels()}>Use sample labels</button>
                    <button onClick={clearAnnotations} disabled={!annotationLabels.length}>Clear boundaries</button>
                    <button className="save-checkpoints" onClick={() => void saveCheckpoints()} disabled={!annotationComplete || datasetCheckpointSaved || datasetBusy}>
                      {datasetCheckpointSaved ? "Checkpoint saved" : datasetEpisode ? "Save episode checkpoint" : "Save checkpoints.json"}
                    </button>
                    {datasetEpisode && datasetCheckpointSaved && (
                      <button className="next-dataset-episode" disabled={datasetBusy} onClick={() => void loadNextDatasetEpisode()}>
                        Next episode →
                      </button>
                    )}
                  </div>
                </div>

                {annotationLabels.length > 0 && (
                  <>
                    <div className="annotation-progress"><span style={{ width: `${(annotatedFrames / Math.max(1, totalFrames)) * 100}%` }} /></div>
                    <div className="boundary-editor">
                      <div><span>Selected segment</span><strong>{activeSegmentIndex + 1}. {annotationLabels[activeSegmentIndex]}</strong></div>
                      <label><span>Start frame</span><input value={selectedStart} disabled /></label>
                      <label><span>End exclusive</span><input type="number" min={selectedStart + 1} max={totalFrames} value={selectedEnd} onChange={(event) => setAnnotationPreviewEnd(clamp(Number(event.target.value), selectedStart + 1, totalFrames))} /></label>
                      <div><span>Frames</span><strong>{Math.max(0, selectedEnd - selectedStart)}</strong></div>
                      {activeSegmentIndex === annotationLabels.length - 1 && <button onClick={() => setAnnotationPreviewEnd(totalFrames)}>Set end to video end</button>}
                      <div className="enter-to-save"><kbd>Enter</kbd><span>Save segment and continue</span></div>
                      <div className="enter-to-save"><kbd>Ctrl Z</kbd><span>Undo last annotation change</span></div>
                    </div>
                  </>
                )}
              </div>
            </aside>

            <aside className="segments-panel">
              <div className="inspector-segments">
                <div className="inspector-segments-heading">
                  <div><p className="eyebrow">TEMPORAL SEGMENTS</p><span>Choose the segment to annotate</span></div>
                  <strong>{annotationSegments.length}/{annotationLabels.length}</strong>
                </div>
                {annotationLabels.length ? (
                  <div className="segment-selector vertical" aria-label="Checkpoint segment list">
                    {annotationLabels.map((label, index) => {
                      const segment = annotationSegments[index];
                      const unavailable = index > annotationSegments.length;
                      return (
                        <button
                          key={`${index}-${label}`}
                          className={`${index === activeSegmentIndex ? "is-active" : ""} ${segment ? "is-defined" : ""}`}
                          aria-label={`Select segment ${index + 1}: ${label}`}
                          onClick={() => selectSegment(index)}
                        >
                          <i style={{ background: SEGMENT_COLORS[index % SEGMENT_COLORS.length] }} />
                          <span>{index + 1}. {label}</span>
                          <small>{segment ? `${segment.start_frame}–${segment.end_frame_inclusive}` : unavailable ? "waiting" : "next"}</small>
                        </button>
                      );
                    })}
                  </div>
                ) : (
                  <div className="inspector-segments-empty">Load a label-list or checkpoint JSON below.</div>
                )}
              </div>
            </aside>
          </div>

          <div className="transport">
            <div className="transport-main">
              <button onClick={() => seekFrame(currentFrame - 1)} aria-label="Previous frame"><span className="transport-glyph">‹</span></button>
              <button className="play-button" onClick={togglePlayback} aria-label={isPlaying ? "Pause" : "Play"}><span className="play-glyph">{isPlaying ? "Ⅱ" : "▶"}</span></button>
              <button onClick={() => seekFrame(currentFrame + 1)} aria-label="Next frame"><span className="transport-glyph">›</span></button>
            </div>
            <div className="transport-time"><strong>{formatTimecode(currentFrame, fps)}</strong><span>{formatClock(currentTime)} / {formatClock(duration)}</span></div>
            <div className={`shuttle-state ${holdSpeed > 1 ? "is-fast" : ""}`}><span className="gauge-glyph">◒</span><span>Arrow shuttle</span><strong>{holdSpeed}×</strong></div>
          </div>

          <div className="timeline-panel">
            <div className="timeline-heading">
              <div><p className="eyebrow">SINGLE ANNOTATION TIMELINE</p><span>{interactionMode === "annotate"
                ? "Drag to preview the selected segment end, then press Enter to save and continue"
                : interactionMode === "progress"
                  ? "Scrub outside the progress lane to position the frame cursor; click the lane to create its point"
                  : "Drag the playhead or click anywhere to seek"}</span></div>
              <div className="timeline-tools">
                <div className="mode-switch" aria-label="Timeline interaction mode">
                  <button className={interactionMode === "scrub" ? "is-active" : ""} onClick={() => switchTimelineMode("scrub")}>Scrub <kbd>S</kbd></button>
                  <button className={interactionMode === "annotate" ? "is-active" : ""} disabled={!annotationLabels.length} onClick={() => switchTimelineMode("annotate")}>Annotate <kbd>A</kbd></button>
                  <button className={interactionMode === "progress" ? "is-active" : ""} disabled={!annotationSegments.length} onClick={() => switchTimelineMode("progress")}>Progress Edit <kbd>P</kbd></button>
                </div>
                <div className="keyboard-hints">
                  <kbd>↑</kbd><kbd>↓</kbd><span>select</span><kbd>←</kbd><kbd>→</kbd><span>frame</span><kbd>Enter</kbd><span>save</span>
                  {interactionMode === "progress" && <><kbd>Shift</kbd><span>vertical</span><kbd>Ctrl</kbd><span>horizontal</span></>}
                  <kbd>Ctrl Z</kbd><span>undo</span><kbd>Ctrl Y</kbd><span>redo</span><kbd>Space</kbd><span>{interactionMode === "annotate" ? "reverse progress" : "play"}</span>
                </div>
              </div>
            </div>
            <div
              className={`timeline ${interactionMode === "annotate" ? "is-annotating" : ""} ${interactionMode === "progress" ? "is-progress-editing" : ""}`}
              onPointerDown={startTimelineInteraction}
              onPointerMove={moveTimelineInteraction}
              onPointerUp={finishTimelineInteraction}
              onPointerCancel={() => { annotationDragRef.current = null; progressDragRef.current = null; setAnnotationPreviewEnd(null); setPendingProgressPoint(null); setIsScrubbing(false); }}
              onPointerLeave={() => { if (!isScrubbing) setHoverFrame(null); }}
            >
              <div className="ruler-grid" />
              {timelineTicks.map((tick) => (
                <div className="major-tick" style={{ left: `${tick.ratio * 100}%` }} key={tick.frame}>
                  <span>{formatTimecode(tick.frame, fps)}</span><small>F {tick.frame}</small>
                </div>
              ))}
              <div className="annotation-track">
                {!annotationLabels.length && <div className="track-empty"><span className="video-glyph">▰</span><strong>{fileName.replace(/\.[^/.]+$/, "")}</strong><span>{totalFrames} source frames · {fps.toFixed(3)} FPS</span></div>}
                {annotationSegments.map((segment, index) => (
                  <div
                    className={`timeline-segment ${index === activeSegmentIndex ? "is-selected" : ""}`}
                    key={`${segment.segment_id}-${segment.end_frame_exclusive}`}
                    style={{
                      left: `${(segment.start_frame / totalFrames) * 100}%`,
                      width: `${(segment.num_frames / totalFrames) * 100}%`,
                      background: SEGMENT_COLORS[index % SEGMENT_COLORS.length],
                    }}
                  >
                    <strong>{index + 1}. {segment.label}</strong>
                    <span>{segment.start_frame}–{segment.end_frame_inclusive}</span>
                  </div>
                ))}
                {annotationPreviewEnd !== null && interactionMode === "annotate" && (
                  <div
                    className="timeline-segment is-preview"
                    style={{
                      left: `${(previewStart / totalFrames) * 100}%`,
                      width: `${((clamp(annotationPreviewEnd, previewStart + 1, totalFrames) - previewStart) / totalFrames) * 100}%`,
                      background: SEGMENT_COLORS[activeSegmentIndex % SEGMENT_COLORS.length],
                    }}
                  >
                    <strong>{activeSegmentIndex + 1}. {annotationLabels[activeSegmentIndex]}</strong>
                    <span>release to set boundary</span>
                  </div>
                )}
              </div>
              <div ref={progressTrackRef} className="progress-track" aria-label="Within-segment progress from zero to one">
                <canvas ref={progressCanvasRef} />
                {(progressControlPoints[activeSegmentIndex] ?? [])
                  .filter((point) => point.frame !== pendingProgressPoint?.originalFrame)
                  .map((point) => (
                  <i
                    className="progress-control-dot is-saved"
                    key={`${activeSegmentIndex}-${point.frame}`}
                    style={{
                      left: `${(point.frame / Math.max(1, maxFrame)) * 100}%`,
                      top: `${(1 - point.progress) * 100}%`,
                      borderColor: SEGMENT_COLORS[activeSegmentIndex % SEGMENT_COLORS.length],
                    }}
                  />
                ))}
                {progressEditPoint && (
                  <i
                    className={`progress-control-dot is-current ${pendingProgressPoint ? "is-pending" : ""}`}
                    style={{
                      left: `${(progressEditPoint.frame / Math.max(1, maxFrame)) * 100}%`,
                      top: `${(1 - progressEditPoint.progress) * 100}%`,
                      background: SEGMENT_COLORS[activeSegmentIndex % SEGMENT_COLORS.length],
                    }}
                  />
                )}
                <span className="progress-axis progress-one">1</span>
                <span className="progress-axis progress-zero">0</span>
                {activeProgress && (
                  <strong className="progress-readout" style={{ color: SEGMENT_COLORS[activeSegmentIndex % SEGMENT_COLORS.length] }}>
                    P {activeProgressPoint?.progress.toFixed(3) ?? "0.000"} · {activeProgressDirection === "increasing" ? "↗ increasing" : "↘ decreasing"}
                  </strong>
                )}
              </div>
              {hoverFrame !== null && (
                <div className="hover-playhead" style={{ left: `${(hoverFrame / Math.max(1, maxFrame)) * 100}%` }} />
              )}
              <div className="active-playhead" style={{ left: `${progress * 100}%` }}>
                <div className="playhead-cap" /><div className="playhead-label">F {currentFrame}</div>
              </div>
            </div>
          </div>
        </section>
      )}

      {isDraggingFile && <div className="drop-overlay"><span className="drop-glyph">↓</span><strong>Drop to inspect this video</strong><span>Nothing leaves your computer</span></div>}
    </main>
  );
}
