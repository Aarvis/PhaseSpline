# FrameLine

FrameLine is a local, frame-accurate video viewer for checkpoint annotation and timeline inspection. Files stay on the computer and are never uploaded.

Older MPEG-4 Part 2 videos are converted automatically to a temporary browser-compatible copy by the local FFmpeg installation. This is the codec used by the LeHome episode videos.

## Start locally

Requires Node.js 22 or newer and FFmpeg available on `PATH`.

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\Video-Frames-Timeline-Viewer
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and drag a video onto the page.

On Windows, you can also double-click `start-viewer.bat` from this folder.

## LeHome dataset annotation mode

The local helper is configured by default for:

```text
D:\pretrain_lehome_all_garment_data_z180
```

From the opening screen, select **Start 25% sampling session**. FrameLine creates one persistent random sampling session covering the garment categories in this order:

1. Pant
2. Shorts
3. Top — long sleeve
4. Top — short sleeve

Exactly 25% of each category is targeted (1,045 episodes in total for the current 4,180-episode dataset). Episodes that already have a temporal checkpoint are excluded before random sampling. The saved session queue survives page reloads and can be resumed.

The dataset stores frames inside Parquet files, so FrameLine creates a temporary H.264 viewing copy for each selected episode. These temporary videos are kept outside the dataset and do not modify the source Parquet files.

Complete the contiguous annotation, select **Save episode checkpoint**, and then select **Next episode**. The checkpoint is written automatically to:

```text
<dataset>\annotations\temporal_checkpoints\chunk-NNN\episode_NNNNNN\checkpoints.json
```

The session tracker at the top reports sampled and completed episode counts overall and per garment category. Once all sampled episodes in the active category are complete, **Next episode** moves to the first remaining episode in the next category.

Category-specific checkpoint vocabularies are loaded from `public/garment-segment-labels.json`. Edit that file before beginning a production annotation session if the phase names need to change.

To use a different compatible dataset root for a launch:

```powershell
$env:FRAMELINE_DATASET_ROOT = "D:\path\to\dataset"
npm run dev
```

## Controls

- Drag and drop, or select **Open video**, to load a local video.
- Click or drag anywhere on the timeline to scrub.
- Press Left/Right Arrow to move exactly one source frame.
- Hold Left/Right Arrow to accelerate through 1×, 3×, 6×, and 12× frame steps.
- Press Space to play or pause.
- Enter an exact source frame in **Jump to frame**.
- Edit **Source FPS** when inspecting a non-MP4 file or overriding unusual metadata.
- Adjust **Video size** to scale the preview between 50% and 100% without changing the source video or frame indexing.
- The video stays centered at its native aspect ratio; Annotation Workflow sits vertically between Current Position and Temporal Segments.
- While annotating, dragging or editing previews the current boundary. Press **Enter** to commit that segment and advance to the next one.
- Use **Up/Down** to select any temporal-segment row, or click a row directly. After Enter, the following segment is selected by default.
- Press **Ctrl+Z** to cancel a pending boundary or restore the annotation state from before the latest Enter commit or Clear Boundaries action.
- A second timeline lane visualizes normalized progress inside every saved segment. Progress starts at 0 and normally reaches 1 with a constant slope.
- In **Annotate** mode, press **Space** at a frame to reverse the progress direction without leaving the segment; press Space again later to switch back to increasing. In **Scrub** mode, Space still controls playback.
- Saved checkpoints include direction-change keyframes and dense per-frame progress values for every segment.
- **Progress Edit** mode exposes the active segment's progress curve. Click in the progress lane to create a control point, drag vertically to change progress and horizontally to change its frame, then press **Enter** to commit it.
- Each committed control point splits the curve into independently sloped linear pieces. Add as many points as needed; the segment endpoints remain fixed at progress 0 and 1.
- Clicking directly inside any completed segment's progress bar automatically selects that segment and opens **Progress Edit** mode at the clicked frame and progress value.
- While dragging a progress point, hold **Shift** for vertical-only progress changes or **Ctrl** for horizontal-only frame changes. Point placement uses the progress lane's exact cursor coordinates.
- Use **Ctrl+Z** to undo and **Ctrl+Y** to redo committed segment or progress changes. Making a new edit after undo clears the redo history.
- Press **S** for Scrub mode, **A** for Annotate mode, or **P** for Progress Edit mode whenever focus is not inside an input field.
- In Progress Edit mode, scrub outside the progress lane to position the red frame cursor. Clicking the progress lane prepares a point at that cursor frame and the curve's existing progress value; the click itself changes neither axis. Dragging then adjusts the point relative to its starting position.

## Contiguous temporal annotation

After opening a video, use **Load JSON** to load either:

- A label-list JSON containing an ordered `labels` array.
- An existing `checkpoints.json` containing both `labels` and `segments`.

You can also select **Use sample labels** to load `public/sample-segments.json`, which contains the nine pants-folding phases used by the LeHome examples.

Select **Annotate**, choose a segment, and drag on the single timeline to set its exclusive end boundary. The first segment always starts at frame 0. Each following segment automatically starts at the previous segment's exclusive end, so overlaps and gaps cannot be created. Select any completed segment to edit its boundary numerically.

When every label covers the video through its final frame, **Save checkpoints.json** becomes available. Supported browsers show a local save picker; other browsers download the file. The generated JSON uses zero-based `[start_frame, end_frame_exclusive)` ranges and includes `end_frame_inclusive`, `num_frames`, summary, frame metadata, and complete-coverage flags.

For MP4, MOV, and M4V inputs, MP4Box reads the video track's exact sample count, duration, FPS, codec, and dimensions. WebM and other browser-supported inputs use video duration plus the editable FPS value.

## Build and test

```powershell
npm run build
npm test
```
