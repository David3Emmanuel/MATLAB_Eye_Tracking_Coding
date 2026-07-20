%% Simplest Live Gaze-Selection Demo
% Goal: prove the actual interaction loop works — look at a region,
% dwell on it, it gets "selected" — using the SAME eye-crop pipeline
% that scored 95.6% in gaze_accuracy_test.m.
%
% This is deliberately the smallest possible version of the real app:
%   - 4 regions instead of 9 (matches your "4 max levels" instinct —
%     coarser regions = more robust to noisy webcam signal)
%   - short calibration (10 frames/region, no train/test split — for
%     a live demo we want every calibration frame informing the model)
%   - continuous classification loop with DWELL-TO-CONFIRM + visual
%     progress indicator, per your Key Decisions (#2)
%   - selecting a region just prints/displays a placeholder label —
%     swap those labels for real menu categories once this feels good
%
% Requires: Computer Vision Toolbox + webcam. Same as the POC.

clear; clc;

%% Config
nRegions      = 4;                 % 2x2 grid
framesPerCal  = 10;                % calibration frames per region
dwellSeconds  = 1.5;                % how long you must look to confirm
loopHz        = 8;                 % classification rate during live phase
labels        = {'Snippets', 'Variables', 'Loops', 'Run'}; % placeholders
cursorSmoothing = 0.25;             % 0-1, lower = smoother/slower cursor, higher = snappier
cameraAlpha     = 0.18;              % 0-1, faintness of background webcam feed (0=invisible)
fpsSmoothingA   = 0.9;               % 0-1, higher = smoother/slower-reacting FPS readout
liveDetectEveryN   = 5;   % re-run FULL face detection every N live frames; other frames
                           % reuse the cached bbox ("tracking") instead of re-scanning —
                           % this is the single biggest speed win below
liveDownsample     = 0.5; % downsample factor applied before face detection during live phase
cameraDisplayScale = 0.4; % downsample factor for the background feed image (display only)

%% Setup webcam + detectors (identical to POC)
cam = webcam;
faceDetector = vision.CascadeObjectDetector();
eyeDetector  = vision.CascadeObjectDetector('EyePairBig');
faceDetector.MinSize = [60 60]; % skip scanning tiny scales — biggest single lever on
                                 % cascade detector runtime, safe since a face filling
                                 % a webcam frame is almost always well above this

scrSize = get(0, 'ScreenSize');
scrW = scrSize(3); scrH = scrSize(4);

% 2x2 grid centers
[gx, gy] = meshgrid(1:2, 1:2);
ptX = (gx(:) - 0.5) / 2 * scrW;
ptY = (gy(:) - 0.5) / 2 * scrH;

targetSize = [30 60];

%% --- Calibration phase (reuses POC capture logic) ---
fig = figure('Units','pixels','Position',[1 1 scrW scrH], ...
             'MenuBar','none','ToolBar','none','Color','k');

classVecs = cell(nRegions,1);

for p = 1:nRegions
    clf(fig);
    ax = axes('Parent', fig, 'Position', [0 0 1 1], 'Color', 'k');
    hold(ax, 'on'); axis(ax, [0 scrW 0 scrH]); axis(ax, 'off');
    plot(ax, ptX(p), scrH - ptY(p), 'r.', 'MarkerSize', 60);
    title(ax, sprintf('Calibrating: look at "%s" (%d of %d)', ...
          labels{p}, p, nRegions), 'Color', 'w', 'FontSize', 18);
    drawnow;
    pause(1.0);

    vecs = [];
    for f = 1:framesPerCal
        frame = snapshot(cam);
        crop = extractEyeCrop(frame, faceDetector, eyeDetector);
        v = cropToVector(crop, targetSize);
        if ~isempty(v)
            vecs = [vecs; v]; %#ok<AGROW>
        end
        pause(0.1);
    end

    if isempty(vecs)
        close(fig); clear cam;
        error('No valid eye crops captured for region %d. Check lighting/webcam position.', p);
    end
    classVecs{p} = vecs;
end

classMeans = zeros(nRegions, size(classVecs{1},2));
for p = 1:nRegions
    classMeans(p,:) = mean(classVecs{p}, 1);
end

fprintf('Calibration complete. Starting live demo — look at a box and hold your gaze.\n');
fprintf('Close the figure window to exit.\n');

%% --- Live phase: continuous classification + dwell-to-confirm UI ---
clf(fig);
ax = axes('Parent', fig, 'Position', [0 0 1 1], 'Color', 'k');
hold(ax, 'on'); axis(ax, [0 scrW 0 scrH]); axis(ax, 'off');
ax.YDir = 'normal'; % keep consistent with the scrH-ptY(p) convention used below

% Faint background webcam feed — created FIRST so it renders behind
% everything else. AlphaData controls how "faint" it looks; CData gets
% refreshed every loop iteration with the live frame.
camImg = image(ax, 'CData', zeros(2,2,3,'uint8'), ...
    'XData', [0 scrW], 'YData', [scrH 0], 'AlphaData', cameraAlpha);

boxW = scrW/2 * 0.8; boxH = scrH/2 * 0.8;
boxHandles = gobjects(nRegions,1);
progHandles = gobjects(nRegions,1);
for p = 1:nRegions
    cx = ptX(p); cy = scrH - ptY(p);
    boxHandles(p) = rectangle(ax, 'Position', [cx-boxW/2, cy-boxH/2, boxW, boxH], ...
        'FaceColor', [0.15 0.15 0.15], 'EdgeColor', 'w', 'LineWidth', 2);
    text(ax, cx, cy, labels{p}, 'Color', 'w', 'FontSize', 20, ...
        'HorizontalAlignment', 'center');
    progHandles(p) = rectangle(ax, 'Position', [cx-boxW/2, cy-boxH/2, 0, 8], ...
        'FaceColor', [0.2 0.9 0.3], 'EdgeColor', 'none');
end
statusText = text(ax, scrW/2, 30, '', 'Color', 'y', 'FontSize', 16, ...
    'HorizontalAlignment', 'center');

cursorPos = [scrW/2, scrH/2]; % smoothed cursor position, starts centered
cursorHandle = rectangle(ax, ...
    'Position', [cursorPos(1)-15, cursorPos(2)-15, 30, 30], ...
    'Curvature', [1 1], 'FaceColor', [1 0.2 0.2], 'EdgeColor', 'w', 'LineWidth', 1.5);

% HUD: FPS + a couple of other live-tuning stats, top-left corner
hudText = text(ax, 20, scrH - 30, '', 'Color', [0.6 0.6 0.6], ...
    'FontSize', 13, 'FontName', 'Consolas', 'HorizontalAlignment', 'left');

currentRegion = 0;
dwellStart = tic;
confirmed = false;
fpsSmoothed = loopHz;
lastFrameTime = tic;
cachedFaceBBox = [];  % tracked face location; [] forces a fresh detection
frameCounter = 0;

while ishandle(fig)
    dt = toc(lastFrameTime);
    lastFrameTime = tic;
    if dt > 0
        fpsSmoothed = fpsSmoothingA * fpsSmoothed + (1 - fpsSmoothingA) * (1/dt);
    end

    frame = snapshot(cam);
    camImg.CData = imresize(frame, cameraDisplayScale); % cheaper to draw than full-res

    frameCounter = frameCounter + 1;
    if mod(frameCounter, liveDetectEveryN) == 0
        cachedFaceBBox = []; % periodically force a fresh detection to correct drift
    end
    [crop, cachedFaceBBox] = extractEyeCropTracked(frame, faceDetector, eyeDetector, ...
        cachedFaceBBox, liveDownsample);
    v = cropToVector(crop, targetSize);

    hudText.String = sprintf('FPS: %5.1f   frame: %dx%d   dwell: %.1fs   tracking: %s', ...
        fpsSmoothed, size(frame,2), size(frame,1), dwellSeconds, ...
        string(~isempty(cachedFaceBBox)));

    if isempty(v)
        statusText.String = 'Eye not detected...';
        currentRegion = 0;
        dwellStart = tic;
    else
        d = vecnorm(classMeans - v, 2, 2);
        [~, region] = min(d);

        % --- Continuous cursor position: inverse-distance weighted blend
        % of all region centers, so the cursor glides instead of jumping
        % between only 4 fixed spots.
        w = 1 ./ (d + 1e-6);
        w = w / sum(w);
        targetX = sum(w .* ptX);
        targetY = sum(w .* (scrH - ptY));
        cursorPos = (1 - cursorSmoothing) * cursorPos + cursorSmoothing * [targetX, targetY];
        cursorHandle.Position = [cursorPos(1)-15, cursorPos(2)-15, 30, 30];

        if region ~= currentRegion
            currentRegion = region;
            dwellStart = tic;
            confirmed = false;
        end

        elapsed = toc(dwellStart);
        frac = min(elapsed / dwellSeconds, 1);
        statusText.String = sprintf('Looking at: %s (%.0f%%)', labels{region}, frac*100);

        for p = 1:nRegions
            cx = ptX(p); cy = scrH - ptY(p);
            if p == region
                boxHandles(p).FaceColor = [0.1 0.3 0.6];
                progHandles(p).Position = [cx-boxW/2, cy-boxH/2, boxW*frac, 8];
            else
                boxHandles(p).FaceColor = [0.15 0.15 0.15];
                progHandles(p).Position = [cx-boxW/2, cy-boxH/2, 0, 8];
            end
        end

        if frac >= 1 && ~confirmed
            confirmed = true;
            boxHandles(region).FaceColor = [0.2 0.9 0.3]; % flash green = selected
            fprintf('SELECTED: %s\n', labels{region});
            drawnow;
            pause(0.3); % brief flash so it's visible
            dwellStart = tic; % require a fresh dwell to re-select
        end
    end

    drawnow;
    pause(1/loopHz);
end

clear cam;
fprintf('Demo ended.\n');

%% --- Helper functions (unchanged from gaze_accuracy_test.m) ---
function crop = extractEyeCrop(frame, faceDetector, eyeDetector)
    crop = [];
    faceBBox = step(faceDetector, frame);
    if isempty(faceBBox)
        return;
    end
    areas = faceBBox(:,3) .* faceBBox(:,4);
    [~, idx] = max(areas);
    fb = faceBBox(idx,:);
    faceImg = imcrop(frame, fb);

    eyeBBox = step(eyeDetector, faceImg);
    if isempty(eyeBBox)
        return;
    end
    areas2 = eyeBBox(:,3) .* eyeBBox(:,4);
    [~, idx2] = max(areas2);
    eb = eyeBBox(idx2,:);
    crop = imcrop(faceImg, eb);
end

function [crop, faceBBoxOut] = extractEyeCropTracked(frame, faceDetector, eyeDetector, faceBBoxIn, downsampleFactor)
    % Same idea as extractEyeCrop, but avoids re-running the expensive
    % full-frame face scan every call. If faceBBoxIn is given, it's reused
    % directly (tracking); only when it's empty do we pay for a fresh
    % detection, and even then on a downsampled frame for speed.
    crop = [];
    faceBBoxOut = faceBBoxIn;

    if isempty(faceBBoxIn)
        small = imresize(frame, downsampleFactor);
        bbox = step(faceDetector, small);
        if isempty(bbox)
            return; % faceBBoxOut stays [] -> next call will retry full detection
        end
        areas = bbox(:,3) .* bbox(:,4);
        [~, idx] = max(areas);
        faceBBoxOut = bbox(idx,:) / downsampleFactor; % scale back to full-res coords
    end

    fb = faceBBoxOut;
    [h, w, ~] = size(frame);
    fb(1) = max(1, fb(1)); fb(2) = max(1, fb(2));
    fb(3) = min(fb(3), w - fb(1)); fb(4) = min(fb(4), h - fb(2));
    if fb(3) <= 1 || fb(4) <= 1
        faceBBoxOut = []; % bbox drifted off-frame -> force a fresh detection next time
        return;
    end

    faceImg = imcrop(frame, fb);
    eyeBBox = step(eyeDetector, faceImg);
    if isempty(eyeBBox)
        return; % keep the cached face bbox; just no eye crop this particular frame
    end
    areas2 = eyeBBox(:,3) .* eyeBBox(:,4);
    [~, idx2] = max(areas2);
    eb = eyeBBox(idx2,:);
    crop = imcrop(faceImg, eb);
end

function v = cropToVector(crop, targetSize)
    v = [];
    if isempty(crop)
        return;
    end
    gray = im2gray(crop);
    resized = imresize(gray, targetSize);
    v = double(reshape(resized, 1, []));
    v = v / 255;
end
