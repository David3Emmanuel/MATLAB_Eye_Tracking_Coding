%% Minimal Webcam Gaze Accuracy Test
% Goal: answer ONE question — can your webcam distinguish which of 9
% screen regions someone is looking at, using just eye-crop images?
%
% No Deep Learning Toolbox needed for this test. We use a simple
% nearest-neighbor classifier on eye-crop pixels as a quick proxy for
% "is there enough signal in this webcam feed at all." If this crude
% method gets reasonable accuracy, a trained CNN will do better.
% If this crude method fails badly, that's important to know NOW.
%
% Requires: Computer Vision Toolbox (for vision.CascadeObjectDetector)
% and a webcam. That's it.

clear; clc;

%% Config
gridRows = 3;
gridCols = 3;
framesPerPoint = 15;   % frames captured per calibration point
holdoutFrames = 5;     % of those, held out for testing (not used in "training")

%% Setup webcam and face/eye detector
cam = webcam;
faceDetector = vision.CascadeObjectDetector();               % face
eyeDetector  = vision.CascadeObjectDetector('EyePairBig');    % eye pair

scrSize = get(0, 'ScreenSize');  % [left bottom width height]
scrW = scrSize(3); scrH = scrSize(4);

% Build 9 target points (centers of a 3x3 grid across the screen)
[gx, gy] = meshgrid(1:gridCols, 1:gridRows);
ptX = (gx(:) - 0.5) / gridCols * scrW;
ptY = (gy(:) - 0.5) / gridRows * scrH;
nPoints = numel(ptX);

%% Collect data: show each point, capture eye crops while user looks at it
allCrops = cell(nPoints, framesPerPoint);
fig = figure('Units','pixels','Position',[1 1 scrW scrH], ...
             'MenuBar','none','ToolBar','none','Color','k');

for p = 1:nPoints
    clf(fig);
    ax = axes('Parent', fig, 'Position', [0 0 1 1], 'Color', 'k');
    hold(ax, 'on'); axis(ax, [0 scrW 0 scrH]); axis(ax, 'off');
    plot(ax, ptX(p), scrH - ptY(p), 'r.', 'MarkerSize', 60);
    title(ax, sprintf('Look at the dot — point %d of %d', p, nPoints), ...
          'Color', 'w', 'FontSize', 16);
    drawnow;
    pause(1.0); % settle time

    for f = 1:framesPerPoint
        frame = snapshot(cam);
        crop = extractEyeCrop(frame, faceDetector, eyeDetector);
        allCrops{p, f} = crop;  % may be [] if detection failed
        pause(0.1);
    end
end
close(fig);
clear cam;

%% Split train/test, resize crops to fixed size, flatten to vectors
targetSize = [30 60]; % rows x cols, eye-pair crops are wide
trainVecs = []; trainLabels = [];
testVecs  = []; testLabels  = [];

for p = 1:nPoints
    idx = randperm(framesPerPoint);
    testIdx  = idx(1:holdoutFrames);
    trainIdx = idx(holdoutFrames+1:end);

    for f = trainIdx
        v = cropToVector(allCrops{p,f}, targetSize);
        if ~isempty(v)
            trainVecs = [trainVecs; v]; %#ok<AGROW>
            trainLabels = [trainLabels; p]; %#ok<AGROW>
        end
    end
    for f = testIdx
        v = cropToVector(allCrops{p,f}, targetSize);
        if ~isempty(v)
            testVecs = [testVecs; v]; %#ok<AGROW>
            testLabels = [testLabels; p]; %#ok<AGROW>
        end
    end
end

fprintf('Valid training samples: %d / %d\n', size(trainVecs,1), nPoints*(framesPerPoint-holdoutFrames));
fprintf('Valid test samples: %d / %d\n', size(testVecs,1), nPoints*holdoutFrames);

if size(trainVecs,1) < nPoints || size(testVecs,1) < 1
    error(['Eye detection failed too often to run this test. ' ...
           'This itself is a useful (bad) result: your webcam/lighting ' ...
           'setup may not support reliable eye detection at all.']);
end

%% Classify test samples via nearest-neighbor to training mean per class
classMeans = zeros(nPoints, size(trainVecs,2));
for p = 1:nPoints
    rows = trainLabels == p;
    if any(rows)
        classMeans(p,:) = mean(trainVecs(rows,:), 1);
    else
        classMeans(p,:) = NaN;
    end
end

predLabels = zeros(size(testLabels));
for i = 1:size(testVecs,1)
    d = vecnorm(classMeans - testVecs(i,:), 2, 2);
    [~, predLabels(i)] = min(d);
end

%% Report accuracy
acc = mean(predLabels == testLabels);
fprintf('\n=== RESULT ===\n');
fprintf('9-region gaze classification accuracy: %.1f%%\n', acc*100);
fprintf('(chance level = %.1f%%)\n', 100/nPoints);

confMat = confusionmat(testLabels, predLabels);
disp('Confusion matrix (rows=true, cols=predicted):');
disp(confMat);

%% Interpretation guide (print, don't automate — you should eyeball this)
fprintf('\n--- How to read this ---\n');
fprintf('>70%% : promising, a real CNN should beat this comfortably.\n');
fprintf('40-70%%: workable IF you reduce to fewer regions (try 4-6).\n');
fprintf('<40%% : close to chance. Try better lighting/camera position\n');
fprintf('        before concluding gaze tracking is infeasible.\n');

%% --- Helper functions ---
function crop = extractEyeCrop(frame, faceDetector, eyeDetector)
    crop = [];
    faceBBox = step(faceDetector, frame);
    if isempty(faceBBox)
        return;
    end
    % use largest detected face
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

function v = cropToVector(crop, targetSize)
    v = [];
    if isempty(crop)
        return;
    end
    gray = im2gray(crop);
    resized = imresize(gray, targetSize);
    v = double(reshape(resized, 1, []));
    v = v / 255; % normalize
end
