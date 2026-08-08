import {
    FilesetResolver,
    HandLandmarker,
} from 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/+esm';
import { GoogleGenAI } from 'https://cdn.jsdelivr.net/npm/@google/genai@latest/+esm';

const MEDIAPIPE_WASM_URL = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm';
const HAND_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';
const GEMINI_MODEL = 'gemini-3.6-flash';
const API_KEY_STORAGE_KEY = 'handy-ai-gemini-key';
const LANDMARK_SMOOTHING = 0.42;
const PALM_TRIANGLES = [[0, 1, 5], [0, 5, 9], [0, 9, 13], [0, 13, 17]];
const FINGERS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];

const video = document.getElementById('videoElement');
const board = document.getElementById('drawingCanvas');
const boardContext = board.getContext('2d');
const inkCanvas = document.createElement('canvas');
const inkContext = inkCanvas.getContext('2d');
const startButton = document.getElementById('startBtn');
const solveButton = document.getElementById('solveBtn');
const clearButton = document.getElementById('clearBtn');
const apiKeyInput = document.getElementById('apiKeyInput');
const toggleKeyButton = document.getElementById('toggleKeyBtn');
const statusElement = document.getElementById('status');
const boardHint = document.getElementById('boardHint');
const cameraIndicator = document.getElementById('cameraIndicator');
const canvasMessage = document.getElementById('canvasMessage');
const outputArea = document.getElementById('outputArea');
const solveState = document.getElementById('solveState');

let handLandmarker;
let cameraStream;
let cameraRunning = false;
let isSolving = false;
let hasDrawing = false;
let lastVideoTime = -1;
let smoothedLandmarks = null;
let previousDrawPoint = null;
let lastRenderedLandmarks = null;
let frameRequestId;
const heldGestures = new Map();

function setStatus(message, type = 'neutral') {
    statusElement.textContent = message;
    statusElement.dataset.type = type;
}

function setCameraState(active) {
    cameraIndicator.textContent = active ? 'Camera on' : 'Camera off';
    cameraIndicator.classList.toggle('is-active', active);
    startButton.textContent = active ? 'Restart camera' : 'Start camera';
    solveButton.disabled = !active;
    clearButton.disabled = !active;
}

function setSolveState(message, active = false) {
    solveState.textContent = message;
    solveState.classList.toggle('is-active', active);
}

function waitForVideoMetadata() {
    if (video.readyState >= HTMLMediaElement.HAVE_METADATA) {
        return Promise.resolve();
    }

    return new Promise((resolve) => {
        video.addEventListener('loadedmetadata', resolve, { once: true });
    });
}

async function initializeHandLandmarker() {
    if (handLandmarker) {
        return;
    }

    setStatus('Loading on-device hand tracking…');
    const vision = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_URL);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: HAND_MODEL_URL },
        runningMode: 'VIDEO',
        numHands: 1,
        minHandDetectionConfidence: 0.7,
        minHandPresenceConfidence: 0.65,
        minTrackingConfidence: 0.6,
    });
}

function resizeCanvases() {
    const width = video.videoWidth;
    const height = video.videoHeight;
    if (!width || !height || (board.width === width && board.height === height)) {
        return;
    }

    const previousInk = document.createElement('canvas');
    previousInk.width = inkCanvas.width;
    previousInk.height = inkCanvas.height;
    if (inkCanvas.width && inkCanvas.height) {
        previousInk.getContext('2d').drawImage(inkCanvas, 0, 0);
    }

    board.width = width;
    board.height = height;
    inkCanvas.width = width;
    inkCanvas.height = height;
    if (previousInk.width && previousInk.height) {
        inkContext.drawImage(previousInk, 0, 0, width, height);
    }
    configureInkBrush();
}

function configureInkBrush() {
    inkContext.lineCap = 'round';
    inkContext.lineJoin = 'round';
    inkContext.strokeStyle = '#fb5cdb';
    inkContext.lineWidth = Math.max(7, Math.round(board.width * 0.008));
}

function pointOnBoard(landmark) {
    return {
        x: (1 - landmark.x) * board.width,
        y: landmark.y * board.height,
    };
}

function smoothHand(landmarks) {
    if (!landmarks) {
        smoothedLandmarks = null;
        return null;
    }

    if (!smoothedLandmarks || smoothedLandmarks.length !== landmarks.length) {
        smoothedLandmarks = landmarks.map((point) => ({ ...point }));
        return smoothedLandmarks;
    }

    smoothedLandmarks = landmarks.map((point, index) => ({
        x: smoothedLandmarks[index].x + (point.x - smoothedLandmarks[index].x) * LANDMARK_SMOOTHING,
        y: smoothedLandmarks[index].y + (point.y - smoothedLandmarks[index].y) * LANDMARK_SMOOTHING,
        z: smoothedLandmarks[index].z + (point.z - smoothedLandmarks[index].z) * LANDMARK_SMOOTHING,
    }));
    return smoothedLandmarks;
}

function isFingerExtended(landmarks, tip, pip) {
    return landmarks[tip].y < landmarks[pip].y - 0.015;
}

function isThumbExtended(landmarks) {
    const thumbSpan = Math.abs(landmarks[4].x - landmarks[2].x);
    const thumbSegment = Math.abs(landmarks[4].x - landmarks[3].x);
    return thumbSpan > 0.1 && thumbSegment > 0.035;
}

function getFingerState(landmarks) {
    return {
        thumb: isThumbExtended(landmarks),
        index: isFingerExtended(landmarks, 8, 6),
        middle: isFingerExtended(landmarks, 12, 10),
        ring: isFingerExtended(landmarks, 16, 14),
        pinky: isFingerExtended(landmarks, 20, 18),
    };
}

function classifyGesture(fingers) {
    const raisedNonThumb = [fingers.index, fingers.middle, fingers.ring, fingers.pinky]
        .filter(Boolean).length;
    const isOpenPalm = fingers.thumb && raisedNonThumb === 4;
    const isFist = raisedNonThumb === 0;
    const isPause = fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky && fingers.thumb;
    const isWriting = fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky && !fingers.thumb;

    if (isOpenPalm) return 'solve';
    if (isFist) return 'clear';
    if (isPause) return 'pause';
    if (isWriting) return 'write';
    return 'tracking';
}

function gestureHeldFor(gesture, minimumMs) {
    const now = performance.now();
    const current = heldGestures.get(gesture);
    if (!current) {
        heldGestures.set(gesture, { startedAt: now, fired: false });
        return false;
    }

    if (current.fired || now - current.startedAt < minimumMs) {
        return false;
    }

    current.fired = true;
    return true;
}

function resetOtherHeldGestures(activeGesture) {
    for (const gesture of heldGestures.keys()) {
        if (gesture !== activeGesture) {
            heldGestures.delete(gesture);
        }
    }
}

function clearDrawing({ clearResponse = false } = {}) {
    inkContext.clearRect(0, 0, inkCanvas.width, inkCanvas.height);
    hasDrawing = false;
    previousDrawPoint = null;
    if (clearResponse) {
        outputArea.textContent = 'Your worked solution will appear here.';
        setSolveState('Ready when you are');
    }
}

function drawAt(point) {
    if (!previousDrawPoint) {
        previousDrawPoint = point;
        return;
    }

    inkContext.beginPath();
    inkContext.moveTo(previousDrawPoint.x, previousDrawPoint.y);
    inkContext.lineTo(point.x, point.y);
    inkContext.stroke();
    previousDrawPoint = point;
    hasDrawing = true;
}

function processGesture(landmarks) {
    if (!landmarks) {
        previousDrawPoint = null;
        heldGestures.clear();
        boardHint.textContent = 'Show one hand to the camera.';
        return;
    }

    const gesture = classifyGesture(getFingerState(landmarks));
    resetOtherHeldGestures(gesture);

    if (gesture === 'write') {
        drawAt(pointOnBoard(landmarks[8]));
        boardHint.textContent = 'Writing…';
        return;
    }

    previousDrawPoint = null;
    if (gesture === 'pause') {
        boardHint.textContent = 'Paused — show only your index finger to write.';
        return;
    }

    if (gesture === 'clear') {
        boardHint.textContent = 'Hold the fist to clear.';
        if (gestureHeldFor('clear', 450)) {
            clearDrawing();
            boardHint.textContent = 'Drawing cleared.';
            setStatus('Canvas cleared.', 'success');
        }
        return;
    }

    if (gesture === 'solve') {
        boardHint.textContent = 'Hold the open palm to solve.';
        if (gestureHeldFor('solve', 900)) {
            solveDrawing();
        }
        return;
    }

    boardHint.textContent = 'Tracking your hand…';
}

function drawLowPolyHand(landmarks) {
    if (!landmarks) {
        return;
    }

    const points = landmarks.map(pointOnBoard);
    boardContext.save();
    boardContext.fillStyle = 'rgba(237, 243, 255, 0.35)';
    boardContext.strokeStyle = 'rgba(19, 28, 50, 0.9)';
    boardContext.lineWidth = Math.max(2, Math.round(board.width * 0.002));

    for (const triangle of PALM_TRIANGLES) {
        boardContext.beginPath();
        boardContext.moveTo(points[triangle[0]].x, points[triangle[0]].y);
        boardContext.lineTo(points[triangle[1]].x, points[triangle[1]].y);
        boardContext.lineTo(points[triangle[2]].x, points[triangle[2]].y);
        boardContext.closePath();
        boardContext.fill();
        boardContext.stroke();
    }

    boardContext.lineCap = 'round';
    for (const finger of FINGERS) {
        boardContext.beginPath();
        boardContext.moveTo(points[finger[0]].x, points[finger[0]].y);
        for (let index = 1; index < finger.length; index += 1) {
            boardContext.lineTo(points[finger[index]].x, points[finger[index]].y);
        }
        boardContext.strokeStyle = 'rgba(230, 238, 255, 0.7)';
        boardContext.lineWidth = Math.max(9, Math.round(board.width * 0.014));
        boardContext.stroke();
        boardContext.strokeStyle = 'rgba(19, 28, 50, 0.9)';
        boardContext.lineWidth = Math.max(2, Math.round(board.width * 0.002));
        boardContext.stroke();

        for (const landmarkIndex of finger) {
            boardContext.beginPath();
            boardContext.arc(points[landmarkIndex].x, points[landmarkIndex].y, Math.max(3, board.width * 0.005), 0, Math.PI * 2);
            boardContext.fillStyle = 'rgba(19, 28, 50, 0.95)';
            boardContext.fill();
        }
    }
    boardContext.restore();
}

function renderBoard() {
    if (!board.width || !board.height) {
        return;
    }

    boardContext.clearRect(0, 0, board.width, board.height);
    boardContext.fillStyle = '#0b1020';
    boardContext.fillRect(0, 0, board.width, board.height);

    boardContext.drawImage(inkCanvas, 0, 0);
    drawLowPolyHand(lastRenderedLandmarks);
}

function renderLoop() {
    if (!cameraRunning) {
        return;
    }

    if (video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
        lastVideoTime = video.currentTime;
        const result = handLandmarker.detectForVideo(video, performance.now());
        const landmarks = result.landmarks?.[0] || null;
        lastRenderedLandmarks = smoothHand(landmarks);
        processGesture(lastRenderedLandmarks);
    }

    renderBoard();
    frameRequestId = requestAnimationFrame(renderLoop);
}

function createProblemImage() {
    const problemCanvas = document.createElement('canvas');
    problemCanvas.width = inkCanvas.width;
    problemCanvas.height = inkCanvas.height;
    const context = problemCanvas.getContext('2d');
    context.fillStyle = '#ffffff';
    context.fillRect(0, 0, problemCanvas.width, problemCanvas.height);
    context.drawImage(inkCanvas, 0, 0);
    return problemCanvas.toDataURL('image/png').split(',', 2)[1];
}

function cleanAiText(text) {
    return text
        .replace(/[#$*_`]/g, '')
        .replace(/\n{3,}/g, '\n\n')
        .trim();
}

async function solveDrawing() {
    if (isSolving) {
        return;
    }

    if (!hasDrawing) {
        outputArea.textContent = 'Write a problem first, then press “Solve drawing” or hold an open palm.';
        setSolveState('Waiting for a drawing');
        setStatus('There is no drawing to solve yet.', 'warning');
        return;
    }

    const apiKey = apiKeyInput.value.trim();
    if (!apiKey) {
        outputArea.textContent = 'Add your Gemini API key above to solve the drawing. The key stays in this browser tab and is never saved in this project.';
        setSolveState('Gemini key needed');
        setStatus('Add a Gemini API key to use AI solving.', 'warning');
        return;
    }

    isSolving = true;
    solveButton.disabled = true;
    setSolveState('Solving…', true);
    setStatus('Sending the drawing to Gemini…');
    outputArea.textContent = 'Solving your math problem…';

    try {
        const ai = new GoogleGenAI({ apiKey });
        const interaction = await ai.interactions.create({
            model: GEMINI_MODEL,
            input: [
                {
                    type: 'text',
                    text: 'Read the handwritten math problem in this image. Solve it exactly, show the important working steps, and state the final answer clearly. Use plain text only. Do not use Markdown or formatting symbols such as asterisks, hash signs, dollar signs, backticks, or underscores. If the writing is not readable, say what needs to be clearer.',
                },
                {
                    type: 'image',
                    mime_type: 'image/png',
                    data: createProblemImage(),
                },
            ],
        });
        const answer = cleanAiText(interaction.output_text || '');
        if (!answer) {
            throw new Error('Gemini returned no text response. Please try again with a clearer drawing.');
        }

        outputArea.textContent = answer;
        setSolveState('Solved');
        setStatus('Solution ready.', 'success');
    } catch (error) {
        outputArea.textContent = `Could not solve the drawing. ${error.message}`;
        setSolveState('Could not solve');
        setStatus('Gemini request failed. Check the key and try again.', 'error');
    } finally {
        isSolving = false;
        solveButton.disabled = !cameraRunning;
    }
}

async function startCamera() {
    startButton.disabled = true;
    try {
        if (cameraStream) {
            cameraStream.getTracks().forEach((track) => track.stop());
            cancelAnimationFrame(frameRequestId);
            cameraRunning = false;
        }

        await initializeHandLandmarker();
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
            audio: false,
        });
        video.srcObject = cameraStream;
        await waitForVideoMetadata();
        await video.play();
        resizeCanvases();
        cameraRunning = true;
        canvasMessage.hidden = true;
        setCameraState(true);
        setStatus('Camera connected. Show one hand to start.', 'success');
        boardHint.textContent = 'Show one hand to the camera.';
        renderLoop();
    } catch (error) {
        cameraRunning = false;
        setCameraState(false);
        canvasMessage.hidden = false;
        canvasMessage.textContent = 'Camera access is needed to use hand gestures.';
        setStatus(`Could not start the camera: ${error.message}`, 'error');
    } finally {
        startButton.disabled = false;
    }
}

function loadSessionKey() {
    try {
        apiKeyInput.value = sessionStorage.getItem(API_KEY_STORAGE_KEY) || '';
    } catch {
        // Private browsing can disable session storage. The field still works for this page view.
    }
}

apiKeyInput.addEventListener('input', () => {
    try {
        sessionStorage.setItem(API_KEY_STORAGE_KEY, apiKeyInput.value);
    } catch {
        // Do not block use of the app if storage is unavailable.
    }
});

toggleKeyButton.addEventListener('click', () => {
    const showing = apiKeyInput.type === 'text';
    apiKeyInput.type = showing ? 'password' : 'text';
    toggleKeyButton.textContent = showing ? 'Show' : 'Hide';
    toggleKeyButton.setAttribute('aria-label', showing ? 'Show API key' : 'Hide API key');
});

startButton.addEventListener('click', startCamera);
solveButton.addEventListener('click', solveDrawing);
clearButton.addEventListener('click', () => {
    clearDrawing({ clearResponse: true });
    setStatus('Canvas cleared.', 'success');
    boardHint.textContent = 'Show one hand to the camera.';
});

window.addEventListener('beforeunload', () => {
    cameraStream?.getTracks().forEach((track) => track.stop());
    handLandmarker?.close?.();
});

loadSessionKey();
setCameraState(false);
setStatus('Ready. Start the camera to load hand tracking.');
