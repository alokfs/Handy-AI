const DEFAULT_BACKEND_URL = 'http://127.0.0.1:5000';
const API_CANDIDATES = (() => {
    const candidates = [];
    const { origin, protocol } = window.location;

    if (protocol === 'http:' || protocol === 'https:') {
        candidates.push(origin);
    }

    candidates.push(DEFAULT_BACKEND_URL, 'http://localhost:5000');
    return [...new Set(candidates)];
})();

let apiBaseUrl = DEFAULT_BACKEND_URL;

const frameImage = document.getElementById('frameImage');
const videoElement = document.getElementById('videoElement');
const outputArea = document.getElementById('outputArea');
const clearBtn = document.getElementById('clearBtn');
const statusEl = document.getElementById('status');

const captureCanvas = document.createElement('canvas');
const captureCtx = captureCanvas.getContext('2d');

const FRAME_POLL_INTERVAL_MS = 60;
const OUTPUT_POLL_INTERVAL_MS = 500;
const UPLOAD_INTERVAL_MS = 80;
const HEALTH_CHECK_INTERVAL_MS = 2000;
const MAX_UPLOAD_WIDTH = 960;
const JPEG_QUALITY = 0.65;

let backendAvailable = false;
let localCameraActive = false;
let pollingStarted = false;
let frameRequestInProgress = false;
let uploadInProgress = false;

function setStatus(message, isActive) {
    statusEl.textContent = message;
    statusEl.className = `status ${isActive ? 'active' : 'inactive'}`;
}

function showLocalPreview() {
    if (!localCameraActive) {
        return;
    }

    videoElement.style.display = '';
    frameImage.style.display = 'none';
}

function showProcessedFrame(frame) {
    frameImage.src = `data:image/jpeg;base64,${frame}`;
    frameImage.style.display = '';
    videoElement.style.display = 'none';
}

async function pingBackend(baseUrl) {
    try {
        const response = await fetch(`${baseUrl}/api/health`);
        if (!response.ok) {
            throw new Error(`Health check failed with ${response.status}`);
        }

        const data = await response.json();
        return data.status === 'ok' ? data : null;
    } catch (error) {
        console.warn('Backend health check failed:', error);
        return null;
    }
}

async function checkBackend() {
    for (const candidate of API_CANDIDATES) {
        const health = await pingBackend(candidate);
        if (health) {
            apiBaseUrl = candidate;
            return health;
        }
    }

    return null;
}

async function initLocalCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        console.warn('getUserMedia is not supported in this browser.');
        return;
    }

    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        localCameraActive = true;
        videoElement.srcObject = stream;
        showLocalPreview();
        setStatus('Camera connected. Sending frames to app.py...', true);
    } catch (error) {
        console.error('Could not start local camera:', error);
    }
}

async function updateFrame() {
    if (!backendAvailable || frameRequestInProgress) {
        return;
    }

    frameRequestInProgress = true;
    try {
        const response = await fetch(`${apiBaseUrl}/api/frame`);
        if (!response.ok) {
            throw new Error(`Frame request failed with ${response.status}`);
        }

        const data = await response.json();
        if (data.frame) {
            showProcessedFrame(data.frame);
        } else if (localCameraActive) {
            showLocalPreview();
        }

        if (data.statusText) {
            setStatus(data.statusText, true);
        }
    } catch (error) {
        console.error('Error fetching frame:', error);
        setStatus('Connected to backend, but frame updates failed.', false);
    } finally {
        frameRequestInProgress = false;
    }
}

async function updateOutput() {
    if (!backendAvailable) {
        return;
    }

    try {
        const response = await fetch(`${apiBaseUrl}/api/output`);
        if (!response.ok) {
            throw new Error(`Output request failed with ${response.status}`);
        }

        const data = await response.json();
        outputArea.textContent = data.text || '';
    } catch (error) {
        console.error('Error fetching output:', error);
    }
}

async function sendLocalFrame() {
    if (!backendAvailable || !localCameraActive || uploadInProgress) {
        return;
    }

    const sourceWidth = videoElement.videoWidth || videoElement.clientWidth;
    const sourceHeight = videoElement.videoHeight || videoElement.clientHeight;
    if (!sourceWidth || !sourceHeight) {
        return;
    }

    const scale = Math.min(1, MAX_UPLOAD_WIDTH / sourceWidth);
    const width = Math.round(sourceWidth * scale);
    const height = Math.round(sourceHeight * scale);

    captureCanvas.width = width;
    captureCanvas.height = height;
    captureCtx.drawImage(videoElement, 0, 0, width, height);

    uploadInProgress = true;
    try {
        const response = await fetch(`${apiBaseUrl}/api/upload`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                frame: captureCanvas.toDataURL('image/jpeg', JPEG_QUALITY)
            })
        });

        if (!response.ok) {
            throw new Error(`Upload failed with ${response.status}`);
        }
    } catch (error) {
        console.error('Error uploading frame:', error);
    } finally {
        uploadInProgress = false;
    }
}

function startPolling() {
    if (pollingStarted) {
        return;
    }

    pollingStarted = true;
    setInterval(updateFrame, FRAME_POLL_INTERVAL_MS);
    setInterval(updateOutput, OUTPUT_POLL_INTERVAL_MS);
    setInterval(sendLocalFrame, UPLOAD_INTERVAL_MS);
}

async function waitForBackend() {
    const health = await checkBackend();
    if (!health) {
        setStatus('Backend not running. Start python app.py on port 5000.', false);
        setTimeout(waitForBackend, HEALTH_CHECK_INTERVAL_MS);
        return;
    }

    backendAvailable = true;
    startPolling();
    await initLocalCamera();

    if (!localCameraActive && !health.serverCameraEnabled) {
        setStatus(
            'Backend connected. Allow browser camera access or enable HANDY_AI_USE_SERVER_CAMERA=1.',
            false
        );
    } else if (!localCameraActive && !health.serverCameraAvailable) {
        setStatus(
            health.serverCameraError || 'Backend connected, but no camera is available yet.',
            false
        );
    } else if (!health.aiConfigured) {
        setStatus('Backend connected. Set GOOGLE_API_KEY to enable AI solving.', true);
    } else {
        setStatus('Backend connected. Waiting for camera frames...', true);
    }

    updateFrame();
    updateOutput();
}

clearBtn.addEventListener('click', async () => {
    try {
        const response = await fetch(`${apiBaseUrl}/api/clear`, { method: 'POST' });
        if (!response.ok) {
            throw new Error(`Clear request failed with ${response.status}`);
        }

        outputArea.textContent = '';
        if (localCameraActive) {
            showLocalPreview();
        }
        setStatus('Canvas cleared.', true);
    } catch (error) {
        console.error('Error clearing canvas:', error);
        setStatus('Could not clear the canvas.', false);
    }
});

setStatus('Connecting to app.py backend...', false);
waitForBackend();
