import * as tf from '@tensorflow/tfjs';
import * as faceLandmarksDetection from '@tensorflow-models/face-landmarks-detection';

// DOM elements
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const leftCounter = document.getElementById('leftCounter');
const rightCounter = document.getElementById('rightCounter');
const bothCounter = document.getElementById('bothCounter');
const totalBlinks = document.getElementById('totalBlinks');
const blinkRate = document.getElementById('blinkRate');
const noBlinkTime = document.getElementById('noBlinkTime');
const blinkTimeline = document.getElementById('blinkTimeline');
const leftEyeStatus = document.getElementById('leftEyeStatus');
const rightEyeStatus = document.getElementById('rightEyeStatus');
const leftPupil = document.getElementById('leftPupil');
const rightPupil = document.getElementById('rightPupil');

// Counters and state
let leftBlinkCount = 0;
let rightBlinkCount = 0;
let bothBlinkCount = 0;
let lastBlinkTime = Date.now();
let startTime = Date.now();
let blinkHistory = [];
let isLeftEyeClosed = false;
let isRightEyeClosed = false;
let prevLeftEyeClosed = false;
let prevRightEyeClosed = false;
let leftEyeOpenness = 1.0;
let rightEyeOpenness = 1.0;

// EAR (Eye Aspect Ratio) threshold for blink detection
const EAR_THRESHOLD = 0.2;

// Initialize the webcam
async function setupCamera() {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480 },
        audio: false
    });
    video.srcObject = stream;
    
    return new Promise((resolve) => {
        video.onloadedmetadata = () => {
            resolve(video);
        };
    });
}

// Set up canvas dimensions
function setupCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// Load TensorFlow.js model
async function loadModel() {
    const model = await faceLandmarksDetection.load(
        faceLandmarksDetection.SupportedPackages.mediapipeFacemesh,
        { maxFaces: 1 }
    );
    return model;
}

// Calculate Eye Aspect Ratio (EAR)
function calculateEAR(landmarks, isLeftEye) {
    // Define eye landmarks indices (specific to MediaPipe FaceMesh)
    let eyePoints;
    if (isLeftEye) {
        // Left eye landmarks (simplified for clarity)
        eyePoints = [33, 160, 158, 133, 153, 144];
    } else {
        // Right eye landmarks (simplified for clarity)
        eyePoints = [362, 385, 387, 263, 373, 380];
    }
    
    // Get the coordinates of the eye landmarks
    const points = eyePoints.map(index => landmarks[index]);
    
    // Calculate the vertical distances
    const verticalDist1 = distance(points[1], points[4]);
    const verticalDist2 = distance(points[2], points[5]);
    
    // Calculate the horizontal distance
    const horizontalDist = distance(points[0], points[3]);
    
    // Calculate EAR
    const ear = (verticalDist1 + verticalDist2) / (2 * horizontalDist);
    
    return ear;
}

// Calculate distance between two points
function distance(point1, point2) {
    return Math.sqrt(
        Math.pow(point2[0] - point1[0], 2) + 
        Math.pow(point2[1] - point1[1], 2)
    );
}

// Update blink counters
function updateBlinkCounters() {
    const leftBlinked = prevLeftEyeClosed && !isLeftEyeClosed;
    const rightBlinked = prevRightEyeClosed && !isRightEyeClosed;
    
    if (leftBlinked && rightBlinked) {
        bothBlinkCount++;
        bothCounter.textContent = bothBlinkCount;
        addBlinkToHistory('both');
    } else if (leftBlinked) {
        leftBlinkCount++;
        leftCounter.textContent = leftBlinkCount;
        addBlinkToHistory('left');
    } else if (rightBlinked) {
        rightBlinkCount++;
        rightCounter.textContent = rightBlinkCount;
        addBlinkToHistory('right');
    }
    
    // Update total blinks
    const total = leftBlinkCount + rightBlinkCount + bothBlinkCount;
    totalBlinks.textContent = total;
    
    // Update blink rate (blinks per minute)
    const elapsedMinutes = (Date.now() - startTime) / 60000;
    const rate = total / elapsedMinutes;
    blinkRate.textContent = `${rate.toFixed(1)} blinks/min`;
    
    // Update time without blinking
    if (leftBlinked || rightBlinked) {
        lastBlinkTime = Date.now();
    }
    
    const timeSinceLastBlink = (Date.now() - lastBlinkTime) / 1000;
    noBlinkTime.textContent = `${timeSinceLastBlink.toFixed(1)}s`;
}

// Add blink to history timeline
function addBlinkToHistory(type) {
    const now = Date.now();
    blinkHistory.push({ time: now, type });
    
    // Keep only the last 30 seconds of history
    blinkHistory = blinkHistory.filter(item => now - item.time < 30000);
    
    // Render timeline
    renderBlinkTimeline();
}

// Render blink timeline
function renderBlinkTimeline() {
    blinkTimeline.innerHTML = '';
    const timelineWidth = blinkTimeline.offsetWidth;
    const now = Date.now();
    
    blinkHistory.forEach(blink => {
        const timeDiff = now - blink.time;
        const position = timelineWidth - (timeDiff / 30000) * timelineWidth;
        
        const marker = document.createElement('div');
        marker.className = `blink-marker ${blink.type}`;
        marker.style.left = `${position}px`;
        blinkTimeline.appendChild(marker);
    });
}

// Update pupil visualization based on eye openness
function updatePupilVisualization() {
    // Map eye openness to pupil scale
    const leftPupilScale = Math.max(0.1, leftEyeOpenness * 2.5);
    const rightPupilScale = Math.max(0.1, rightEyeOpenness * 2.5);
    
    // Update left eye
    if (isLeftEyeClosed) {
        leftEyeStatus.classList.add('eye-closed');
    } else {
        leftEyeStatus.classList.remove('eye-closed');
        leftPupil.style.transform = `scale(${leftPupilScale})`;
    }
    
    // Update right eye
    if (isRightEyeClosed) {
        rightEyeStatus.classList.add('eye-closed');
    } else {
        rightEyeStatus.classList.remove('eye-closed');
        rightPupil.style.transform = `scale(${rightPupilScale})`;
    }
}

// Main detection function
async function detectFace(model) {
    // Get predictions from the model
    const predictions = await model.estimateFaces({
        input: video,
        returnTensors: false,
        flipHorizontal: false,
        predictIrises: true
    });
    
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    if (predictions.length > 0) {
        const landmarks = predictions[0].scaledMesh;
        
        // Calculate EAR for both eyes
        const leftEAR = calculateEAR(landmarks, true);
        const rightEAR = calculateEAR(landmarks, false);
        
        // Update eye openness values (0-1 range)
        leftEyeOpenness = Math.min(1, leftEAR / 0.3);
        rightEyeOpenness = Math.min(1, rightEAR / 0.3);
        
        // Store previous states
        prevLeftEyeClosed = isLeftEyeClosed;
        prevRightEyeClosed = isRightEyeClosed;
        
        // Determine if eyes are closed
        isLeftEyeClosed = leftEAR < EAR_THRESHOLD;
        isRightEyeClosed = rightEAR < EAR_THRESHOLD;
        
        // Draw facial landmarks for visualization
        drawFacialLandmarks(landmarks);
        
        // Update counters if a blink is detected
        updateBlinkCounters();
        
        // Update pupil visualization
        updatePupilVisualization();
    }
    
    // Update time without blinking
    const timeSinceLastBlink = (Date.now() - lastBlinkTime) / 1000;
    noBlinkTime.textContent = `${timeSinceLastBlink.toFixed(1)}s`;
    
    // Request next frame
    requestAnimationFrame(() => detectFace(model));
}

// Draw facial landmarks for debugging
function drawFacialLandmarks(landmarks) {
    ctx.fillStyle = 'rgba(0, 255, 0, 0.2)';
    
    // Draw left and right eye contours
    const leftEyeIndices = [33, 160, 158, 133, 153, 144];
    const rightEyeIndices = [362, 385, 387, 263, 373, 380];
    
    ctx.beginPath();
    leftEyeIndices.forEach((index, i) => {
        const [x, y] = landmarks[index];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = isLeftEyeClosed ? 'rgba(255, 0, 0, 0.5)' : 'rgba(0, 255, 0, 0.2)';
    ctx.fill();
    
    ctx.beginPath();
    rightEyeIndices.forEach((index, i) => {
        const [x, y] = landmarks[index];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = isRightEyeClosed ? 'rgba(255, 0, 0, 0.5)' : 'rgba(0, 255, 0, 0.2)';
    ctx.fill();
}

// Initialize application
async function init() {
    try {
        await setupCamera();
        setupCanvas();
        const model = await loadModel();
        detectFace(model);
        
        // Update timeline periodically (for time-based rendering)
        setInterval(renderBlinkTimeline, 1000);
    } catch (error) {
        console.error('Error initializing app:', error);
    }
}

// Start the application when the page is loaded
window.addEventListener('DOMContentLoaded', init);

