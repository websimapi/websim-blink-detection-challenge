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
let detectionActive = false;

// EAR (Eye Aspect Ratio) threshold for blink detection - adjusted to be more sensitive
const EAR_THRESHOLD = 0.23;

// Left and right eye landmark indices for MediaPipe FaceMesh
// Updated to use more reliable points for eye tracking
const LEFT_EYE_INDICES = [
    // Upper eyelid
    159, 145, 144, 160, 33, 133,
    // Lower eyelid
    33, 7, 163, 144, 153, 145
];

const RIGHT_EYE_INDICES = [
    // Upper eyelid
    386, 374, 373, 390, 362, 263,
    // Lower eyelid
    362, 382, 381, 380, 374, 373
];

// Initialize the webcam
async function setupCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { width: 640, height: 480, facingMode: "user" },
            audio: false
        });
        video.srcObject = stream;
        
        return new Promise((resolve) => {
            video.onloadedmetadata = () => {
                resolve(video);
            };
        });
    } catch (error) {
        console.error("Camera access error:", error);
        alert("Cannot access camera. Please ensure camera permissions are granted and no other app is using it.");
        throw error;
    }
}

// Set up canvas dimensions
function setupCanvas() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
}

// Load TensorFlow.js model
async function loadModel() {
    try {
        // Fix: Use the correct API structure for loading the model
        // The SupportedPackages.mediapipeFacemesh syntax is outdated
        const model = await faceLandmarksDetection.createDetector(
            faceLandmarksDetection.SupportedModels.MediaPipeFaceMesh,
            { 
                runtime: 'tfjs',
                maxFaces: 1,
                refineLandmarks: true,
                detectionConfidence: 0.8
            }
        );
        detectionActive = true;
        return model;
    } catch (error) {
        console.error("Model loading error:", error);
        alert("Failed to load face tracking model. Please try reloading the page.");
        throw error;
    }
}

// Improved EAR calculation using convex hull approach
function calculateEAR(landmarks, eyeIndices) {
    try {
        // Get the eye landmarks - adapting to new format
        const points = eyeIndices.map(index => {
            const point = landmarks.find(lm => lm.name === `face_landmark_${index}`);
            return point ? [point.x, point.y] : [0, 0];
        });
        
        // Calculate vertical distances (average of multiple points for robustness)
        const verticalDist1 = distance(points[1], points[7]);
        const verticalDist2 = distance(points[2], points[10]);
        const verticalDist3 = distance(points[3], points[9]);
        
        // Calculate horizontal distance
        const horizontalDist = distance(points[0], points[5]);
        
        if (horizontalDist === 0) return 1.0; // Prevent division by zero
        
        // Calculate EAR (average of multiple vertical distances)
        const ear = (verticalDist1 + verticalDist2 + verticalDist3) / (3 * horizontalDist);
        
        return ear;
    } catch (error) {
        console.error("EAR calculation error:", error);
        return 1.0; // Return default value on error
    }
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
    // Detect completed blinks (eye was closed and now is open)
    const leftBlinked = prevLeftEyeClosed && !isLeftEyeClosed;
    const rightBlinked = prevRightEyeClosed && !isRightEyeClosed;
    
    if (leftBlinked && rightBlinked) {
        bothBlinkCount++;
        bothCounter.textContent = bothBlinkCount;
        addBlinkToHistory('both');
        
        // Visual feedback for both eyes blink
        leftEyeStatus.classList.add('blink-animation');
        rightEyeStatus.classList.add('blink-animation');
        setTimeout(() => {
            leftEyeStatus.classList.remove('blink-animation');
            rightEyeStatus.classList.remove('blink-animation');
        }, 300);
    } else if (leftBlinked) {
        leftBlinkCount++;
        leftCounter.textContent = leftBlinkCount;
        addBlinkToHistory('left');
        
        // Visual feedback for left eye blink
        leftEyeStatus.classList.add('blink-animation');
        setTimeout(() => {
            leftEyeStatus.classList.remove('blink-animation');
        }, 300);
    } else if (rightBlinked) {
        rightBlinkCount++;
        rightCounter.textContent = rightBlinkCount;
        addBlinkToHistory('right');
        
        // Visual feedback for right eye blink
        rightEyeStatus.classList.add('blink-animation');
        setTimeout(() => {
            rightEyeStatus.classList.remove('blink-animation');
        }, 300);
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

// Improved pupil visualization
function updatePupilVisualization() {
    // Map eye openness to pupil scale with more pronounced effect
    const leftPupilScale = Math.max(0.1, leftEyeOpenness * 2);
    const rightPupilScale = Math.max(0.1, rightEyeOpenness * 2);
    
    // Update left eye
    if (isLeftEyeClosed) {
        leftEyeStatus.classList.add('eye-closed');
        leftPupil.style.transform = 'scaleY(0.1)';
    } else {
        leftEyeStatus.classList.remove('eye-closed');
        leftPupil.style.transform = `scale(${leftPupilScale})`;
    }
    
    // Update right eye
    if (isRightEyeClosed) {
        rightEyeStatus.classList.add('eye-closed');
        rightPupil.style.transform = 'scaleY(0.1)';
    } else {
        rightEyeStatus.classList.remove('eye-closed');
        rightPupil.style.transform = `scale(${rightPupilScale})`;
    }
}

// Main detection function with improved error handling and tracking
async function detectFace(model) {
    try {
        // Get predictions from the model - updated API call
        const predictions = await model.estimateFaces(video);
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (predictions.length > 0) {
            // Updated to get landmarks from the new API structure
            const landmarks = predictions[0].keypoints;
            
            // Calculate EAR for both eyes with improved method
            const leftEAR = calculateEAR(landmarks, LEFT_EYE_INDICES);
            const rightEAR = calculateEAR(landmarks, RIGHT_EYE_INDICES);
            
            // Update eye openness values (0-1 range) with smoother transition
            leftEyeOpenness = Math.min(1, leftEAR / 0.25);
            rightEyeOpenness = Math.min(1, rightEAR / 0.25);
            
            // Store previous states
            prevLeftEyeClosed = isLeftEyeClosed;
            prevRightEyeClosed = isRightEyeClosed;
            
            // Determine if eyes are closed with improved threshold
            isLeftEyeClosed = leftEAR < EAR_THRESHOLD;
            isRightEyeClosed = rightEAR < EAR_THRESHOLD;
            
            // Draw facial landmarks for visualization
            drawFacialLandmarks(landmarks);
            
            // Update status indicators
            document.body.classList.remove('no-face-detected');
            
            // Debug info on canvas
            ctx.fillStyle = 'white';
            ctx.font = '12px Arial';
            ctx.fillText(`Left EAR: ${leftEAR.toFixed(3)} (${isLeftEyeClosed ? 'CLOSED' : 'OPEN'})`, 10, 20);
            ctx.fillText(`Right EAR: ${rightEAR.toFixed(3)} (${isRightEyeClosed ? 'CLOSED' : 'OPEN'})`, 10, 40);
        } else {
            // No face detected
            document.body.classList.add('no-face-detected');
        }
        
        // Update counters if a blink is detected
        updateBlinkCounters();
        
        // Update pupil visualization
        updatePupilVisualization();
        
        // Update time without blinking
        const timeSinceLastBlink = (Date.now() - lastBlinkTime) / 1000;
        noBlinkTime.textContent = `${timeSinceLastBlink.toFixed(1)}s`;
    } catch (error) {
        console.error('Face detection error:', error);
    }
    
    // Request next frame
    requestAnimationFrame(() => detectFace(model));
}

// Draw facial landmarks with improved visualization
function drawFacialLandmarks(landmarks) {
    // Draw eye contours with higher visibility
    drawEyeContour(landmarks, LEFT_EYE_INDICES, isLeftEyeClosed);
    drawEyeContour(landmarks, RIGHT_EYE_INDICES, isRightEyeClosed);
}

// Helper function to draw eye contours
function drawEyeContour(landmarks, indices, isClosed) {
    ctx.beginPath();
    indices.forEach((index, i) => {
        const point = landmarks.find(lm => lm.name === `face_landmark_${index}`);
        const [x, y] = point ? [point.x, point.y] : [0, 0];
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = isClosed ? 'rgba(255, 0, 0, 0.5)' : 'rgba(0, 255, 0, 0.3)';
    ctx.fill();
    ctx.strokeStyle = isClosed ? 'red' : 'green';
    ctx.lineWidth = 2;
    ctx.stroke();
}

// Initialize application
async function init() {
    try {
        // Show loading message
        const statusElement = document.createElement('div');
        statusElement.id = 'status-message';
        statusElement.textContent = 'Initializing face tracking...';
        document.body.appendChild(statusElement);
        
        await setupCamera();
        setupCanvas();
        
        // Update status
        statusElement.textContent = 'Loading AI model...';
        
        const model = await loadModel();
        
        // Update status and remove message
        statusElement.textContent = 'Tracking active';
        setTimeout(() => {
            statusElement.classList.add('fade-out');
            setTimeout(() => statusElement.remove(), 1000);
        }, 2000);
        
        // Start face detection
        detectFace(model);
        
        // Update timeline periodically
        setInterval(renderBlinkTimeline, 1000);
    } catch (error) {
        console.error('Error initializing app:', error);
        const statusElement = document.getElementById('status-message') || document.createElement('div');
        statusElement.id = 'status-message';
        statusElement.className = 'error';
        statusElement.textContent = `Error: ${error.message}. Please reload the page.`;
        document.body.appendChild(statusElement);
    }
}

// Start the application when the page is loaded
window.addEventListener('DOMContentLoaded', init);