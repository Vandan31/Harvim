// Global Variables
let webcamStream = null;
let isWebcamActive = false;
let decoderReady = false;

// Statistics
let stats = {
    totalDecoded: 0,
    totalFailed: 0,
    confidenceScores: []
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    initializePage();
});

function initializePage() {
    // Check decoder status
    checkDecoderStatus();
    
    // Setup event listeners
    setupUploadListeners();
    setupWebcamListeners();
    setupWatermarkListeners();
    setupButtonListeners();
    
    // Load stats from localStorage
    loadStats();
}

/**
 * Check if decoder is ready
 */
function checkDecoderStatus() {
    fetch('/api/status')
        .then(response => response.json())
        .then(data => {
            decoderReady = data.ready;
            updateStatusAlert(
                data.ready ? 'success' : 'error',
                data.ready 
                    ? `✓ Decoder ready on ${data.device}`
                    : `✗ Error: ${data.error}`
            );
        })
        .catch(error => {
            console.error('Status check failed:', error);
            updateStatusAlert('error', `✗ Failed to connect: ${error.message}`);
        });
}

/**
 * Setup file upload listeners
 */
function setupUploadListeners() {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const selectImageBtn = document.getElementById('selectImageBtn');
    
    // Click to select
    selectImageBtn.addEventListener('click', () => imageInput.click());
    uploadArea.addEventListener('click', () => imageInput.click());
    
    // File input change
    imageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleImageUpload(e.target.files[0]);
        }
    });
    
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.add('drag-over');
    });
    
    uploadArea.addEventListener('dragleave', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        e.stopPropagation();
        uploadArea.classList.remove('drag-over');
        
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type.startsWith('image/')) {
                handleImageUpload(file);
            } else {
                showError('Please drop an image file');
            }
        }
    });
}

/**
 * Setup webcam listeners
 */
function setupWebcamListeners() {
    const startWebcamBtn = document.getElementById('startWebcamBtn');
    const stopWebcamBtn = document.getElementById('stopWebcamBtn');
    const captureWebcamBtn = document.getElementById('captureWebcamBtn');
    const webcamVideo = document.getElementById('webcamVideo');
    const webcamNotSupported = document.getElementById('webcamNotSupported');
    
    // Check if running on HTTPS or localhost (required for camera access)
    const isSecureContext = window.isSecureContext;
    const isLocalhost = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1';
    
    // Check webcam support
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        webcamNotSupported.style.display = 'block';
        webcamNotSupported.textContent = '❌ Camera API not supported';
        startWebcamBtn.disabled = true;
        return;
    }
    
    // Check HTTPS requirement (except for localhost)
    if (!isSecureContext && !isLocalhost) {
        webcamNotSupported.style.display = 'block';
        webcamNotSupported.innerHTML = '⚠️ Camera requires HTTPS. Use "https://" or localhost.';
        startWebcamBtn.disabled = true;
        return;
    }
    
    webcamNotSupported.style.display = 'none';
    
    startWebcamBtn.addEventListener('click', () => startWebcam(webcamVideo));
    stopWebcamBtn.addEventListener('click', () => stopWebcam(webcamVideo));
    captureWebcamBtn.addEventListener('click', () => captureWebcam(webcamVideo));
}

/**
 * Setup watermark listeners
 */
function setupWatermarkListeners() {
    const selectWatermarkBtn = document.getElementById('selectWatermarkImageBtn');
    const watermarkImageInput = document.getElementById('watermarkImageInput');
    const watermarkUploadArea = document.getElementById('watermarkUploadArea');
    const watermarkBtn = document.getElementById('watermarkBtn');
    const downloadWatermarkBtn = document.getElementById('downloadWatermarkBtn');
    
    // Click to select image
    if (selectWatermarkBtn) {
        selectWatermarkBtn.addEventListener('click', () => watermarkImageInput.click());
    }
    
    if (watermarkUploadArea) {
        watermarkUploadArea.addEventListener('click', () => watermarkImageInput.click());
    }
    
    // File input change
    watermarkImageInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleWatermarkImageUpload(e.target.files[0]);
        }
    });
    
    // Drag and drop
    if (watermarkUploadArea) {
        watermarkUploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.stopPropagation();
            watermarkUploadArea.classList.add('drag-over');
        });
        
        watermarkUploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            e.stopPropagation();
            watermarkUploadArea.classList.remove('drag-over');
        });
        
        watermarkUploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            e.stopPropagation();
            watermarkUploadArea.classList.remove('drag-over');
            
            if (e.dataTransfer.files.length > 0) {
                const file = e.dataTransfer.files[0];
                if (file.type.startsWith('image/')) {
                    handleWatermarkImageUpload(file);
                } else {
                    showWatermarkError('Please drop an image file');
                }
            }
        });
    }
    
    // Watermark button
    if (watermarkBtn) {
        watermarkBtn.addEventListener('click', applyWatermark);
    }
    
    // Download button
    if (downloadWatermarkBtn) {
        downloadWatermarkBtn.addEventListener('click', downloadWatermarkedImage);
    }
}

/**
 * Setup additional button listeners
 */
function setupButtonListeners() {
    const clearLogBtn = document.getElementById('clearLogBtn');
    if (clearLogBtn) {
        clearLogBtn.addEventListener('click', () => {
            document.getElementById('logContainer').innerHTML = '<em class="text-muted">No history</em>';
            stats.totalDecoded = 0;
            stats.totalFailed = 0;
            stats.confidenceScores = [];
            document.getElementById('totalDecoded').textContent = '0';
            document.getElementById('totalFailed').textContent = '0';
            document.getElementById('avgConfidence').textContent = '0%';
            saveStats();
        });
    }
}

/**
 * Start webcam stream
 */
function startWebcam(videoElement) {
    // Try with environment camera for phones, front for fallback
    const constraints = [
        { video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } },
        { video: true }
    ];
    
    async function tryCamera(constraintsList, index = 0) {
        if (index >= constraintsList.length) {
            showError('Could not access camera. Please check permissions and try again.');
            return;
        }
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia(constraintsList[index]);
            webcamStream = stream;
            isWebcamActive = true;
            videoElement.srcObject = stream;
            videoElement.style.display = 'block';
            
            // Wait for video to load
            videoElement.onloadedmetadata = () => {
                videoElement.play();
            };
            
            document.getElementById('startWebcamBtn').style.display = 'none';
            document.getElementById('stopWebcamBtn').style.display = 'inline-block';
            document.getElementById('captureWebcamBtn').style.display = 'inline-block';
            
            showSuccess('✓ Webcam started');
        } catch (error) {
            console.error(`Camera attempt ${index + 1} failed:`, error);
            // Try next constraint
            tryCamera(constraintsList, index + 1);
        }
    }
    
    tryCamera(constraints);
}

/**
 * Stop webcam stream
 */
function stopWebcam(videoElement) {
    if (webcamStream) {
        webcamStream.getTracks().forEach(track => track.stop());
        videoElement.srcObject = null;
        videoElement.style.display = 'none';
        isWebcamActive = false;
        
        document.getElementById('startWebcamBtn').style.display = 'inline-block';
        document.getElementById('stopWebcamBtn').style.display = 'none';
        document.getElementById('captureWebcamBtn').style.display = 'none';
        
        showSuccess('Webcam stopped');
    }
}

/**
 * Capture and decode webcam frame
 */
function captureWebcam(videoElement) {
    if (!isWebcamActive) {
        showError('Webcam is not active');
        return;
    }
    
    const canvas = document.createElement('canvas');
    canvas.width = videoElement.videoWidth;
    canvas.height = videoElement.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoElement, 0, 0);
    
    const imageBase64 = canvas.toDataURL('image/png');
    decodeImageBase64(imageBase64);
}

/**
 * Setup button listeners
 */
function setupButtonListeners() {
    document.getElementById('clearBtn').addEventListener('click', clearResults);
    document.getElementById('clearLogBtn').addEventListener('click', clearLog);
}

/**
 * Handle file upload
 */
function handleImageUpload(file) {
    if (!decoderReady) {
        showError('Decoder is not ready yet');
        return;
    }
    
    const formData = new FormData();
    formData.append('image', file);
    
    showLoading(true);
    updateStatusAlert('info', 'Decoding image...');
    
    fetch('/api/decode', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        displayResults(data.result, data.image);
        addLogEntry(data.result, file.name);
        updateStats(data.result);
    })
    .catch(error => {
        showLoading(false);
        console.error('Decode error:', error);
        showError(`Decoding failed: ${error.message}`);
    });
}

/**
 * Decode image from base64
 */
function decodeImageBase64(imageBase64) {
    if (!decoderReady) {
        showError('Decoder is not ready yet');
        return;
    }
    
    showLoading(true);
    updateStatusAlert('info', 'Decoding image...');
    
    fetch('/api/decode-base64', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ image: imageBase64 })
    })
    .then(response => response.json())
    .then(data => {
        showLoading(false);
        displayResults(data.result, data.image);
        addLogEntry(data.result, 'Webcam Capture');
        updateStats(data.result);
    })
    .catch(error => {
        showLoading(false);
        console.error('Decode error:', error);
        showError(`Decoding failed: ${error.message}`);
    });
}

/**
 * Display decode results
 */
function displayResults(result, imageBase64) {
    const previewImage = document.getElementById('previewImage');
    const noImagePlaceholder = document.getElementById('noImagePlaceholder');
    const resultBox = document.getElementById('resultBox');
    const confidenceBar = document.getElementById('confidenceBar');
    const confidenceText = document.getElementById('confidenceText');
    
    // Display image
    if (imageBase64) {
        previewImage.src = imageBase64;
        previewImage.style.display = 'block';
        noImagePlaceholder.style.display = 'none';
    }
    
    // Update confidence
    const confidencePercent = Math.round(result.confidence * 100);
    confidenceBar.style.width = confidencePercent + '%';
    confidenceBar.setAttribute('aria-valuenow', confidencePercent);
    confidenceText.textContent = confidencePercent + '%';
    
    // Update result
    if (result.success) {
        resultBox.textContent = result.message;
        resultBox.className = 'p-3 bg-light rounded success';
        updateStatusAlert('success', '✓ Image decoded successfully!');
    } else {
        resultBox.textContent = result.error || 'Failed to decode';
        resultBox.className = 'p-3 bg-light rounded error';
        updateStatusAlert('warning', `⚠️ ${result.error || 'Could not decode message'}`);
    }
}

/**
 * Update statistics
 */
function updateStats(result) {
    if (result.success) {
        stats.totalDecoded++;
    } else {
        stats.totalFailed++;
    }
    stats.confidenceScores.push(result.confidence);
    
    // Update UI
    document.getElementById('totalDecoded').textContent = stats.totalDecoded;
    document.getElementById('totalFailed').textContent = stats.totalFailed;
    
    if (stats.confidenceScores.length > 0) {
        const avgConfidence = Math.round(
            (stats.confidenceScores.reduce((a, b) => a + b, 0) / 
             stats.confidenceScores.length) * 100
        );
        document.getElementById('avgConfidence').textContent = avgConfidence + '%';
    }
    
    // Save stats
    saveStats();
}

/**
 * Add entry to decode history log
 */
function addLogEntry(result, source) {
    const logContainer = document.getElementById('logContainer');
    
    // Clear placeholder if needed
    if (logContainer.querySelector('em')) {
        logContainer.innerHTML = '';
    }
    
    const entry = document.createElement('div');
    entry.className = `log-entry ${result.success ? 'success' : 'error'}`;
    
    const timestamp = new Date().toLocaleTimeString();
    entry.innerHTML = `
        <span class="log-timestamp">[${timestamp}]</span>
        <span class="log-message">
            <strong>${source}:</strong> 
            ${result.success ? '✓ ' + result.message : '✗ ' + result.error}
            <small>(${Math.round(result.confidence * 100)}% confidence)</small>
        </span>
    `;
    
    logContainer.insertBefore(entry, logContainer.firstChild);
    
    // Keep only recent entries
    while (logContainer.children.length > 50) {
        logContainer.removeChild(logContainer.lastChild);
    }
}

/**
 * Clear results
 */
function clearResults() {
    document.getElementById('previewImage').style.display = 'none';
    document.getElementById('noImagePlaceholder').style.display = 'flex';
    document.getElementById('resultBox').textContent = 'Waiting for decode...';
    document.getElementById('resultBox').className = 'p-3 bg-light rounded';
    document.getElementById('confidenceBar').style.width = '0%';
    document.getElementById('confidenceText').textContent = '0%';
    document.getElementById('statusMessages').querySelectorAll('.alert').forEach(a => a.style.display = 'none');
}

/**
 * Clear log
 */
function clearLog() {
    const logContainer = document.getElementById('logContainer');
    logContainer.innerHTML = '<em class="text-muted">No decode history yet</em>';
    stats = {
        totalDecoded: 0,
        totalFailed: 0,
        confidenceScores: []
    };
    document.getElementById('totalDecoded').textContent = '0';
    document.getElementById('totalFailed').textContent = '0';
    document.getElementById('avgConfidence').textContent = '0%';
    saveStats();
}

/**
 * Update status alert
 */
function updateStatusAlert(type, message) {
    const statusAlert = document.getElementById('statusAlert');
    const iconMap = {
        'info': 'fa-info-circle',
        'success': 'fa-check-circle',
        'error': 'fa-exclamation-circle',
        'warning': 'fa-exclamation-triangle'
    };
    
    statusAlert.className = `alert alert-${type}`;
    statusAlert.innerHTML = `<i class="fas ${iconMap[type]}"></i> ${message}`;
}

/**
 * Show error message
 */
function showError(message) {
    const errorMessage = document.getElementById('errorMessage');
    document.getElementById('errorText').textContent = message;
    errorMessage.style.display = 'block';
    document.getElementById('successMessage').style.display = 'none';
    document.getElementById('warningMessage').style.display = 'none';
    setTimeout(() => errorMessage.style.display = 'none', 5000);
}

/**
 * Show success message
 */
function showSuccess(message) {
    const successMessage = document.getElementById('successMessage');
    document.getElementById('successText').textContent = message;
    successMessage.style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('warningMessage').style.display = 'none';
    setTimeout(() => successMessage.style.display = 'none', 5000);
}

/**
 * Show warning message
 */
function showWarning(message) {
    const warningMessage = document.getElementById('warningMessage');
    document.getElementById('warningText').textContent = message;
    warningMessage.style.display = 'block';
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('successMessage').style.display = 'none';
    setTimeout(() => warningMessage.style.display = 'none', 5000);
}

/**
 * Show/hide loading state
 */
function showLoading(isLoading) {
    const buttons = [
        document.getElementById('selectImageBtn'),
        document.getElementById('startWebcamBtn'),
        document.getElementById('captureWebcamBtn')
    ];
    
    buttons.forEach(btn => {
        if (isLoading) {
            btn.disabled = true;
            btn.innerHTML = '<span class="spinner"></span> Processing...';
        } else {
            btn.disabled = false;
            btn.innerHTML = btn.id === 'selectImageBtn' 
                ? '<i class="fas fa-folder-open"></i> Select Image'
                : btn.id === 'startWebcamBtn'
                ? '<i class="fas fa-camera"></i> Start Webcam'
                : '<i class="fas fa-camera-retro"></i> Capture & Decode';
        }
    });
}

/**
 * Save stats to localStorage
 */
function saveStats() {
    localStorage.setItem('decoderStats', JSON.stringify(stats));
}

/**
 * Load stats from localStorage
 */
function loadStats() {
    const saved = localStorage.getItem('decoderStats');
    if (saved) {
        stats = JSON.parse(saved);
        document.getElementById('totalDecoded').textContent = stats.totalDecoded;
        document.getElementById('totalFailed').textContent = stats.totalFailed;
        if (stats.confidenceScores.length > 0) {
            const avgConfidence = Math.round(
                (stats.confidenceScores.reduce((a, b) => a + b, 0) / 
                 stats.confidenceScores.length) * 100
            );
            document.getElementById('avgConfidence').textContent = avgConfidence + '%';
        }
    }
}

/**
 * Handle watermark image upload
 */
function handleWatermarkImageUpload(file) {
    const reader = new FileReader();
    
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            const previewImg = document.getElementById('watermarkOriginalImage');
            const placeholder = document.getElementById('watermarkOriginalPlaceholder');
            
            previewImg.src = e.target.result;
            previewImg.style.display = 'block';
            placeholder.style.display = 'none';
            
            // Store image data for later use
            window.watermarkImageData = e.target.result;
            
            showWatermarkSuccess('Image selected');
        };
        img.src = e.target.result;
    };
    
    reader.readAsDataURL(file);
}

/**
 * Apply watermark to image
 */
function applyWatermark() {
    const secretMessage = document.getElementById('secretMessage').value || '';
    const watermarkImageInput = document.getElementById('watermarkImageInput');
    
    if (!watermarkImageInput.files || watermarkImageInput.files.length === 0) {
        showWatermarkError('Please select an image');
        return;
    }
    
    const watermarkBtn = document.getElementById('watermarkBtn');
    watermarkBtn.disabled = true;
    watermarkBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing (1-2 min)...';
    
    // Create FormData
    const formData = new FormData();
    formData.append('image', watermarkImageInput.files[0]);
    if (secretMessage) {
        formData.append('secret', secretMessage);
    }
    
    // Make API call
    fetch('/api/watermark', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Display watermarked result
            const resultImg = document.getElementById('watermarkResultImage');
            const resultPlaceholder = document.getElementById('watermarkResultPlaceholder');
            const downloadBtn = document.getElementById('downloadWatermarkBtn');
            
            resultImg.src = data.watermarked_image;
            resultImg.style.display = 'block';
            resultPlaceholder.style.display = 'none';
            downloadBtn.classList.remove('d-none');
            
            // Store result for download
            window.watermarkResultData = data.watermarked_image;
            
            const watermarkType = secretMessage ? `visible + invisible ("${secretMessage}")` : 'visible only';
            showWatermarkSuccess(`✓ Watermark applied! (${watermarkType})`);
        } else {
            showWatermarkError(`Error: ${data.error}`);
        }
    })
    .catch(error => {
        console.error('Watermark error:', error);
        showWatermarkError(`Request failed: ${error.message}`);
    })
    .finally(() => {
        watermarkBtn.disabled = false;
        watermarkBtn.innerHTML = '<i class="fas fa-water"></i> Apply Watermark';
    });
}

/**
 * Download watermarked image
 */
function downloadWatermarkedImage() {
    if (!window.watermarkResultData) {
        showWatermarkError('No watermarked image to download');
        return;
    }
    
    const link = document.createElement('a');
    link.href = window.watermarkResultData;
    link.download = 'watermarked_image.png';
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    showWatermarkSuccess('Image downloaded');
}

/**
 * Show watermark error message
 */
function showWatermarkError(message) {
    const errorElement = document.getElementById('watermarkErrorMessage');
    document.getElementById('watermarkErrorText').textContent = message;
    errorElement.style.display = 'block';
    document.getElementById('watermarkSuccessMessage').style.display = 'none';
    setTimeout(() => errorElement.style.display = 'none', 5000);
}

/**
 * Show watermark success message
 */
function showWatermarkSuccess(message) {
    const successElement = document.getElementById('watermarkSuccessMessage');
    document.getElementById('watermarkSuccessText').textContent = message;
    successElement.style.display = 'block';
    document.getElementById('watermarkErrorMessage').style.display = 'none';
    setTimeout(() => successElement.style.display = 'none', 5000);
}

/**
 * Continuous decoding mode (future enhancement)
 */
function startContinuousDecoding() {
    if (!isWebcamActive) {
        showError('Please start webcam first');
        return;
    }
    
    const videoElement = document.getElementById('webcamVideo');
    setInterval(() => {
        captureWebcam(videoElement);
    }, 2000); // Capture every 2 seconds
}
