/**
 * SpectraForensic - AI Image Detector Frontend Controller
 */

document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const loader = document.getElementById('loader');
    const loaderStep = document.getElementById('loaderStep');
    const resultsSection = document.getElementById('resultsSection');
    const sampleButtons = document.querySelectorAll('.sample-btn');

    // Result elements
    const gaugePath = document.getElementById('gaugePath');
    const aiProbValue = document.getElementById('aiProbValue');
    const verdictBadge = document.getElementById('verdictBadge');
    const confidenceTag = document.getElementById('confidenceTag');
    const verdictTitle = document.getElementById('verdictTitle');
    const verdictDesc = document.getElementById('verdictDesc');
    const barAiVal = document.getElementById('barAiVal');
    const barRealVal = document.getElementById('barRealVal');
    const barAiProgress = document.getElementById('barAiProgress');
    const barRealProgress = document.getElementById('barRealProgress');
    const metaFilename = document.getElementById('metaFilename');
    const metaDimensions = document.getElementById('metaDimensions');
    const metaSize = document.getElementById('metaSize');
    const metaTime = document.getElementById('metaTime');
    const indicatorsGrid = document.getElementById('indicatorsGrid');
    const btnDownloadJson = document.getElementById('btnDownloadJson');

    // Visualizer tabs & display
    const tabButtons = document.querySelectorAll('.tab-btn');
    const activeVisualImage = document.getElementById('activeVisualImage');
    const visualizerExplanation = document.getElementById('visualizerExplanation');

    let currentAnalysisData = null;

    const layerExplanations = {
        'preview': '<strong>Original Image View:</strong> The input image rendered in full standard RGB color space.',
        'fft': '<strong>2D FFT Power Spectrum:</strong> Visualizes spatial frequency energy distribution. Natural photos show smooth isotropic decay (1/f² power law). Synthetic diffusion and GAN models exhibit abnormal high-frequency spikes, radial distortion, and grid checkerboard artifacts.',
        'ela': '<strong>Error Level Analysis (ELA):</strong> Evaluates multi-pass compression differences. Organic camera photos have uniform compression error rates across surfaces of similar texture. AI images often show abnormal disparities between ultra-smooth and hyper-detailed areas.',
        'noise': '<strong>High-Pass Noise Residual:</strong> Visualizes high-frequency noise after removing low-frequency scene structure. Physical cameras exhibit natural Poisson/Gaussian sensor grain (PRNU); generative AI often shows unnatural over-smoothing or non-physical noise kurtosis.',
        'chroma': '<strong>Color Saturation & Chromatic Map:</strong> Displays HSV saturation and chromatic dispersion across channels. AI models frequently exhibit unphysical cross-channel correlations or unnaturally saturated color clustering without optical lens dispersion.'
    };

    // ----------------------------------------------------
    // Drag & Drop Handling
    // ----------------------------------------------------
    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('dragover');
        });
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('dragover');
        });
    });

    dropZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileUpload(e.target.files[0]);
        }
    });

    // ----------------------------------------------------
    // Quick Sample Buttons
    // ----------------------------------------------------
    sampleButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const sampleName = btn.getAttribute('data-sample');
            handleSampleTest(sampleName);
        });
    });

    // ----------------------------------------------------
    // API Upload & Analysis
    // ----------------------------------------------------
    function handleFileUpload(file) {
        if (!file.type.match('image.*')) {
            alert('Please upload a valid image file (PNG, JPG, WEBP, BMP).');
            return;
        }

        const formData = new FormData();
        formData.append('image', file);

        startLoading();

        fetch('/api/detect', {
            method: 'POST',
            body: formData
        })
        .then(res => {
            if (!res.ok) throw new Error('Analysis failed on server.');
            return res.json();
        })
        .then(data => {
            if (data.status === 'error') {
                alert(data.message);
                resetToUpload();
                return;
            }
            displayResults(data);
        })
        .catch(err => {
            console.error(err);
            alert('Failed to analyze image: ' + err.message);
            resetToUpload();
        });
    }

    function handleSampleTest(sampleName) {
        startLoading();

        fetch(`/api/sample/${sampleName}`)
        .then(res => {
            if (!res.ok) throw new Error('Sample not found.');
            return res.json();
        })
        .then(data => {
            if (data.status === 'error') {
                alert(data.message);
                resetToUpload();
                return;
            }
            displayResults(data);
        })
        .catch(err => {
            console.error(err);
            alert('Error loading sample: ' + err.message);
            resetToUpload();
        });
    }

    // ----------------------------------------------------
    // Loading State Animation
    // ----------------------------------------------------
    let stepInterval = null;
    function startLoading() {
        resultsSection.style.display = 'none';
        loader.style.display = 'block';

        const steps = [
            'Decomposing spatial frequencies via 2D Fast Fourier Transform (FFT)...',
            'Computing Error Level Analysis (ELA) at 90% quality compression...',
            'Extracting high-pass sensor PRNU noise residuals...',
            'Analyzing inter-channel chromatic correlation and saturation...',
            'Synthesizing calibrated Bayesian probability scores...'
        ];
        let idx = 0;
        loaderStep.textContent = steps[0];

        clearInterval(stepInterval);
        stepInterval = setInterval(() => {
            idx = (idx + 1) % steps.length;
            loaderStep.textContent = steps[idx];
        }, 600);

        // Scroll to loader
        loader.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function stopLoading() {
        clearInterval(stepInterval);
        loader.style.display = 'none';
    }

    // ----------------------------------------------------
    // Display Results
    // ----------------------------------------------------
    function displayResults(data) {
        stopLoading();
        currentAnalysisData = data;
        resultsSection.style.display = 'block';

        const aiProb = data.ai_probability;
        const realProb = data.real_probability;

        // Animate circular gauge
        aiProbValue.textContent = `${aiProb}%`;
        gaugePath.setAttribute('stroke-dasharray', `${aiProb}, 100`);

        // Gauge & Badge Colors
        let strokeColor = '#EF4444'; // Red default
        let badgeClass = 'badge-ai';

        if (aiProb >= 65) {
            strokeColor = '#EF4444';
            badgeClass = 'badge-ai';
        } else if (aiProb >= 45) {
            strokeColor = '#F59E0B';
            badgeClass = 'badge-uncertain';
        } else {
            strokeColor = '#10B981';
            badgeClass = 'badge-real';
        }

        gaugePath.style.stroke = strokeColor;
        aiProbValue.style.color = strokeColor;

        // Verdict Badge & Titles
        verdictBadge.className = `badge ${badgeClass}`;
        verdictBadge.textContent = data.verdict_class === 'ai' ? 'AI-Generated' : (
            data.verdict_class === 'real' ? 'Authentic Human Image' : 'Mixed / Inconclusive'
        );
        confidenceTag.textContent = `Confidence: ${data.confidence}`;
        verdictTitle.textContent = data.verdict;
        verdictDesc.textContent = data.explanation;

        // Dual comparison progress bar
        barAiVal.textContent = `${aiProb}%`;
        barRealVal.textContent = `${realProb}%`;
        barAiProgress.style.width = `${aiProb}%`;
        barRealProgress.style.width = `${realProb}%`;

        // Metadata row
        metaFilename.textContent = data.filename || 'uploaded_image';
        metaDimensions.textContent = `${data.image_info.width} × ${data.image_info.height} px`;
        metaSize.textContent = `${data.file_size_kb || 0} KB`;
        metaTime.textContent = `${data.elapsed_seconds || 0.4}s scan`;

        // Populate Forensic Indicator Cards
        indicatorsGrid.innerHTML = '';
        const indicators = data.indicators;
        for (const [key, item] of Object.entries(indicators)) {
            const card = document.createElement('div');
            card.className = 'indicator-card';

            const score = item.score;
            let barColor = score > 60 ? '#EF4444' : (score > 40 ? '#F59E0B' : '#10B981');

            card.innerHTML = `
                <div class="indicator-header">
                    <span class="indicator-name">${item.name}</span>
                    <span class="indicator-score" style="color: ${barColor}">${score}% AI</span>
                </div>
                <div class="score-bar">
                    <div class="score-fill" style="width: ${score}%; background: ${barColor};"></div>
                </div>
                <div class="indicator-status">${item.status}</div>
                <div class="indicator-details">${typeof item.details === 'object' ? JSON.stringify(item.details) : item.details}</div>
            `;
            indicatorsGrid.appendChild(card);
        }

        // Set Default Visualizer Tab to Preview
        switchVisualizerTab('preview');

        // Scroll to results
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    // ----------------------------------------------------
    // Visualizer Tab Switching
    // ----------------------------------------------------
    function switchVisualizerTab(tabName) {
        tabButtons.forEach(btn => {
            if (btn.getAttribute('data-tab') === tabName) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });

        if (currentAnalysisData && currentAnalysisData.heatmaps && currentAnalysisData.heatmaps[tabName]) {
            activeVisualImage.src = currentAnalysisData.heatmaps[tabName];
            visualizerExplanation.innerHTML = layerExplanations[tabName] || '';
        }
    }

    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            const tabName = btn.getAttribute('data-tab');
            switchVisualizerTab(tabName);
        });
    });

    // ----------------------------------------------------
    // JSON Report Download
    // ----------------------------------------------------
    btnDownloadJson.addEventListener('click', () => {
        if (!currentAnalysisData) return;

        // Omit large base64 images from JSON download for compact file
        const exportData = { ...currentAnalysisData };
        exportData.heatmaps = '[Base64 image data excluded for export]';

        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(exportData, null, 2));
        const downloadAnchor = document.createElement('a');
        downloadAnchor.setAttribute("href", dataStr);
        downloadAnchor.setAttribute("download", `forensic_report_${exportData.filename || 'analysis'}.json`);
        document.body.appendChild(downloadAnchor);
        downloadAnchor.click();
        downloadAnchor.remove();
    });

    // Global reset
    window.resetToUpload = function() {
        resultsSection.style.display = 'none';
        loader.style.display = 'none';
        fileInput.value = '';
        currentAnalysisData = null;
        dropZone.scrollIntoView({ behavior: 'smooth', block: 'center' });
    };
});
