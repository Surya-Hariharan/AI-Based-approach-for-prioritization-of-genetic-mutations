// App State
let selectedFile = null;
let predictionCount = 0;

// Initialize app
document.addEventListener('DOMContentLoaded', () => {
    initApp();
    setupEventListeners();
});

// Initialize application
async function initApp() {
    await checkHealth();
    await loadStats();
}

// Check API health
async function checkHealth() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        updateStatus(data.status === 'healthy', data.models_loaded);
        document.getElementById('modelCount').textContent = data.models_loaded.length;
        
        // Update model select dropdown
        const modelSelect = document.getElementById('modelSelect');
        modelSelect.innerHTML = '';
        data.models_loaded.forEach(model => {
            const option = document.createElement('option');
            option.value = model;
            option.textContent = formatModelName(model);
            modelSelect.appendChild(option);
        });
        
    } catch (error) {
        console.error('Health check failed:', error);
        updateStatus(false, []);
    }
}

// Update status indicator
function updateStatus(isHealthy, models) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');
    
    if (isHealthy) {
        statusDot.style.background = 'var(--success)';
        statusText.textContent = `Online • ${models.length} model${models.length !== 1 ? 's' : ''} loaded`;
    } else {
        statusDot.style.background = 'var(--danger)';
        statusText.textContent = 'Offline';
    }
}

// Load statistics
async function loadStats() {
    try {
        const response = await fetch('/api/stats');
        const data = await response.json();
        
        const statsContent = document.getElementById('statsContent');
        
        if (data.error) {
            statsContent.innerHTML = `<p class="loading-text">Unable to load statistics</p>`;
            return;
        }
        
        let statsHTML = '<div class="stats-grid">';
        
        statsHTML += `
            <div class="stat-item">
                <h4>Models Available</h4>
                <p>${data.models_available}</p>
            </div>
        `;
        
        if (data.total_genes_ranked) {
            statsHTML += `
                <div class="stat-item">
                    <h4>Genes Ranked</h4>
                    <p>${data.total_genes_ranked.toLocaleString()}</p>
                </div>
            `;
        }
        
        statsHTML += '</div>';
        
        if (data.top_genes && data.top_genes.length > 0) {
            statsHTML += '<h3 style="margin-top: 1.5rem; margin-bottom: 1rem;">Top Ranked Genes</h3>';
            statsHTML += '<div style="overflow-x: auto;">';
            statsHTML += '<table style="width: 100%; border-collapse: collapse;">';
            statsHTML += '<thead><tr style="border-bottom: 1px solid var(--border);"><th style="padding: 0.75rem; text-align: left;">Gene</th><th style="padding: 0.75rem; text-align: right;">Score</th></tr></thead>';
            statsHTML += '<tbody>';
            data.top_genes.forEach((gene, index) => {
                statsHTML += `
                    <tr style="border-bottom: 1px solid var(--border);">
                        <td style="padding: 0.75rem;">${gene.gene}</td>
                        <td style="padding: 0.75rem; text-align: right; font-weight: 500;">${gene.mean_score.toFixed(4)}</td>
                    </tr>
                `;
            });
            statsHTML += '</tbody></table></div>';
        }
        
        statsContent.innerHTML = statsHTML;
        
    } catch (error) {
        console.error('Failed to load stats:', error);
        document.getElementById('statsContent').innerHTML = 
            '<p class="loading-text">Unable to load statistics</p>';
    }
}

// Setup event listeners
function setupEventListeners() {
    // Tab switching
    document.querySelectorAll('.tab-button').forEach(button => {
        button.addEventListener('click', () => switchTab(button.dataset.tab));
    });
    
    // File upload
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });
    
    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
    
    document.getElementById('clearFile').addEventListener('click', clearFile);
    document.getElementById('uploadBtn').addEventListener('click', uploadAndPredict);
    document.getElementById('predictBtn').addEventListener('click', predictManual);
}

// Switch between tabs
function switchTab(tabName) {
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    
    document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');
    document.getElementById(tabName).classList.add('active');
}

// Handle file selection
function handleFileSelect(file) {
    if (!file.name.match(/\.(csv|vcf)$/)) {
        alert('Please select a CSV or VCF file');
        return;
    }
    
    selectedFile = file;
    document.getElementById('fileName').textContent = file.name;
    document.getElementById('fileInfo').style.display = 'flex';
    document.getElementById('uploadBtn').disabled = false;
}

// Clear selected file
function clearFile() {
    selectedFile = null;
    document.getElementById('fileInput').value = '';
    document.getElementById('fileInfo').style.display = 'none';
    document.getElementById('uploadBtn').disabled = true;
}

// Upload and predict
async function uploadAndPredict() {
    if (!selectedFile) return;
    
    const modelName = document.getElementById('modelSelect').value;
    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('model', modelName);
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults(data.predictions, modelName);
            predictionCount += data.count;
            document.getElementById('predictionCount').textContent = predictionCount;
            clearFile();
        } else {
            alert(`Error: ${data.error}`);
        }
        
    } catch (error) {
        console.error('Prediction failed:', error);
        alert('Prediction failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Manual prediction
async function predictManual() {
    const featureInput = document.getElementById('featureInput').value.trim();
    
    if (!featureInput) {
        alert('Please enter feature values');
        return;
    }
    
    // Parse comma-separated values
    const features = featureInput.split(',').map(v => parseFloat(v.trim())).filter(v => !isNaN(v));
    
    if (features.length === 0) {
        alert('Please enter valid numeric values');
        return;
    }
    
    const modelName = document.getElementById('modelSelect').value;
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                features: features,
                model: modelName
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayResults([data.prediction], modelName);
            predictionCount++;
            document.getElementById('predictionCount').textContent = predictionCount;
        } else {
            alert(`Error: ${data.error}`);
        }
        
    } catch (error) {
        console.error('Prediction failed:', error);
        alert('Prediction failed. Please try again.');
    } finally {
        showLoading(false);
    }
}

// Display prediction results
function displayResults(predictions, modelName) {
    const resultsCard = document.getElementById('resultsCard');
    const resultsContent = document.getElementById('resultsContent');
    
    let html = `<div style="margin-bottom: 1rem; padding: 0.75rem; background: var(--bg-primary); border-radius: 0.5rem;">
        <strong>Model:</strong> ${formatModelName(modelName)}
    </div>`;
    
    predictions.forEach((result, index) => {
        const isPredictionArray = Array.isArray(predictions);
        const showIndex = isPredictionArray && predictions.length > 1;
        
        html += `
            <div class="result-item">
                ${showIndex ? `<h3 style="margin-bottom: 1rem; color: var(--text-secondary);">Mutation #${index + 1}</h3>` : ''}
                <div class="result-header">
                    <span class="prediction-badge ${result.prediction.toLowerCase()}">
                        ${result.prediction}
                    </span>
                    <span class="confidence-badge ${result.confidence.toLowerCase()}">
                        ${result.confidence} Confidence
                    </span>
                </div>
                <div class="probability-bars">
                    <div class="probability-bar">
                        <div class="probability-label">
                            <span>Pathogenic</span>
                            <span><strong>${(result.pathogenic_score * 100).toFixed(1)}%</strong></span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill pathogenic" style="width: ${result.pathogenic_score * 100}%"></div>
                        </div>
                    </div>
                    <div class="probability-bar">
                        <div class="probability-label">
                            <span>Benign</span>
                            <span><strong>${(result.benign_score * 100).toFixed(1)}%</strong></span>
                        </div>
                        <div class="progress-bar">
                            <div class="progress-fill benign" style="width: ${result.benign_score * 100}%"></div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    });
    
    resultsContent.innerHTML = html;
    resultsCard.style.display = 'block';
    
    // Scroll to results
    resultsCard.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Show/hide loading overlay
function showLoading(show) {
    document.getElementById('loadingOverlay').style.display = show ? 'flex' : 'none';
}

// Format model name for display
function formatModelName(name) {
    const names = {
        'mlp': 'MLP (Neural Network)',
        'baseline': 'Baseline (Logistic Regression)',
        'ensemble': 'Ensemble (Stacking)'
    };
    return names[name] || name.toUpperCase();
}

// Show documentation
function showDocumentation() {
    alert(`Genetic Mutation Prioritization System

This application uses machine learning to predict the pathogenicity of genetic mutations.

How to use:
1. Select a model (MLP, Baseline, or Ensemble)
2. Upload a CSV file with mutation features, OR enter feature values manually
3. Click "Predict" to get results

The system will return:
- Prediction (Pathogenic or Benign)
- Probability scores
- Confidence level

For more information, visit the GitHub repository.`);
}
