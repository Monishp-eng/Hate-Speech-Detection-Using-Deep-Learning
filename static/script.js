const textArea = document.getElementById('text-input');
const analyzeBtn = document.getElementById('analyze-btn');
const clearBtn = document.getElementById('clear-btn');
const spinner = document.getElementById('loading-spinner');
const errorBox = document.getElementById('error-box');
const resultSection = document.getElementById('result-section');
const predictionBadge = document.getElementById('prediction-badge');
const confidenceValue = document.getElementById('confidence-value');
const probList = document.getElementById('prob-list');

let distChart = null;

// Automatically run initial fetch on load
window.addEventListener('DOMContentLoaded', () => {
    fetchAnalytics();
    fetchHistory();
});

// Load example text
function loadExample(element) {
    textArea.value = element.textContent;
    textArea.focus();
}

// Clear UX and Text
function clearAll() {
    textArea.value = '';
    textArea.focus();
    hideResult();
    hideError();
}

async function analyzeText() {
    const text = textArea.value.trim();

    if (!text) {
        showError("Please enter some text to analyze.");
        return;
    }

    hideError();
    hideResult();
    setLoading(true);

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: text })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || `Server responded with status ${response.status}`);
        }

        displayResult(data);
        
        // Refresh analytics and history
        fetchAnalytics();
        fetchHistory();

    } catch (err) {
        showError("Failed to analyze text: " + err.message);
    } finally {
        setLoading(false);
    }
}

function displayResult(data) {
    const prediction = data.prediction;
    const confidence = parseFloat(data.confidence);
    const probabilities = data.probabilities;

    predictionBadge.textContent = prediction.toUpperCase();
    confidenceValue.textContent = (confidence * 100).toFixed(2) + '%';

    predictionBadge.className = 'badge';
    resultSection.className = 'result-card';

    if (prediction.includes("Hate Speech")) {
        predictionBadge.classList.add('hate');
        resultSection.classList.add('hate');
    } else if (prediction.includes("Offensive Language")) {
        predictionBadge.classList.add('offensive');
        resultSection.classList.add('offensive');
    } else if (prediction.includes("Neutral")) {
        predictionBadge.classList.add('neutral');
        resultSection.classList.add('neutral');
    } else {
        predictionBadge.classList.add('uncertain');
    }

    probList.innerHTML = '';
    for (const [cls, prob] of Object.entries(probabilities)) {
        const li = document.createElement('li');
        const spanClass = document.createElement('span');
        spanClass.textContent = cls;
        const spanProb = document.createElement('span');
        spanProb.textContent = (parseFloat(prob) * 100).toFixed(2) + '%';
        
        li.appendChild(spanClass);
        li.appendChild(spanProb);
        
        if (cls === prediction) {
            li.style.fontWeight = 'bold';
            li.style.color = 'var(--text-main)';
        }
        
        probList.appendChild(li);
    }

    resultSection.classList.remove('hidden');
}

// ----------------------------------------------------
// Analytics & History Board
// ----------------------------------------------------

async function fetchAnalytics() {
    try {
        const response = await fetch('/analytics');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        document.getElementById('stat-total').textContent = data.total_predictions || 0;
        document.getElementById('stat-avg').textContent = (data.average_confidence * 100).toFixed(2) + '%';

        renderChart(data.distribution);
    } catch (e) {
        console.error("Analytics fetch failed:", e);
    }
}

function renderChart(distribution) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    // Map colors to matches
    const labels = Object.keys(distribution);
    const counts = Object.values(distribution);
    const colors = labels.map(L => {
        if(L.includes("Hate")) return '#f44336';
        if(L.includes("Offensive")) return '#ff9800';
        if(L.includes("Neutral")) return '#4CAF50';
        return '#9e9e9e';
    });

    if (distChart) {
        distChart.data.labels = labels;
        distChart.data.datasets[0].data = counts;
        distChart.data.datasets[0].backgroundColor = colors;
        distChart.update();
    } else {
        distChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: counts,
                    backgroundColor: colors,
                    borderWidth: 2,
                    borderColor: '#2b2b2b'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        labels: {
                            color: '#b0b0b0'
                        }
                    }
                }
            }
        });
    }
}

async function fetchHistory() {
    try {
        const response = await fetch('/history');
        const data = await response.json();
        if (data.error) throw new Error(data.error);

        const tbody = document.getElementById('history-body');
        tbody.innerHTML = '';

        data.history.forEach(item => {
            const tr = document.createElement('tr');
            
            // Truncate text for grid
            let truncText = item.text.length > 25 ? item.text.substring(0, 25) + "..." : item.text;
            
            tr.innerHTML = `
                <td title="${item.text}">${truncText}</td>
                <td><span style="font-weight:600">${item.prediction}</span></td>
                <td>${(item.confidence * 100).toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });

    } catch (e) {
        console.error("History fetch failed:", e);
    }
}

async function clearHistory() {
    if (!confirm("Are you sure you want to delete all prediction history?")) return;
    
    try {
        const response = await fetch('/history', {
            method: 'DELETE'
        });
        
        const data = await response.json();
        if (!response.ok) throw new Error(data.error);
        
        // Refresh views
        fetchAnalytics();
        fetchHistory();
    } catch (e) {
        console.error("Failed to clear history:", e);
        showError("Failed to clear history: " + e.message);
    }
}

// ----------------------------------------------------
// UI Logic & Helpers
// ----------------------------------------------------

function setLoading(isLoading) {
    if (isLoading) {
        analyzeBtn.disabled = true;
        analyzeBtn.style.opacity = '0.7';
        spinner.classList.remove('hidden');
    } else {
        analyzeBtn.disabled = false;
        analyzeBtn.style.opacity = '1';
        spinner.classList.add('hidden');
    }
}

function showError(message) {
    errorBox.textContent = message;
    errorBox.classList.remove('hidden');
}

function hideError() {
    errorBox.classList.add('hidden');
}

function hideResult() {
    resultSection.classList.add('hidden');
}

textArea.addEventListener('keydown', function(e) {
    if (e.ctrlKey && e.key === 'Enter') {
        analyzeText();
    }
});

// ----------------------------------------------------
// Theme Toggle
// ----------------------------------------------------
function toggleTheme() {
    const htmlElem = document.documentElement;
    const currentTheme = htmlElem.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    htmlElem.setAttribute('data-theme', newTheme);
    
    const icon = document.getElementById('theme-icon');
    if (newTheme === 'dark') {
        icon.className = 'fas fa-sun';
        if (distChart) distChart.options.datasets.borderColor = '#2b2b2b';
    } else {
        icon.className = 'fas fa-moon';
        if (distChart) distChart.options.datasets.borderColor = '#ffffff';
    }
    
    if (distChart) {
        distChart.options.plugins.legend.labels.color = newTheme === 'dark' ? '#b0b0b0' : '#666666';
        distChart.update();
    }
}
