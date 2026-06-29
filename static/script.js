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

// Character counter setup
const charCounter = document.createElement('div');
charCounter.className = 'char-counter';
charCounter.innerHTML = '<span id="char-num">0</span>/500';
if (textArea && textArea.parentNode) {
    textArea.parentNode.insertBefore(charCounter, textArea.nextSibling);
    
    // Listen for inputs
    textArea.addEventListener('input', () => {
        const count = textArea.value.length;
        const numSpan = document.getElementById('char-num');
        numSpan.textContent = count;
        if (count > 500) {
            charCounter.classList.add('limit-exceeded');
        } else {
            charCounter.classList.remove('limit-exceeded');
        }
    });
}

// Automatically run initial fetch on load
window.addEventListener('DOMContentLoaded', () => {
    fetchAnalytics();
    fetchHistory();
});

// Load example text
function loadExample(element) {
    textArea.value = element.textContent;
    textArea.focus();
    // Trigger input event to update character count
    const event = new Event('input', { bubbles: true });
    textArea.dispatchEvent(event);
}

// Clear UX and Text
function clearAll() {
    textArea.value = '';
    textArea.focus();
    // Trigger input event to update character count
    const event = new Event('input', { bubbles: true });
    textArea.dispatchEvent(event);
    hideResult();
    hideError();
}

async function analyzeText() {
    const text = textArea.value.trim();

    if (!text) {
        showError("Please enter some text to analyze.");
        return;
    }

    if (text.length > 500) {
        showError("Input text exceeds the maximum limit of 500 characters.");
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
    confidenceValue.textContent = (confidence * 100).toFixed(1) + '%';

    predictionBadge.className = 'badge';
    resultSection.className = 'result-card';

    // Set classes for badge styling
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
        resultSection.classList.add('uncertain');
    }

    // Render beautiful probability progress bars
    probList.innerHTML = '';
    for (const [cls, prob] of Object.entries(probabilities)) {
        const percentage = (parseFloat(prob) * 100).toFixed(1);
        
        let barColor = 'var(--color-uncertain)';
        if (cls.includes("Hate")) barColor = 'var(--color-hate)';
        else if (cls.includes("Offensive")) barColor = 'var(--color-offensive)';
        else if (cls.includes("Neutral")) barColor = 'var(--color-neutral)';
        
        const li = document.createElement('li');
        li.className = 'prob-item';
        
        if (cls === prediction) {
            li.classList.add('active-class');
        }
        
        li.innerHTML = `
            <div class="prob-info">
                <span class="prob-label">${cls}</span>
                <span class="prob-percent">${percentage}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-bar" style="width: ${percentage}%; background: ${barColor};"></div>
            </div>
        `;
        
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
        document.getElementById('stat-avg').textContent = (data.average_confidence * 100).toFixed(1) + '%';

        renderChart(data.distribution);
    } catch (e) {
        console.error("Analytics fetch failed:", e);
    }
}

function renderChart(distribution) {
    const ctx = document.getElementById('distributionChart').getContext('2d');
    
    const labels = Object.keys(distribution);
    const counts = Object.values(distribution);
    const colors = labels.map(L => {
        if(L.includes("Hate")) return '#ff4757';
        if(L.includes("Offensive")) return '#ffa502';
        if(L.includes("Neutral")) return '#2ed573';
        return '#747d8c';
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
                    borderColor: 'var(--bg-input)'
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                cutout: '70%',
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: 'var(--text-muted)',
                            font: {
                                family: 'Plus Jakarta Sans',
                                weight: '600',
                                size: 11
                            },
                            padding: 15
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

        if (!data.history || data.history.length === 0) {
            tbody.innerHTML = `<tr><td colspan="3" style="text-align:center; color: var(--text-muted);">No records found</td></tr>`;
            return;
        }

        data.history.forEach(item => {
            const tr = document.createElement('tr');
            
            // Truncate text for grid
            let truncText = item.text.length > 25 ? item.text.substring(0, 25) + "..." : item.text;
            
            let badgeClass = 'history-badge';
            if (item.prediction.includes("Hate")) badgeClass += ' hate';
            else if (item.prediction.includes("Offensive")) badgeClass += ' offensive';
            else if (item.prediction.includes("Neutral")) badgeClass += ' neutral';
            else badgeClass += ' uncertain';

            tr.innerHTML = `
                <td title="${item.text}">${truncText}</td>
                <td><span class="${badgeClass}">${item.prediction}</span></td>
                <td style="font-weight: 700; text-align: right;">${(item.confidence * 100).toFixed(1)}%</td>
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
    } else {
        icon.className = 'fas fa-moon';
    }
    
    if (distChart) {
        distChart.data.datasets[0].borderColor = newTheme === 'dark' ? 'rgba(15, 20, 35, 0.8)' : 'rgba(255, 255, 255, 0.9)';
        distChart.options.plugins.legend.labels.color = newTheme === 'dark' ? '#9ca3af' : '#475569';
        distChart.update();
    }
}
