// ==========================================
// 1. DOM Elements Selection
// ==========================================
// We grab elements from HTML so JS can control them.
const themeToggle = document.getElementById('theme-toggle');
const body = document.body;

const btnNews = document.getElementById('btn-news');
const btnUrl = document.getElementById('btn-url');
const newsSection = document.getElementById('news-section');
const urlSection = document.getElementById('url-section');

const submitNewsBtn = document.getElementById('submit-news');
const newsInput = document.getElementById('news-input');

const submitUrlBtn = document.getElementById('submit-url');
const urlInput = document.getElementById('url-input');

const loadingSpinner = document.getElementById('loading-spinner');
const resultSection = document.getElementById('result-section');
const resultTitle = document.getElementById('result-title');
const resultExplanation = document.getElementById('result-explanation');

let confidenceChartInstance = null; // Stores our Chart.js object

// ==========================================
// 2. Dark Mode Toggle Logic
// ==========================================
themeToggle.addEventListener('click', () => {
    // Toggles the 'dark-mode' class on the body element
    body.classList.toggle('dark-mode');
    
    // Check active mode status
    const isDark = body.classList.contains('dark-mode');
    
    // Change the icon from moon to sun based on the mode
    const icon = themeToggle.querySelector('i');
    if (isDark) {
        icon.classList.remove('fa-moon');
        icon.classList.add('fa-sun');
    } else {
        icon.classList.remove('fa-sun');
        icon.classList.add('fa-moon');
    }

    // Update existing chart legend text color dynamically
    if (confidenceChartInstance) {
        const chartTextColor = isDark ? '#adb5bd' : '#666666';
        confidenceChartInstance.options.plugins.legend.labels.color = chartTextColor;
        confidenceChartInstance.update();
    }
});

// ==========================================
// 3. Tab Switching Logic
// ==========================================
// When 'Fake News' button is clicked
btnNews.addEventListener('click', () => {
    btnNews.classList.replace('btn-outline-primary', 'btn-primary');
    btnUrl.classList.replace('btn-primary', 'btn-outline-primary');
    newsSection.classList.remove('d-none'); // Show News
    urlSection.classList.add('d-none');     // Hide URL
    hideResults();                          // Clear previous results
});

// When 'Phishing URL' button is clicked
btnUrl.addEventListener('click', () => {
    btnUrl.classList.replace('btn-outline-primary', 'btn-primary');
    btnNews.classList.replace('btn-primary', 'btn-outline-primary');
    urlSection.classList.remove('d-none');  // Show URL
    newsSection.classList.add('d-none');    // Hide News
    hideResults();
});

function hideResults() {
    resultSection.classList.add('d-none');
    loadingSpinner.classList.add('d-none');
}

// ==========================================
// 4. Form Submission & Input Validation
// ==========================================
submitNewsBtn.addEventListener('click', async () => {
    const text = newsInput.value.trim(); // .trim() removes extra spaces
    if (text.length < 10) {
        alert("Please enter a valid news article (at least 10 characters).");
        return; // Stop function execution
    }
    await processPrediction('news', { text: text });
});

submitUrlBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url.startsWith('http')) {
        alert("Please enter a valid URL starting with http:// or https://");
        return;
    }
    await processPrediction('url', { url: url });
});

// ==========================================
// 5. API Communication (Fetch, Async/Await)
// ==========================================
/**
 * Sends data to our Python backend and waits for the AI prediction.
 * @param {string} type - 'news' or 'url'
 * @param {object} payload - The JSON data to send
 */
async function processPrediction(type, payload) {
    // 1. Hide UI and show loading spinner
    newsSection.classList.add('d-none');
    urlSection.classList.add('d-none');
    resultSection.classList.add('d-none');
    loadingSpinner.classList.remove('d-none');

    // Define the endpoint based on the type
    const baseURL = 'https://ai-fraud-backend-qxs0.onrender.com';
    const endpoint = type === 'news' ? `${baseURL}/api/predict/news` : `${baseURL}/api/predict/url`;

    try {
        // 2. Fetch API: Talk to the Python server
        const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (!response.ok) throw new Error("Server Error");

        // 3. Parse JSON response
        const data = await response.json(); 
        
        // 4. Update UI with results
        showResults(data);

    } catch (error) {
        console.error("Error connecting to server:", error);
        alert("Could not connect to the AI Server. Make sure the Flask backend is running.");
        // Reset UI if error occurs
        loadingSpinner.classList.add('d-none');
        if(type === 'news') newsSection.classList.remove('d-none');
        else urlSection.classList.remove('d-none');
    }
}

// ==========================================
// 6. Rendering Results & Chart.js
// ==========================================
function showResults(data) {
    // Hide spinner, show results card
    loadingSpinner.classList.add('d-none');
    resultSection.classList.remove('d-none');

    // Update Text and Colors
    if (data.prediction === 'Fake' || data.prediction === 'Phishing') {
        resultTitle.innerHTML = `<i class="fas fa-exclamation-triangle"></i> ${data.prediction} Detected!`;
        resultTitle.className = "fw-bold mb-2 text-danger"; // Make text red
        resultExplanation.innerText = `The AI is ${data.confidence}% confident this is malicious.`;
    } else {
        resultTitle.innerHTML = `<i class="fas fa-check-circle"></i> ${data.prediction}`;
        resultTitle.className = "fw-bold mb-2 text-success"; // Make text green
        resultExplanation.innerText = `The AI is ${data.confidence}% confident this is safe.`;
    }

    // Draw Probability Chart
    drawChart(data.confidence, 100 - data.confidence);
}

function drawChart(confidenceSafe, confidenceDanger) {
    const ctx = document.getElementById('confidenceChart').getContext('2d');

    // Destroy old chart if it exists so they don't overlap
    if (confidenceChartInstance) {
        confidenceChartInstance.destroy();
    }

    // Establish current text color based on active theme
    const isDark = document.body.classList.contains('dark-mode');
    const chartTextColor = isDark ? '#adb5bd' : '#666666';

    // Create new Chart.js Donut Chart
    confidenceChartInstance = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['Prediction Confidence', 'Margin of Error'],
            datasets: [{
                data: [confidenceSafe, confidenceDanger],
                backgroundColor: ['#0d6efd', '#e9ecef'],
                borderWidth: 0
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            cutout: '70%', // Makes the donut hole larger
            plugins: {
                legend: { 
                    position: 'bottom',
                    labels: {
                        color: chartTextColor // Apply dynamic text color to legend
                    }
                }
            }
        }
    });
}