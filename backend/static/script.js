let ws;
const videoImg = document.getElementById('video-feed');
const statusBadge = document.getElementById('connection-status');
const statsList = document.getElementById('stats-list');
const startBtn = document.getElementById('start-btn');
const stopBtn = document.getElementById('stop-btn');

function startStream() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        statusBadge.textContent = 'Connected';
        statusBadge.classList.remove('disconnected');
        statusBadge.classList.add('connected');
        startBtn.disabled = true;
        stopBtn.disabled = false;
    };

    ws.onmessage = (event) => {
        // Check if message is binary (image) or text (stats)
        if (typeof event.data === 'object') {
            // Blob/ArrayBuffer -> Image
            const url = URL.createObjectURL(event.data);
            videoImg.src = url;

            // Clean up old object URL to prevent memory leaks
            videoImg.onload = () => {
                URL.revokeObjectURL(url);
            };
        } else {
            // Text -> Stats
            try {
                const stats = JSON.parse(event.data);
                updateStats(stats);
            } catch (e) {
                console.error("Error parsing stats:", e);
            }
        }
    };

    ws.onclose = () => {
        statusBadge.textContent = 'Disconnected';
        statusBadge.classList.remove('connected');
        statusBadge.classList.add('disconnected');
        startBtn.disabled = false;
        stopBtn.disabled = true;
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };
}

function stopStream() {
    if (ws) {
        ws.close();
    }
}

function updateStats(stats) {
    statsList.innerHTML = '';

    if (stats.count === 0) {
        statsList.innerHTML = '<li>No trash detected</li>';
        return;
    }

    // Count occurrences
    const counts = {};
    stats.objects.forEach(obj => {
        counts[obj] = (counts[obj] || 0) + 1;
    });

    for (const [obj, count] of Object.entries(counts)) {
        const li = document.createElement('li');
        li.textContent = `${obj}: ${count}`;
        statsList.appendChild(li);
    }
}
