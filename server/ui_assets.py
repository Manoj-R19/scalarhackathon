
"""
server/ui_assets.py — Premium CSS and assets for the EmailTriage UI
"""

CSS = """
/* Premium Hackathon Theme */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');

:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --bg-dark: #0f172a;
    --card-bg: rgba(30, 41, 59, 0.7);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --border: rgba(148, 163, 184, 0.1);
    --glass-blur: 12px;
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    font-family: 'Outfit', -apple-system, sans-serif !important;
    color: var(--text-main) !important;
}

.title-header {
    text-align: center;
    padding: 2rem 0;
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
    border-radius: 16px;
}

.title-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin: 0;
    background: linear-gradient(to right, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.terminal-card {
    background: var(--card-bg);
    backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.9rem;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
}

.stat-card {
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    text-align: center;
    transition: transform 0.2s ease-in-out;
}

.stat-card:hover {
    transform: translateY(-5px);
    background: rgba(255, 255, 255, 0.05);
    border-color: var(--primary);
}

.stat-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary);
}

.stat-label {
    font-size: 0.8rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.email-row {
    background: rgba(255, 255, 255, 0.02);
    border-bottom: 1px solid var(--border);
    padding: 0.75rem 1rem;
    cursor: pointer;
    transition: background 0.2s;
}

.email-row:hover {
    background: rgba(255, 255, 255, 0.05);
}

.label-pill {
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
}

.label-spam { background: #ef4444; color: white; }
.label-low { background: #10b981; color: white; }
.label-med { background: #f59e0b; color: white; }
.label-high { background: #f97316; color: white; }
.label-escalate { background: #8b5cf6; color: white; }

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 6px;
}
::-webkit-scrollbar-track {
    background: transparent;
}
::-webkit-scrollbar-thumb {
    background: #334155;
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* Tab styling */
.tabs button.selected {
    border-bottom-color: var(--primary) !important;
    color: var(--primary) !important;
}

footer { visibility: hidden; }
"""
