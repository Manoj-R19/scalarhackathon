"""
server/ui_assets.py — Premium CSS and assets for the EmailTriage UI
"""

CSS = """
/* Ultra-Premium Glassmorphism & Animated Hackathon Theme */
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@400;500&display=swap');

:root {
    --primary: #6366f1;
    --primary-hover: #4f46e5;
    --primary-glow: rgba(99, 102, 241, 0.4);
    --secondary: #ec4899;
    --bg-dark: #0b0f19;
    --bg-surface: rgba(17, 24, 39, 0.65);
    --card-bg: rgba(30, 41, 59, 0.4);
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --border: rgba(148, 163, 184, 0.15);
    --glass-blur: 16px;
    --accent-gradient: linear-gradient(135deg, #818cf8 0%, #c084fc 100%);
}

body, .gradio-container {
    background-color: var(--bg-dark) !important;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(99, 102, 241, 0.12), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(236, 72, 153, 0.1), transparent 25%);
    background-attachment: fixed;
    font-family: 'Outfit', 'Inter', -apple-system, sans-serif !important;
    color: var(--text-main) !important;
}

/* Glassmorphism Containers */
.container {
    background: var(--bg-surface);
    backdrop-filter: blur(var(--glass-blur));
    -webkit-backdrop-filter: blur(var(--glass-blur));
    border: 1px solid var(--border);
    border-radius: 20px;
    box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
    padding: 2rem;
}

/* Header with glowing text */
.title-header {
    text-align: center;
    padding: 2.5rem 0;
    background: linear-gradient(135deg, rgba(30,41,59,0.8) 0%, rgba(15,23,42,0.6) 100%);
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    border-radius: 16px;
    position: relative;
    overflow: hidden;
}

.title-header::before {
    content: "";
    position: absolute;
    top: 0; left: -100%;
    width: 50%; height: 100%;
    background: linear-gradient(to right, transparent, rgba(255,255,255,0.05), transparent);
    animation: shine 4s infinite;
}

@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}

.title-header h1 {
    font-size: 2.8rem;
    font-weight: 700;
    margin: 0;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0px 4px 20px rgba(139, 92, 246, 0.3);
    letter-spacing: -0.02em;
}

/* Terminal & Logs */
.terminal-card {
    background: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(8px);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.5rem;
    font-family: 'Fira Code', monospace;
    font-size: 0.9rem;
    box-shadow: inset 0 2px 4px 0 rgba(0, 0, 0, 0.3);
    color: #a7f3d0;
}

/* Micro-Animated Stat Cards */
.stat-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.01) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 1.8rem;
    text-align: center;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}

.stat-card:hover {
    transform: translateY(-5px) scale(1.02);
    background: linear-gradient(145deg, rgba(255,255,255,0.08) 0%, rgba(255,255,255,0.02) 100%);
    border-color: rgba(99, 102, 241, 0.4);
    box-shadow: 0 10px 25px -5px var(--primary-glow);
}

.stat-value {
    font-size: 2.5rem;
    font-weight: 700;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
}

.stat-label {
    font-size: 0.85rem;
    font-family: 'Inter', sans-serif;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.5rem;
}

/* Dynamic Email Inbox Rows */
.email-row {
    background: rgba(30, 41, 59, 0.3);
    border-bottom: 1px solid var(--border);
    padding: 1rem 1.25rem;
    cursor: pointer;
    transition: all 0.2s ease;
    border-radius: 8px;
    margin-bottom: 6px;
}

.email-row:hover {
    background: rgba(51, 65, 85, 0.5);
    transform: translateX(4px);
}

.label-pill {
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255,255,255,0.2);
}

/* Buttons */
button.primary {
    background: var(--accent-gradient) !important;
    border: none !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    transition: all 0.3s ease !important;
}

button.primary:hover {
    box-shadow: 0 0 15px var(--primary-glow) !important;
    transform: scale(1.02);
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}
::-webkit-scrollbar-track {
    background: rgba(0,0,0,0.1);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb {
    background: rgba(148, 163, 184, 0.3);
    border-radius: 10px;
}
::-webkit-scrollbar-thumb:hover {
    background: rgba(148, 163, 184, 0.5);
}

/* Tabs overriding */
.tabs button.selected {
    border-bottom-color: var(--primary) !important;
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    background: rgba(99, 102, 241, 0.1) !important;
}

footer { display: none !important; }
"""
