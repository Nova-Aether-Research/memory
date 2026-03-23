# Grok API + Local GUI Setup on MacBook Air
**Date:** March 2025 (setup completed around March 23, 2026 timestamp reference)  
**Author:** William Chappell (@chapcl)  
**Purpose:** Create a lightweight, native macOS GUI to interact directly with Grok API (xAI) for research cycles, paper writing/rewriting, hardware state logging, and Git sync — without relying on web browser or external LLMs for core tasks.

## Background & Motivation
- Existing workflow (Nova-Aether-Research/memory repo):
  - Strong autonomous cycle execution via AI-Scientist-v2 (BFTS, ideation, code gen, execution on Linux/GPU machines).
  - Logs saved in `memory/logs/cycle_*.md`.
  - Papers manually written via web Grok → saved in `memory/papers/`.
  - Outside LLMs used for paper review/feedback → good practice, to be preserved.
- Pain points:
  - No direct way to ask Grok API to write/rewrite papers locally.
  - No integrated way to log current MacBook Air hardware/software state into repo.
  - Desired: simple GUI with buttons for cycles, paper tasks, chat, hardware logging, Git commits.

## What We Built
A single-file Python/Tkinter GUI application: `grok_research_gui.py`

Location (recommended): `~/Nova-Aether-Research/tools/grok_research_gui.py`  
(or wherever convenient, e.g., `~/grok_research_gui.py`)

### Core Features Implemented
- Sidebar buttons:
  - 🚀 Begin New Cycle → auto-numbers & saves to `logs/cycle_xx.md`
  - 🔄 Continue Last Cycle → appends to latest cycle file
  - 📝 Write Paper Draft → synthesizes into `papers/YYYY-MM-DD-paper-draft.md`
  - ✏️ Rewrite Paper (with feedback) → takes pasted external feedback, rewrites latest paper
  - 💻 Log Hardware/Software State → appends JSON snapshot to `logs/hardware_state.md` + Git commit
  - 📤 Manual Git Commit → add/commit everything in repo
- Live chat window: Grok responses appear in real time
- Bottom input box: free-form commands to Grok (hit SEND or Return)
- Auto Git commit after structured actions
- Real-time system state (CPU, memory, disk, OS version, etc.) included in every saved log
- Dark theme, MacBook Air friendly layout (1200×800)

### Technical Details
- Language: Python 3 (Homebrew 3.14 in this case)
- Dependencies: `tkinter` (built-in), `psutil`, `openai` (pip installed in venv)
- API Client: OpenAI-compatible client pointed at `https://api.x.ai/v1`
- Model used: `grok-4.20-reasoning` (configurable)
- Virtual environment: `~/Nova-Aether-Research/tools/venv/` (or similar)
- Repo integration: `REPO_PATH="$HOME/Nova-Aether-Research/memory"` (permanent in `~/.zshrc`)
- API Key: `XAI_API_KEY` (permanent in `~/.zshrc`)

## How to Run (Quick Reference)
```bash
cd ~/Nova-Aether-Research/tools           # or wherever script lives
source venv/bin/activate
python grok_research_gui.py
