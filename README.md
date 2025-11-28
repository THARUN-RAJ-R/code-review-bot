# Code Review Bot (Flask + MongoDB + Qwen)

Backend service for automatic code reviews used by a Zoho SalesIQ Zobot.

## Endpoints

- `GET /ping` health check.
- `POST /review-code` run model-powered code review and store in MongoDB.
- `POST /feedback` attach user satisfaction + feedback to a review.
- `GET /admin?pw=...` simple admin dashboard (password from `ADMIN_PASSWORD`).

## Run locally

1. Create `.env` based on `.env.example`.
2. Activate virtualenv: `./venv/Scripts/Activate.ps1` (Windows PowerShell).
3. Install deps: `pip install -r requirements.txt`.
4. Run: `python app.py`.
