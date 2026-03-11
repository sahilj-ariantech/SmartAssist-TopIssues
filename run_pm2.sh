#!/bin/bash
set -euo pipefail
cd /Users/mustafa-ats/Desktop/mustafa/Code/python/conversion-probability/SmartAssist-TopIssues
exec /Users/mustafa-ats/Desktop/mustafa/Code/python/conversion-probability/SmartAssist-TopIssues/.venv/bin/python -m uvicorn api:app --host 0.0.0.0 --port 8000
