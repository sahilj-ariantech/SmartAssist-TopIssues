module.exports = {
  apps: [
    {
      name: 'smartassist',
      cwd: '/Users/mustafa-ats/Desktop/mustafa/Code/python/conversion-probability/SmartAssist-TopIssues',
      script: '/Users/mustafa-ats/Desktop/mustafa/Code/python/conversion-probability/SmartAssist-TopIssues/.venv/bin/uvicorn',
      args: 'api:app --host 0.0.0.0 --port 8000',
      interpreter: 'none',
      autorestart: true,
      watch: false,
      max_restarts: 10,
      restart_delay: 3000
    }
  ]
};
