module.exports = {
  run: [
    // Windows: never upgrade packages while webui holds _safetensors_rust.pyd
    {
      method: "script.stop",
      params: {
        uri: "start.js"
      }
    },
    // Snapshot custom files (local-preserve + offline backup) before git/pip
    {
      method: "shell.run",
      params: {
        path: "app",
        venv: "env",
        message: [
          "python ../scripts/env_guard.py snapshot",
          "python ../scripts/env_guard.py stop_holders"
        ]
      }
    },
    // Launcher + monorepo pull (tracks mine/main with local customizations)
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Nested stock install only: separate app/.git (skipped for monorepo)
    {
      when: "{{exists('app/.git')}}",
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "python ../scripts/env_guard.py post_update"
        ]
      }
    },
    {
      method: "notify",
      params: {
        html: "Update complete. safetensors verified. Click Start — do not Update while the app is running."
      }
    }
  ]
}
