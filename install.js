module.exports = {
  run: [
    // Monorepo: app/ is tracked in Ryan's fork. After Reset, restore it from
    // the local checkout (never clone stock upstream), then fast-forward to mine/main.
    {
      when: "{{!exists('app/webui.py')}}",
      method: "shell.run",
      params: {
        message: [
          "git checkout HEAD -- app",
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        message: [
          "git remote get-url mine || git remote add mine https://github.com/Treyu2023/FlashVSR_plus_pinokio.git",
          "git pull --ff-only mine main",
        ]
      }
    },
    
    {
      method: "script.start",
      params: {
        uri: "torch.js",
        params: {
          venv: "env",
          path: "app",
        }
      }
    },
    
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "python ../scripts/env_guard.py reapply",
          "python ../scripts/env_guard.py post_update"
        ]
      }
    },
  ]
}
