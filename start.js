module.exports = {
  daemon: true,
  run: [
    // Auto-repair broken safetensors (Windows file-lock corruption) before webui import.
    // conda skip: Pinokio auto-launch fires while Pinokio is still activating its own
    // base env. Both run the Khronos OpenCL hook, which locks
    // Library\etc\OpenCL\vendors\temp.txt, and the first Start dies with
    // "The process cannot access the file because it is being used by another process."
    // This app's venv python does not need conda base.
    {
      method: "shell.run",
      params: {
        conda: { skip: true },
        path: "app",
        env: {
          PYTHONUNBUFFERED: "1",
          PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512",
        },
        message: [
          "set \"VIRTUAL_ENV=%CD%\\env\" && set \"PATH=%CD%\\env\\Scripts;%PATH%\" && env\\Scripts\\python.exe -u ..\\scripts\\env_guard.py preflight"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        conda: { skip: true },
        env: {
          PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512",
          PYTHONUNBUFFERED: "1",
        },
        path: "app",
        message: [
          "set \"VIRTUAL_ENV=%CD%\\env\" && set \"PATH=%CD%\\env\\Scripts;%PATH%\" && env\\Scripts\\python.exe -u webui.py"
        ],
        on: [{
          // The regular expression pattern to monitor.
          // When this pattern occurs in the shell terminal, the shell will return,
          // and the script will go onto the next step.
          "event": "/http:\\/\\/\\S+/",

          // "done": true will move to the next step while keeping the shell alive.
          // "kill": true will move to the next step after killing the shell.
          "done": true
        }]
      }
    },
    {
      // This step sets the local variable 'url'.
      // This local variable will be used in pinokio.js to display the "Open WebUI" tab when the value is set.
      method: "local.set",
      params: {
        // the input.event is the regular expression match object from the previous step
        url: "{{input.event[0]}}"
      }
    }
  ]
}
