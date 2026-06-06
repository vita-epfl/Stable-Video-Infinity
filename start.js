module.exports = {
  daemon: true,
  run: [
    {
      method: "shell.run",
      params: {
        venv: "env",
        env: {
          PYTORCH_ENABLE_MPS_FALLBACK: "1",
          TOKENIZERS_PARALLELISM: "false"
        },
        path: "app",
        message: [
          "{{platform === 'win32' && gpu === 'amd' ? 'python main.py --directml --port ' + port : 'python main.py --port ' + port}}"
        ],
        on: [{
          // Capture the URL when server starts
          event: "/(http:\\/\\/[0-9.:]+)/",
          done: true
        }]
      }
    },
    {
      // Set the local variable 'url' for the UI
      method: "local.set",
      params: {
        url: "{{input.event[1]}}"
      }
    }
  ]
}
