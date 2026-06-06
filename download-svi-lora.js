module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      message: [
        "hf download vita-video-gen/svi-model --include \"version-2.0/*\" --local-dir models/loras/SVI-LoRA"
      ],
      path: "app"
    }
  }]
}
