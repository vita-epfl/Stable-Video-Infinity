module.exports = {
  run: [{
    method: "shell.run",
    params: {
      venv: "env",
      message: [
        "hf download Wan-AI/Wan2.1-I2V-14B-480P --local-dir models/diffusers/Wan2.1-I2V-14B-480P"
      ],
      path: "app"
    }
  }]
}
