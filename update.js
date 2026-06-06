module.exports = {
  run: [
    // Update launcher scripts
    {
      method: "shell.run",
      params: {
        message: "git pull"
      }
    },
    // Update ComfyUI
    {
      method: "shell.run",
      params: {
        path: "app",
        message: "git pull"
      }
    },
    // Update ComfyUI-Manager
    {
      method: "shell.run",
      params: {
        path: "app/custom_nodes/ComfyUI-Manager",
        message: "git pull"
      }
    },
    // Update ComfyUI-WanVideoWrapper
    {
      method: "shell.run",
      params: {
        path: "app/custom_nodes/ComfyUI-WanVideoWrapper",
        message: "git pull"
      }
    },
    // Update ComfyUI-KJNodes
    {
      method: "shell.run",
      params: {
        path: "app/custom_nodes/ComfyUI-KJNodes",
        message: "git pull"
      }
    },
    // Update ComfyUI-VideoHelperSuite
    {
      method: "shell.run",
      params: {
        path: "app/custom_nodes/ComfyUI-VideoHelperSuite",
        message: "git pull"
      }
    },
    // Update Stable-Video-Infinity workflows
    {
      method: "shell.run",
      params: {
        path: "workflows/svi",
        message: "git pull"
      }
    },
    // Reinstall dependencies
    {
      method: "shell.run",
      params: {
        path: "app",
        venv: "env",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    // Reinstall custom node dependencies
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/custom_nodes/ComfyUI-WanVideoWrapper",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/custom_nodes/ComfyUI-KJNodes",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    },
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app/custom_nodes/ComfyUI-VideoHelperSuite",
        message: [
          "uv pip install -r requirements.txt"
        ]
      }
    }
  ]
}
