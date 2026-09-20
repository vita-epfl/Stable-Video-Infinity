module.exports = {
  run: [
    // Clone ComfyUI
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/comfyanonymous/ComfyUI app",
        ]
      }
    },
    // Clone ComfyUI-Manager
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/ltdrdata/ComfyUI-Manager",
        ],
        path: "app/custom_nodes"
      }
    },
    // Clone ComfyUI-WanVideoWrapper (required for SVI)
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/kijai/ComfyUI-WanVideoWrapper",
        ],
        path: "app/custom_nodes"
      }
    },
    // Clone ComfyUI-KJNodes (required for SVI)
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/kijai/ComfyUI-KJNodes",
        ],
        path: "app/custom_nodes"
      }
    },
    // Clone ComfyUI-VideoHelperSuite (useful for video workflows)
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite",
        ],
        path: "app/custom_nodes"
      }
    },
    // Clone Stable-Video-Infinity for workflows
    {
      method: "shell.run",
      params: {
        message: [
          "git clone https://github.com/vita-epfl/Stable-Video-Infinity svi",
        ],
        path: "workflows"
      }
    },
    // Install all dependencies into the main app/env venv
    {
      method: "shell.run",
      params: {
        venv: "env",
        path: "app",
        message: [
          "uv pip install -r requirements.txt",
          "uv pip install -r custom_nodes/ComfyUI-WanVideoWrapper/requirements.txt",
          "uv pip install -r custom_nodes/ComfyUI-KJNodes/requirements.txt",
          "uv pip install -r custom_nodes/ComfyUI-VideoHelperSuite/requirements.txt",
        ]
      }
    },
    // Install PyTorch with correct version for platform
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
    // Link model directories for sharing with other Pinokio apps
    {
      method: "fs.link",
      params: {
        drive: {
          checkpoints: "app/models/checkpoints",
          clip: "app/models/clip",
          clip_vision: "app/models/clip_vision",
          controlnet: "app/models/controlnet",
          embeddings: "app/models/embeddings",
          loras: "app/models/loras",
          upscale_models: "app/models/upscale_models",
          vae: "app/models/vae",
          diffusers: "app/models/diffusers",
          unet: "app/models/unet",
        },
        peers: [
          "https://github.com/cocktailpeanut/fluxgym.git",
          "https://github.com/cocktailpeanutlabs/automatic1111.git",
          "https://github.com/cocktailpeanutlabs/fooocus.git",
          "https://github.com/cocktailpeanutlabs/comfyui.git",
          "https://github.com/pinokiofactory/stable-diffusion-webui-forge.git"
        ]
      }
    },
    // Link output directory
    {
      method: "fs.link",
      params: {
        drive: {
          output: "app/output"
        }
      }
    },
    // Copy SVI workflows to ComfyUI workflows folder
    {
      method: "fs.copy",
      params: {
        src: "workflows/svi/comfyui_workflow_svi_1.0",
        dest: "app/user/default/workflows/svi"
      }
    },
  ]
}
