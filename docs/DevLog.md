
## [10-23-Preview] Wan 2.2 5B SVI and Tips for Implementing Custom SVI

We have successfully deployed SVI to Wan 2.2 5B and present preliminary results below (around 10-min inference time for 1-min video on 1 H100). SVI shows good generalizability and can be adapted to different models. Now, we are actively optimizing deployment and training procedures and will release the Wan 2.2 model after stress testing. Recently, we have received several inquiries about deploying SVI to other model architectures. We are publishing the following deployment tips:

1. For models with smaller sizes (e.g., 1.3B/5B), consider using only reference image error rather than full error correction.
2. Ensure that each clip uses a different seed during inference. Our experiments show that using identical noise across all clips accumulates artifacts. 
3. If your computational resources are limited (batch size cannot exceed 64), consider reducing `--clean_prob` to accelerate the network's error-recycling learning.
4. We discovered in Wan 2.1 that when VAE is enabled with bfloat16 and tiling, and the generated video contains static backgrounds, VAE accumulates certain artifacts. The reason is thate these are pixel-level errors, and SVI, the latent-level method, has not encountered during training.
5. For non-Film versions of SVI, the randomly selected padding frame should ideally be chosen from frames outside the current video clip. Besides, you can also reduce `--clean_prob` accordingly, as this scenario is simpler: the model learns image restoration based on unpaired reference images.

[Important Note] While SVI addresses the drift problem, it's better to understand that due to the inherent randomness of the last frame and text prompts, generated semantic content may still exhibit issues such as temporal glitches, object merging, loss of fine details, and unrealistic motion. These phenomena can occur even in short-form videos. **Please note that these are limitations of the base model**, not specific to SVI. SVI can reduce these issues through error recycling, but cannot entirely eliminate them.

<div align="center">
    <img src="./wan22_preview.png" alt="Preview">
</div>

## [10-21-Opt] Error Memory Storage Optimization

During SVI training, we employ `--clean_prob=0.5` to enable error-free training. Recently, we discovered that error from error-free inputs will contaminate the error buffer. To address this, we introduced the `--clean_buffer_update_prob=0.1` parameter to control the update probability for such errors. The svi-film-opt-10212025 version shows noticeable improvements in image quality, particularly for extended-duration generation. This validates that high-quality errors are crucial for mitigating long-horizen drift.

### 10-Prompt I2V (50-sec)

| svi-film-opt-10212025 | svi-film | svi-film-transition |     Wan 2.1 (baseline) |       
| ---------------------- | -------- | ------------------- |  ------------------- |     
| 63.09 | 62.25 | 62.40 |52.83|

### 50-Prompt I2V (250-sec)

| svi-film-opt-10212025 | svi-film | svi-film-transition | Wan 2.1 (baseline) |            
| ---------------------- | -------- | ------------------- |------------------- |
| 61.92 | 59.43 | 57.91 |42.31|

### Optimization Implementation in train_x.py

```python
if use_clean_input:
    p = random.random()
    if p < self.clean_buffer_update_prob:
        self._update_error_buffers_local(noise_error, y_error, timestep)
else:
    self._update_error_buffers_local(noise_error, y_error, timestep)
```