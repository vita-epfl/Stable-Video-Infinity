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