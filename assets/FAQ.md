# Frequently Asked Questions (FAQ)

## Q1: Do you have plans to extend SVI to Wan 2.2 and 720p resolution?

Yes! This is our top priority on the TODO list. We are actively working on upgrading SVI to support Wan 2.2 and higher resolutions including 720p.

**PS:** We are calling for TODO suggestions! We want to meet more real user needs: if you have any ideas, don't hesitate to share them in our Issues.

## Q2: Have you compared SVI-Talk with InfiniteTalk?

To be honest, SVI-Talk was developed under a tight schedule. We only fine-tuned it with 5k toy clips and didn't have time for further optimization.

In our internal comparisons with InfiniteTalk on very small-scale samples:

- **Lip-sync metrics**: Our metrics are very close to InfiniteTalk, though slightly lower (likely because we didn't have time to tune the audio cfg properly)
- **SVI advantages**: Significantly better text detail consistency; Superior visual quality (benefiting from our image restoration characteristics). However, these advantages don't show up in typical talking face metrics. (See below figure)
- **Long-form generation**: We tested 10-minute talking videos with absolutely no drifting issues

<p align="center">
  <img src="talk.png" alt="10-minute talking video without drifting" width="600"/>
</p>

## Q3: SVI-Film sometimes generates sub-optimal videos. Why?

This issue has two main causes:

1. **VAE encoding-decoding errors**: The VAE itself accumulates errors through repeated encoding-decoding, especially in static environments. If you repeatedly encode-decode the same image, you'll notice progressive quality degradation. Since our method operates in latent space, these errors can partially escape SVI's constraints.
2. **Limited training data scope**: Our SVI uses LoRA trained on small-scale datasets, so its style and error patterns are constrained by the training data types. When test images or text prompts differ significantly from the training distribution, sub-optimal generation can occur.

**Best solution**: Fine-tune with a small amount of video clips that match your target style/domain. This is the most effective way to adapt SVI to your specific use case. Moreover, LoRA not only learns error-elimination capabilities but also indirectly learns the generation style of the videos. So, you can better control the long-range style consistency by LoRA fintuning (like Tom & Jerry), which only needs several-hours tuning.

## Q4: Did you consider building upon the Self-Forcing series of works?

Initially, we did want to build upon Self-Forcing, but two critical issues led us to abandon this approach:

1. **T2V-only limitation**: Self-Forcing only supports text-to-video (T2V), whereas most application scenarios—such as talking faces—require image-to-video (I2V) capabilities. While I2V can easily accommodate T2V (simply by providing a T2I-generated first frame), the reverse is much more difficult.
2. **Model scale constraints**: Self-Forcing is based on a 1.3B parameter model, and we found that the visual quality could hardly reach the cinematic level we aimed for (e.g., our Iron Man demo).

## Q5: What do the parameters in test bash scripts mean?

- **`--num_motion_frames`**: Controls the number of cross-clip reference frames. In SVI-Film, this is used to ensure coherence across scene transitions.
- **`--num_clips`**: Specifies how many video clips to generate. Each clip represents 81 frames.
- **`--ref_pad_num`**: Controls the padding method for reference images:

  - **`-1`**: Pads with a random frame. Used for single-scene video generation (e.g., SVI-Shot/Dance/Talk). This simplifies the task into a restoration problem based on unpaired data with the reference image, thereby eliminating any drift or forgetting issues.
  - **`0`**: Pads with zeros. Relaxes the single-scene constraint. In this mode, when the subject moves out of the clip, there is a probability of forgetting.
