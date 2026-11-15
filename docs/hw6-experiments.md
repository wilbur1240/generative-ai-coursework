# First model

```python
unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=2,
    block_out_channels=(128, 256, 512),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "CrossAttnDownBlock2D",
    ),
    up_block_types=(
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    cross_attention_dim=512,  # CLIP text embedding dimension
)
optimizer = torch.optim.Adam(list(unet.parameters()), lr=lr)
noise_scheduler = DDPMScheduler()
```
20 epochs 108240 steps (batch_size=8)

DDIMScheduler inferencing

Score:

244.3059 / 0.1978 / 0.5336

# Second model
```python
unet = UNet2DConditionModel(
    sample_size=32,
    in_channels=4,
    out_channels=4,
    layers_per_block=3,
    block_out_channels=(128, 256, 512, 512),
    down_block_types=(
        "CrossAttnDownBlock2D",
        "AttnDownBlock2D",
        "CrossAttnDownBlock2D",
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "CrossAttnUpBlock2D",
        "AttnUpBlock2D",
        "CrossAttnUpBlock2D",
    ),
    cross_attention_dim=512,  # CLIP text embedding dimension
)
optimizer = torch.optim.Adam(list(unet.parameters()), lr=lr)
noise_scheduler = DDPMScheduler()
```
20 epochs 216480 steps batch_size=4
```python
scheduler = DPMSolverMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
)
scheduler.set_timesteps(100)
```
cfg when training and sampling

score:

132.7391 / 0.2762 / 0.6986

# Third model FiLM + CA
latents first pass through FiLM with char_emb and then make noise_pred from unet with action_emb

FiLM alone can’t rescue: the network simply hasn’t learned a robust “text → latent” mapping in the first place, so no amount of clever conditioning on those untrained weights will produce a red, branch-y “tree monster.”

```python
class FiLM(nn.Module):
    def __init__(self, cond_dim, num_channels):
        super().__init__()
        # maps [batch, cond_dim] -> [batch, 2 * num_channels]
        self.proj = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x, cond_emb):
        # x: [B, C, H, W], cond_emb: [B, cond_dim]
        gamma, beta = self.proj(cond_emb).chunk(2, dim=-1)
        # reshape to [B, C, 1, 1]
        gamma = gamma.view(-1, x.size(1), 1, 1)
        beta  = beta.view(-1, x.size(1), 1, 1)
        return gamma * x + beta
```
```python
        for t in prograce_bar:
            timestep_input = t.to(device)
            # FiLM modulate before cross-attn
            styled = film(latents, char_emb)
            styled = styled.repeat(2,1,1,1)
            # Predict noise and sample x_(t-1)
            noise_pred = unet(styled, timestep_input, encoder_hidden_states=action_emb).sample
            uncond_pred, cond_pred = noise_pred.chunk(2)
            guided = uncond_pred + guidance_scale * (cond_pred - uncond_pred)
            latents = scheduler.step(guided, t, latents).prev_sample
```
cfg when training and sampling

# Forth model FiLM + CA with LR scheduler warmup
latents first pass through FiLM with char_emb and then make noise_pred from unet with (char_emb + action_emb)
cosine_lr_shceduler warmup 500
cfg when training and sampling
119.9784 / 0.2823 / 0.7297

# FiFth model
time embedding layers, mid block CA
```python
unet = UNet2DConditionModel(
        sample_size=32,
        in_channels=4,
        out_channels=4,
        layers_per_block=3,
        block_out_channels=(128, 256, 512, 512),
        down_block_types=(
            "CrossAttnDownBlock2D",
            "AttnDownBlock2D",
            "CrossAttnDownBlock2D",
            "AttnDownBlock2D",
        ),
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=(
            "AttnUpBlock2D",
            "CrossAttnUpBlock2D",   # use class directly here
            "AttnUpBlock2D",
            "CrossAttnUpBlock2D",
        ),
        cross_attention_dim=512,  # CLIP text embedding dimension
        resnet_time_scale_shift="scale_shift",
        time_embedding_type="fourier",
        time_embedding_act_fn="gelu",
        timestep_post_act="gelu",
        dropout=0.1,
    )
```
```python
optimizer = torch.optim.AdamW(
        list(unet.parameters()) + list(film.parameters()),
        lr=lr,
        weight_decay=0.01
    )
scheduler = DPMSolverMultistepScheduler.from_pretrained(
        "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
    )
    scheduler = DPMSolverMultistepScheduler.from_config(
        scheduler.config, algorithm_type="sde-dpmsolver++"
    )
```
117.0652 / 0.2698 / 0.7077