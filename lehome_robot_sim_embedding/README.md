# LeHome Robot Sim Multi-View Embedding

This component trains a top-centric multi-view VAE embedder for the LeHome robot simulation dataset.

Default dataset:

```text
E:/Lehome-Dataset/lehome_round_2_dataset/sim_dataset/robot_sim_ft_lehome_all_garment_data_z180
```

The inspected dataset has:

```text
top RGB camera   : observation.images.top_rgb
left wrist RGB   : observation.images.left_rgb
right wrist RGB  : observation.images.right_rgb
state            : observation.state, 12D
action           : actions, 12D
fps              : 30
episodes         : 1000
```

## Architecture

Each timestep embedding is encoded from the top, left wrist, and right wrist RGB frames. Full robot state/action vectors remain supervision targets for the reconstruction heads, but are not posterior-encoder inputs.

```text
top frame        -> frozen DINOv3-S+ -> top patch/global tokens
left wrist frame -> frozen DINOv3-S+ -> left patch/global tokens
right wrist frame-> frozen DINOv3-S+ -> right patch/global tokens
```

Learned view embeddings are added before fusion:

```text
top tokens   + e_top
left tokens  + e_left
right tokens + e_right
```

Fusion is top-centric:

```text
Q = top tokens
K,V = concat(left wrist tokens, right wrist tokens)
```

Then:

```text
top-query cross attention
-> top-token self attention
-> latent slot resampler
-> 8 slots x 256 dim
-> 2048D timestep latent
```

The posterior is diagonal Gaussian:

```text
q(z_t | top_t, left_t, right_t) = N(mu_t, diag(sigma_t^2))
```

Training samples from the posterior when `stochastic: true`; export uses `mu_t`.

## Loss Heads

Current timestep heads decode from `z_t`:

```text
top DINO patches/global
left wrist DINO patches/global
right wrist DINO patches/global
state vector
action vector
```

Temporal heads use conditional future priors over horizons:

```text
H = [1, 2, 4, 8, 16, 32]
```

```text
p(z_{t+h} | z_t, s_t, a_t, h) = N(mu_prior_{t,h}, diag(sigma_prior_{t,h}^2))
```

Future heads decode from `mu_prior_{t,h}`:

```text
future top DINO patches/global
future left DINO patches/global
future right DINO patches/global
future state
future action
```

## Total Loss

Feature reconstruction is:

```text
L_feat(pred, target) =
  MSE(LN(pred), LN(target))
+ cosine_weight * (1 - cosine(pred, target))
```

Current loss:

```text
L_current =
  w_top_patch   L_feat(top_patch_hat_t, top_patch_t)
+ w_left_patch  L_feat(left_patch_hat_t, left_patch_t)
+ w_right_patch L_feat(right_patch_hat_t, right_patch_t)
+ w_top_global  L_feat(top_global_hat_t, top_global_t)
+ w_left_global L_feat(left_global_hat_t, left_global_t)
+ w_right_global L_feat(right_global_hat_t, right_global_t)
+ w_state       Huber(state_hat_t, state_t)
+ w_action      Huber(action_hat_t, action_t)
+ beta_kl       KL(q(z_t) || N(0,I))
+ w_var         variance_floor(mu_t)
```

Future loss:

```text
w_h = (1 / sqrt(h)) / sum_k(1 / sqrt(k))
```

```text
L_future =
sum_h w_h [
  w_mean       L_feat(mu_prior_{t,h}, stopgrad(mu_{t+h}))
+ w_cond_kl    KL(q(z_{t+h}) || p(z_{t+h} | z_t, s_t, a_t, h))
+ visual future reconstruction losses for top/left/right
+ w_state      Huber(state_hat_{t+h}, state_{t+h})
+ w_action     Huber(action_hat_{t+h}, action_{t+h})
]
```

Final cumulative objective:

```text
L_total = joint_spatial * L_current + joint_temporal * L_future
```

## Commands

Validate a few episodes:

```powershell
cd D:\LeHome-Challenge\Lehome-Spline-ICRA2027\lehome_robot_sim_embedding
python validate_dataset.py --config configs\default.yaml --max-episodes 3
```

Inspect schema:

```powershell
python scripts\inspect_dataset.py E:\Lehome-Dataset\lehome_round_2_dataset\sim_dataset\robot_sim_ft_lehome_all_garment_data_z180
```

Train:

```powershell
python train.py --config configs\default.yaml
```

Run one cumulative full epoch from an existing checkpoint:

```powershell
python train.py --config configs\joint_full_epoch.yaml --init-checkpoint outputs\robot_sim_multiview_vae\checkpoints\final.pt
```

Export embeddings:

```powershell
python export_embeddings.py --config configs\default.yaml --checkpoint outputs\robot_sim_multiview_vae\checkpoints\best.pt
```

Fit splines:

```powershell
python fit_splines.py --config configs\default.yaml
```
