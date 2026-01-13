ISSUE_IMAGE_TO_TEXT = {
    "![image](https://github.com/user-attachments/assets/9174c213-8a5e-48ea-b3e2-fafa239ffaad)": r"""Unlike KAEs [45, 44] that introduce a loss term for rigorous reconstruction of the lookback-window series, we feed the residual $X^{(b+1)}$ as the input of next block for learning a corrective operator. And the model forecast $Y$ is the sum of predicted components $Y_{\text{var}}^{(b)}, Y_{\text{inv}}^{(b)}$ gathered from all Koopa Blocks:

$$
X^{(b+1)}=X_{\text{var}}^{(b)} - \hat{X}_{\text{var}}^{(b)}, \quad Y=\sum\left(Y_{\text{var}}^{(b)}+Y_{\text{inv}}^{(b)}\right)
$$""",
    "![Image](https://github.com/user-attachments/assets/692a8b5f-e438-4cbd-b909-313e4da2671f)": r"""Hyperparameter tuning. Training and hyperparameter tuning were done using nested 5 -fold CV, stratified by subject SBP and DBP, except for the UCI dataset with HOO. We tuned ML models by grid searching the parameter-search-space shown in Table 4 and monitoring the MAE performance of validation sets. For the DL models, we used the Mean Squared Error (MSE) as the loss function, the Adam optimizer, and early stopping with the patience of 15 epochs in the validation loss. Their hyperparameters were greedily searched using the Optuna Toolkit ${ }^{62}$ to monitor the MAE performance. Table 4 lists the tuned hyperparameters.""",
    "![Image](https://github.com/user-attachments/assets/adb29135-1402-4748-9593-af1eed7925e6)": r"""Fold 0:
Train folds: [2, 3, 4]
Validation fold: [1]
Test fold: [0]
Fold 1:
Train folds: [0, 3, 4]
Validation fold: [2]
Test fold: [1]
Fold 2:
Train folds: [0, 1, 4]
Validation fold: [3]
Test fold: [2]
Fold 3:
Train folds: [0, 1, 2]
Validation fold: [4]
Test fold: [3]
Fold 4:
Train folds: [1, 2, 3]
Validation fold: [0]
Test fold: [4]""",
    '<img width="545" alt="image" src="https://user-images.githubusercontent.com/46547949/179350209-57da5aff-7d6f-46c1-8219-4a0540f04206.png">': r"""x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
x, (H, W) = self.patch_embed_d(x)
for i, blk in enumerate(self.blocks_d):
        x = blk(x, H, W, self.relative_pos_d)

B, N, C = x.shape
x = self._fc(x.permute(0, 2, 1).reshape(B, C, H, W))
x = self._bn(x)
x = self._swish(x)
x = self._avg_pooling(x).flatten(start_dim=1)
x = self._drop(x)
x = self.pre_logits(x)
return x""",
    "![image](https://user-images.githubusercontent.com/42391631/200701745-3891b15a-36f9-49f4-935f-d166706f88e7.png)": r"""Denote $Y_{\mathrm{M}}$ the result after applying mask $\mathbf{M}$ to $\mathbf{Y}$. The training objective is to minimize the negative log-likelihood of the masked tokens:
    
$$
\mathcal{L}_{\text {mask }}=-\underset{\mathbf{Y} \in \mathcal{D}}{\mathbb{E}}\left[\sum_{\forall i \in[1, N], m_{i}=1} \log p\left(y_{i} \mid Y_{\overline{\mathbf{M}}}\right)\right]
$$

Concretely, we feed the masked $Y_{\overline{\mathbf{M}}}$ into a multi-layer bidirectional transformer to predict the probabilities $P\left(y_{i} \mid Y_{\overline{\mathbf{M}}}\right)$ for each masked token, where the negative log-likelihood is computed as the cross-entropy between the ground-truth one-hot token and predicted token. Notice the key difference to autoregressive modeling: the conditional dependency in MVTM has two directions, which allows image generation to utilize richer contexts by attending to all tokens in the image.""",
    '<img width="311" alt="image" src="https://user-images.githubusercontent.com/28287182/233439824-a683a7b8-fdb2-4fa9-ae3c-a8622019243e.png">': r"$\boldsymbol{\Sigma}=\frac{1}{N} \sum_{c=1}^{K} \sum_{i: y_{i}=c}\left(f\left(\boldsymbol{x}_{i}\right)-\boldsymbol{\mu}_{c}\right)\left(f\left(\boldsymbol{x}_{i}\right)-\boldsymbol{\mu}_{c}\right)^{\top}$",
    '<img width="894" alt="image" src="https://user-images.githubusercontent.com/28287182/233440161-3b6e8727-9e95-4f23-8acc-f231ef8ae992.png">': r"""class_means = []
class_cov_invs = []
class_covs = []
for c in range(indist_classes):

  mean_now = np.mean(indist_train_embeds_in_touse[indist_train_labels_in == c],axis=0)

  cov_now = np.cov((indist_train_embeds_in_touse[indist_train_labels_in == c]-(mean_now.reshape([1,-1]))).T)
  class_covs.append(cov_now)
  # print(c)

  eps = 1e-8
  cov_inv_now = np.linalg.inv(cov_now)

  class_cov_invs.append(cov_inv_now)
  class_means.append(mean_now)

#the average covariance for class specific
class_cov_invs = [np.linalg.inv(np.mean(np.stack(class_covs,axis=0),axis=0))]*len(class_covs)""",
    "![Image](https://github.com/user-attachments/assets/929b1e5b-52d1-4207-b23c-279678c3f4d3)": r"""$$
\begin{aligned}
\mathbb{E}\left[\sum_{d=1}^{D} & \frac{\left\|\hat{x}_{1}^{d}\left(\mathbf{T}_{t, \tilde{t}}\right)-x_{1}^{d}\right\|^{2}}{1-t}-\log p_{\theta}\left(a_{1}^{d} \mid \mathbf{T}_{t, \tilde{t}}\right)\right. \\
& \left.+\frac{\left\|\log _{r_{t}^{d}}\left(\hat{r}_{1}^{d}\left(\mathbf{T}_{t}\right)\right)-\log _{r_{t}^{d}}\left(r_{1}^{d}\right)\right\|^{2}}{1-t}\right]
\end{aligned}
$$""",
    "![Image](https://github.com/user-attachments/assets/06b907e5-9278-48a3-91ab-b76b1b41261a)": r"""# Timestep used for normalization.
r3_t = noisy_batch['r3_t'] # (B, 1)
so3_t = noisy_batch['so3_t'] # (B, 1)
cat_t = noisy_batch['cat_t'] # (B, 1)
r3_norm_scale = 1 - torch.min(
        r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)
so3_norm_scale = 1 - torch.min(
        so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)) # (B, 1, 1)""",
    "![Image](https://github.com/user-attachments/assets/9da70510-1a5f-44a4-80ac-a3941394369b)": r"""trans_error = (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
trans_loss = training_cfg.translation_loss_weight * torch.sum(
        trans_error ** 2 * loss_mask[..., None],
        dim=(-1, -2)
) / loss_denom
trans_loss = torch.clamp(trans_loss, max=5)""",
    "![image](https://user-images.githubusercontent.com/1367018/127979872-541b6810-a82d-46a2-970f-47de5212be0f.png)": r"iteration $j \in\{10,50,200,500,1000,2000,4000,6000,8000\}$",
    "![image](https://user-images.githubusercontent.com/1367018/127980289-234bdd05-0fd0-42a3-a53e-d1d865f719e4.png)": r"""if 'patches' in self.norm:
    if 10 < it <= 50:
        p = self.p_init / 2
    elif 50 < it <= 200:
        p = self.p_init / 4
    elif 200 < it <= 500:
        p = self.p_init / 8
    elif 500 < it <= 1000:
        p = self.p_init / 16
    elif 1000 < it <= 2000:
        p = self.p_init / 32
    elif 2000 < it <= 4000:
        p = self.p_init / 64
    elif 4000 < it <= 6000:
        p = self.p_init / 128
    elif 6000 < it <= 8000:
        p = self.p_init / 256
    elif 8000 < it:
        p = self.p_init / 512
    else:
        p = self.p_init""",
    '<img width="782" alt="Image" src="https://github.com/user-attachments/assets/76ec739b-a09d-481c-a346-39ea62a2aa41" />': r"""Velocity Consistency Loss While Condition 1 directly constraints the vector field to be consistent, learning vector fields that only satisfy Condition 1 may lead to trivial solutions. On the other hand, Condition 2 ensures the consistency of the vector field from a trajectory viewpoint, offering a way to directly define straight flows. Motivated by this, Consistency-FM learns a consistency vector field to satisfy both conditions:

$$
\begin{aligned}
& \mathcal{L}_{\theta}=E_{t \sim \mathcal{U}} E_{x_{t}, x_{t+\Delta t}}\left\|f_{\theta}\left(t, x_{t}\right)-f_{\theta^{-}}\left(t+\Delta t, x_{t+\Delta t}\right)\right\|_{2}^{2}+\alpha\left\|v_{\theta}\left(t, x_{t}\right)-v_{\theta^{-}}\left(t+\Delta t, x_{t+\Delta t}\right)\right\|_{2}^{2} \\
& f_{\theta}\left(t, x_{t}\right)=x_{t}+(1-t) * v_{\theta}\left(t, x_{t}\right)
\end{aligned}
$$

where $\mathcal{U}$ is the uniform distribution on $[0,1-\Delta t], \alpha$ is a positive scalar, $\Delta t$ denotes a time interval which is a small and positive scalar. $\theta^{-}$denotes the running average of past values of $\theta$ using exponential moving average (EMA), $x_{t}$ and $x_{t+\Delta t}$ follows a pre-defined distribution which can be efficiently sampled, for example, VP-SDE [1] or OT path [7]. Note that by setting $t=1$, Condition 2 implies that $\gamma_{x}(t)+(1-t) * v\left(t, \gamma_{x}(t)\right)=\gamma_{x}(1) \sim p_{1}$, and thus training with $\mathcal{L}_{\theta}$ can not only regularize the velocity but also learn the data distribution. Furthermore, if Condition 2 is met, then the straight flows $\gamma_{x}(t)+(1-t) * v\left(t, \gamma_{x}(t)\right)$ can directly predict $x_{1}$ from each time point $t$ [15].""",
    "![image](https://user-images.githubusercontent.com/7959396/233814138-218c12b3-be06-4041-bb4d-2195ff198f42.png)": r"""$$
\begin{aligned}
\mathcal{L}_{r}\left(\boldsymbol{w}, \boldsymbol{\lambda}_{r}\right) & =\frac{1}{2} \sum_{i=1}^{N_{r}} m\left(\lambda_{r}^{i}\right)\left|\mathcal{N}_{\boldsymbol{x}, t}\left[u\left(\boldsymbol{x}_{r}^{i}, t_{r}^{i} ; \boldsymbol{w}\right)\right]-f\left(\boldsymbol{x}_{r}^{i}, t_{r}^{i}\right)\right|^{2} \\
\mathcal{L}_{b}\left(\boldsymbol{w}, \boldsymbol{\lambda}_{b}\right) & =\frac{1}{2} \sum_{i=1}^{N_{b}} m\left(\lambda_{b}^{i}\right)\left|\mathcal{B}_{\boldsymbol{x}, t}\left[u\left(\boldsymbol{x}_{r}^{i}, t_{r}^{i} ; \boldsymbol{w}\right)\right]-g\left(\boldsymbol{x}_{b}^{i}, t_{b}^{i}\right)\right|^{2} \\
\mathcal{L}_{0}\left(\boldsymbol{w}, \boldsymbol{\lambda}_{0}\right) & =\frac{1}{2} \sum_{i=1}^{N_{0}} m\left(\lambda_{0}^{i}\right)\left|u\left(\boldsymbol{x}_{0}^{i}, 0 ; \boldsymbol{w}\right)-h\left(\boldsymbol{x}_{0}^{i}\right)\right|^{2}
\end{aligned}
$$""",
    '<img width="678" height="144" alt="Image" src="https://github.com/user-attachments/assets/6c47903d-b14d-437c-84f5-677bbaa0dcfc" />': r"""$$
c_{i}=\operatorname{Conf}\left(\boldsymbol{x}_{i}\right)=\frac{1}{K} \sum_{k=1}^{K} \mathbb{1}\left(\boldsymbol{\hat{y}}_{i}=\boldsymbol{y}_{i}{{}^{(k)}}\right)
$$""",
    "![Screenshot from 2022-11-28 12-22-00](https://user-images.githubusercontent.com/32613612/204253818-c9414878-3ad0-466d-996f-da9f370071fa.png)": r"""$$
\begin{aligned}
& P_{1}(\varphi)=\lambda_{1} \sum_{i, j=1}^{m} \xi\left(x_{i}, y_{j}\right)^{2} \\
& P_{2}(\varphi)=\lambda_{2} \sum_{x, y \in B_{x} \cup B_{y}} \min (\xi(x, y), 0)^{2}
\end{aligned}
$$
""",
    '<img width="516" height="279" alt="Image" src="https://github.com/user-attachments/assets/a9dbfd33-f57f-46c6-bb8e-3a81ea95c87a" />': r"""n = matches.shape[0] > 0
m0, m1, _ = matches.transpose().astype(np.int16)
for i, gc in enumerate(gt_classes):
    j = m0 == i
    if n and sum(j) == 1:
        self.matrix[gc, detection_classes[m1[j]]] += 1  # correct
    else:
        self.matrix[gc, self.nc] += 1  # background FP

if n:
    for i, dc in enumerate(detection_classes):
        if not any(m1 == i):
            self.matrix[self.nc,""",
    "![image](https://user-images.githubusercontent.com/36615789/83992749-48326800-a98c-11ea-9fe9-140ed1c96e61.png)": r"""$$
\mathcal{L}_{\text{rec}} = \sum_{l \in \text{layers}} \left\| (C_l^* (x) - C_l^*(G(z, \mathcal{F}, \mathcal{M})) \cdot m_l \right\|_1 \tag{3}
$$""",
    "![Screenshot from 2021-03-30 11-53-58](https://user-images.githubusercontent.com/8075304/112931388-a6fce400-914e-11eb-8f8b-c434d6020727.png)": r"""$$d(p, q) = 1 - \frac{2}{1 + \exp(\|e_p - e_q\|^2)}.\tag{1}$$""",
    '<img width="982" alt="image" src="https://github.com/user-attachments/assets/50aa8b53-d2ec-4eb0-8212-b5e7451a6cd7">': r"""Specifically, we calculate the $\mathfrak{s o}(3)$ element corresponding to the relative rotation between $r_{0}$ and $r_{t}$, given by $r_{t}^{\top} r_{0}$. We divide by $t$ to get a vector which is an element of $\mathfrak{s o}(3)$ and corresponds to the skew-symmetric matrix representation of the velocity vector pointing towards the target $r_{1}$. Finally, we parallel-transport the velocity vector to the tangent space $\mathcal{T}_{r_{t}} \mathrm{SO}(3)$ using left matrix multiplication by $r_{t}$. These operations can be concisely""",
    "![image](https://github.com/user-attachments/assets/b9ce18a6-87c9-4aa9-bb53-b439655fb9ec)": r"""For image editing tasks, the objective is to modify specific regions of the input image while keeping other areas unchanged. Therefore, the difference between the input image and the target image is often small, which allows the model to learn an unexpected shortcut: simply copying the input image as the output to make the related training loss very low. To mitigate this phenomenon, we amplify the loss in the regions of the image where changes occur. More specifically, we calculate the loss weights for each region based on latent representations of input image $\mathbf{x}^{\prime}$ and target image $\mathbf{x}$ :

$$
w_{i, j}= \begin{cases}1 & \text { if } \mathbf{x}_{i, j}=\mathbf{x}_{i, j}^{\prime} \\ \frac{1}{\left\|\mathbf{x}-\mathbf{x}^{\prime}\right\|^{2}} & \text { if } \mathbf{x}_{i, j} \neq \mathbf{x}_{i, j}^{\prime}\end{cases}
$$""",
}
