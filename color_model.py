import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64

def mahalanobis_dist(x, mu, sigma):

    sigma1 = sigma + (1e-4) * np.eye(sigma.shape[0])
    diff = x - mu

    return float(diff.T @ np.linalg.inv(sigma1) @ diff)


# Eqt 14 helper
def projectMatrix_3D_norm_to_2D_plane(n):
  
    M = n.shape[0]
    n_norm = n / n.norm(dim=-1, keepdim=True)

    r = torch.zeros(M, 3, device=n.device, dtype=n.dtype)
    use_y = n_norm[:, 0].abs() >= 0.9
    r[~use_y, 0] = 1.0
    r[use_y, 1] = 1.0

    u = torch.cross(n_norm, r, dim=-1)
    u = u / u.norm(dim=-1, keepdim=True)
    v = torch.cross(n_norm, u, dim=-1)

    P = torch.stack([u, v], dim=-1)  
    return P

def approx_repScore_firstTerm(c, mus, sigma_invs_reg):
 
    diff = c.unsqueeze(1) - mus.unsqueeze(0)  # (P, K, 3)
    sinv_diff = (sigma_invs_reg @ diff.unsqueeze(-1)).squeeze(-1)  # (P, K, 3)
    single_dists = (diff * sinv_diff).sum(-1)  # (P, K)

    return single_dists.min(dim=1).values  


def approx_repScore_secondTerm(c, mus, sigmas, pair_i, pair_j):
 
    P = c.shape[0]
    M = pair_i.shape[0]

    mu1 = mus[pair_i]  # (M, 3)
    mu2 = mus[pair_j]  # (M, 3)
    sig1 = sigmas[pair_i]  # (M, 3, 3)
    sig2 = sigmas[pair_j]  # (M, 3, 3)

    n = mu1 - mu2  # (M, 3)
    ndot = (n * n).sum(-1)  # (M,)

    # Eqt 14: check which side of plane
    s1 = ((c.unsqueeze(1) - mu1.unsqueeze(0)) * n.unsqueeze(0)).sum(-1) 
    s2 = ((c.unsqueeze(1) - mu2.unsqueeze(0)) * n.unsqueeze(0)).sum(-1) 
    valid = s1 * s2 <= 0 

    # Eqt 14: project to planes
    t1 = s1 / ndot.unsqueeze(0) 
    t2 = s2 / ndot.unsqueeze(0) 
    uhat1 = c.unsqueeze(1) - t1.unsqueeze(-1) * n.unsqueeze(0)  
    uhat2 = c.unsqueeze(1) - t2.unsqueeze(-1) * n.unsqueeze(0)  

    # Eqt 15: estimate alpha
    a1 = (c.unsqueeze(1) - uhat2).norm(dim=-1) / ((uhat1 - uhat2).norm(dim=-1) + 1e-16) 
    a1 = a1.clamp(0.0, 1.0)
    a2 = 1.0 - a1

    # projection matrices (per pair, not per pixel)
    P_mat = projectMatrix_3D_norm_to_2D_plane(n)  
    P_matT = P_mat.transpose(-1, -2)  

    # project mus to 2D
    mu1_2d = (P_matT @ mu1.unsqueeze(-1)).squeeze(-1)  
    mu2_2d = (P_matT @ mu2.unsqueeze(-1)).squeeze(-1)  

    # project sigmas to 2D and invert  
    sig1_2d = P_matT @ sig1 @ P_mat + 1e-4 * torch.eye(2, device=DEVICE, dtype=DTYPE)  
    sig2_2d = P_matT @ sig2 @ P_mat + 1e-4 * torch.eye(2, device=DEVICE, dtype=DTYPE)  
    sig1_2d_inv = torch.linalg.inv(sig1_2d)  
    sig2_2d_inv = torch.linalg.inv(sig2_2d)  

    # project uhats to 2D: 
    uhat1_2d = (P_matT.unsqueeze(0) @ uhat1.unsqueeze(-1)).squeeze(-1)  
    uhat2_2d = (P_matT.unsqueeze(0) @ uhat2.unsqueeze(-1)).squeeze(-1)  

    # 2D Mahalanobis
    d1 = uhat1_2d - mu1_2d.unsqueeze(0)  
    du1 = (d1 * (sig1_2d_inv @ d1.unsqueeze(-1)).squeeze(-1)).sum(-1) 

    d2 = uhat2_2d - mu2_2d.unsqueeze(0)  
    du2 = (d2 * (sig2_2d_inv @ d2.unsqueeze(-1)).squeeze(-1)).sum(-1) 

    # Eqt 16
    F_val = a1 * du1 + a2 * du2 
    F_val[~valid] = float('inf')

    return F_val


def approx_RepScore(c, mus, sigmas, sigma_invs_reg, tau):
 
    P = c.shape[0]
    K = mus.shape[0]

    if K == 0:
        return torch.full((P,), tau**2 + 1, device=DEVICE, dtype=DTYPE)

    first_term = approx_repScore_firstTerm(c, mus, sigma_invs_reg)  

    if K == 1:
        return first_term
 
    pair_i = []
    pair_j = []
    for i in range(K):
        for j in range(i+1, K):
            pair_i.append(i)
            pair_j.append(j)
    pair_i = torch.tensor(pair_i, device=DEVICE, dtype=torch.long)
    pair_j = torch.tensor(pair_j, device=DEVICE, dtype=torch.long)

    F_all = approx_repScore_secondTerm(c, mus, sigmas, pair_i, pair_j)   
    second_term = F_all.min(dim=1).values   
    return torch.min(first_term, second_term)


# Eqt 10
def get_Vote(gradMag, repScore):
    vp = torch.exp(-gradMag) * (1 - torch.exp(-repScore))
    return vp

# Eqt 11
def pick_seed(bin_indices, gradMag, best_bin, active_mask):
 
    H, W = gradMag.shape

    match = (
        (bin_indices[:, :, 0] == best_bin[0]) &
        (bin_indices[:, :, 1] == best_bin[1]) &
        (bin_indices[:, :, 2] == best_bin[2]) &
        active_mask
    )  

    if not match.any():
        return None

    match_f = match.float().unsqueeze(0).unsqueeze(0)  
    kernel = torch.ones(1, 1, 21, 21, device=DEVICE, dtype=torch.float32)
    S_p = F.conv2d(match_f, kernel, padding=10).squeeze()  

    score = S_p * torch.exp(-gradMag.float())  
    score[~match] = -float('inf')

    flat_idx = score.argmax()
    y = int(flat_idx // W)
    x = int(flat_idx %  W)
    return (y, x)

def estimate_normal_parameter(im, seed, gf):
    H, W, _ = im.shape
    impulse = np.zeros((H, W), dtype=np.float32)
    impulse[seed[0], seed[1]] = 1.0

    weights_full = gf.filter(impulse).astype(np.float64)
    weights_full = np.clip(weights_full, 0.0, None)

    y, x = seed
    y0, y1 = max(0, y - 10), min(H, y + 10)
    x0, x1 = max(0, x - 10), min(W, x + 10)

    patch = im[y0:y1, x0:x1]   
    weights_patch = weights_full[y0:y1, x0:x1]   

    w_sum = weights_patch.sum()
    if w_sum <= 1e-12:
        print("Stopping: zero local guided-filter weights.")
        return None

    weights_patch = weights_patch / w_sum

    patch_flat = patch.reshape(-1, 3)
    w_flat = weights_patch.reshape(-1)

    mu = np.sum(patch_flat * w_flat[:, None], axis=0)

    diff = patch_flat - mu
    sigma = (diff.T * w_flat) @ diff
    sigma += 1e-4 * np.eye(3)

    return mu, sigma


#  Main

if __name__ == "__main__":
    print(f"Using device: {DEVICE}")

    im = cv.imread('data/img_02.jpg')
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float64)
    im = im / 255.0
    H, W, _ = im.shape

    gx = cv.Sobel(im, cv.CV_64F, 1, 0)
    gy = cv.Sobel(im, cv.CV_64F, 0, 1)
    gradMag_np = np.sqrt(np.sum(gx**2 + gy**2, axis=2))
    gradMag = torch.tensor(gradMag_np, device=DEVICE, dtype=DTYPE)  

    tau = 5
    color_distr = []

    bin_indices_np = np.clip((im * 10).astype(np.int32), 0, 9)
    bin_indices = torch.tensor(bin_indices_np, device=DEVICE, dtype=torch.long) 

    im_t = torch.tensor(im, device=DEVICE, dtype=DTYPE) 
    c_flat = im_t.reshape(-1, 3)  

    gf = cv.ximgproc.createGuidedFilter(im.astype(np.float32), radius=10, eps=1e-5)
    MIN_VOTES = max(5, (H * W) // 2000) # Change this to control how many colors are extracted (lower = more colors, higher = fewer colors)
    represented_mask = torch.zeros(H, W, device=DEVICE, dtype=torch.bool)
 
    mus_t = torch.empty(0, 3, device=DEVICE, dtype=DTYPE)
    sigmas_t = torch.empty(0, 3, 3, device=DEVICE, dtype=DTYPE)
    sigma_invs_reg_t = torch.empty(0, 3, 3, device=DEVICE, dtype=DTYPE)
 
    flat_bin = (bin_indices[:, :, 0] * 100 +
                bin_indices[:, :, 1] * 10  +
                bin_indices[:, :, 2])    
    while True:
        active = ~represented_mask  
        active_flat = active.reshape(-1) 

        # 1. representation score for unrepresented pixels
        c_active = c_flat[active_flat]  
        repScore_active = approx_RepScore(
            c_active, mus_t, sigmas_t, sigma_invs_reg_t, tau
        )

        # scatter back to full image
        repScore_flat = torch.zeros(H * W, device=DEVICE, dtype=DTYPE)
        repScore_flat[active_flat] = repScore_active
        repScore = repScore_flat.reshape(H, W)

        # mark newly represented
        newly_represented = (repScore <= tau**2) & active
        represented_mask |= newly_represented
        unrepresented_count = int((~represented_mask).sum().item())

        # 2. votes only for still-unrepresented pixels
        active = ~represented_mask
        vp = torch.zeros(H, W, device=DEVICE, dtype=DTYPE)
        vp[active] = get_Vote(gradMag[active], repScore[active])

        # 3. 10x10x10 binning
        bins_flat = torch.zeros(1000, device=DEVICE, dtype=DTYPE)
        bins_flat.scatter_add_(0, flat_bin[active].reshape(-1), vp[active].reshape(-1))
        bins = bins_flat.reshape(10, 10, 10)

        best_flat_idx = bins.argmax()
        best_bin = (
            int(best_flat_idx // 100),
            int((best_flat_idx % 100) // 10),
            int(best_flat_idx % 10),
        )

        if bins[best_bin].item() < MIN_VOTES:
            print(f"Stopping: best bin has {bins[best_bin].item():.2f} votes "
                  f"(threshold={MIN_VOTES}).")
            break

        seed = pick_seed(bin_indices, gradMag, best_bin, active)
        if seed is None:
            print("Stopping: No valid seed found.")
            break

        # 4. estimate parameters 
        result = estimate_normal_parameter(im, seed, gf)
        if result is None:
            break
        mu, sigma = result

        color_distr.append({
            'mu': mu,
            'sigma': sigma,
            'sigma_inv': np.linalg.inv(sigma)
        })

        # regularized inverse matching mahalanobis_dist (sigma + 1e-4*I then inv)
        sigma_reg = sigma + 1e-4 * np.eye(3)
        sigma_inv_reg = np.linalg.inv(sigma_reg)

        mus_t = torch.cat([
            mus_t,
            torch.tensor(mu, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        ])
        sigmas_t = torch.cat([
            sigmas_t,
            torch.tensor(sigma, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        ])
        sigma_invs_reg_t = torch.cat([
            sigma_invs_reg_t,
            torch.tensor(sigma_inv_reg, device=DEVICE, dtype=DTYPE).unsqueeze(0)
        ])

        print(f"Added seed at {seed} with mean {mu}. "
              f"Unrepresented left: {unrepresented_count}")

    if len(color_distr) == 0:
        print("No colors were extracted. Check your tau value or gradient calculations.")
    else:
        np.save("color_distribution.npy", color_distr)
        extracted_colors = [dist['mu'] for dist in color_distr]
        extracted_colors = np.clip(extracted_colors, 0.0, 1.0)

        palette = np.array([extracted_colors])

        plt.figure(figsize=(10, 2))
        plt.imshow(palette)
        plt.axis('off')
        plt.title(f"Extracted Color Model ({len(extracted_colors)} Dominant Colors)")
        plt.show()