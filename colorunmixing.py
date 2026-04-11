import torch
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64  

def mahalanobis(x, mu, sigma_inv):
    diff = x - mu.unsqueeze(0)  
    return (diff @ sigma_inv * diff).sum(dim=-1)

def ncg(f_fn, jac_fn, x, lamb, rho, maxiter=100, gtol=1e-6):

    P, D = x.shape
    x_cur = x.clone().clamp(0.0, 1.0)

    def grad_clip(x, grad):
        g = grad.clone()
        g[(x <= 0) & (g > 0)] = 0.0
        g[(x >= 1) & (g < 0)] = 0.0
        return g

    grad = jac_fn(x_cur, lamb, rho)
    g = grad_clip(x_cur, grad)
    d = -g.clone()

    for k in range(maxiter):
        g_norm = g.norm(dim=-1)  
        converged = g_norm < gtol
        gd = (g * d).sum(dim=-1)  
        bad_dir = gd >= 0
        d[bad_dir] = -g[bad_dir]

        alpha = torch.ones(P, 1, device=x.device, dtype=x.dtype)
        c1 = 1e-4
        fx = f_fn(x_cur, lamb, rho)  
        gd = (g * d).sum(dim=-1)     

        x_next = x_cur.clone()
        for _ in range(20):
            x_cand = (x_cur + alpha * d).clamp(0.0, 1.0)
            fx_cand = f_fn(x_cand, lamb, rho)
            accept = fx_cand <= fx + c1 * alpha.squeeze(-1) * gd

            x_next = torch.where(accept.unsqueeze(-1), x_cand, x_next)

            alpha = torch.where(accept.unsqueeze(-1), alpha, alpha * 0.5)
            if accept.all():
                break

        step_small = (x_next - x_cur).norm(dim=-1) < 1e-8
        x_cur = x_next

        if step_small.all():
            break

        g_new = grad_clip(x_cur, jac_fn(x_cur, lamb, rho))

        num = (g_new * (g_new - g)).sum(dim=-1)
        den = (g * g).sum(dim=-1) + 1e-16
        beta = (num / den).clamp(min=0.0)

        d = -g_new + beta.unsqueeze(-1) * d
        g = g_new

    return x_cur

#  SCU  
def solver_SCU(c, color_distr, sigmaa=10):

    P = c.shape[0]
    N = len(color_distr)

    mus = torch.stack([d['mu'] for d in color_distr])               
    sigma_invs = torch.stack([d['sigma_inv'] for d in color_distr])  

    diffs = c.unsqueeze(1) - mus.unsqueeze(0)         
    sinv_diffs = (sigma_invs @ diffs.unsqueeze(-1)).squeeze(-1)  
    dists = (diffs * sinv_diffs).sum(-1)  

    best_idx = dists.argmin(dim=1)  

    alphas0 = torch.zeros(P, N, device=DEVICE, dtype=DTYPE)
    colors0 = mus.unsqueeze(0).expand(P, -1, -1).clone()  

    alphas0[torch.arange(P, device=DEVICE), best_idx] = 1.0
    colors0[torch.arange(P, device=DEVICE), best_idx] = c  

    x0 = torch.cat([alphas0, colors0.reshape(P, N * 3)], dim=1)  

    def G(x):
        a = x[:, :N]                            
        u = x[:, N:].reshape(P, N, 3)           
        h_color = (a.unsqueeze(-1) * u).sum(1) - c  
        h_alpha = a.sum(1, keepdim=True) - 1.0           
        return torch.cat([h_color ** 2, h_alpha ** 2], dim=1)  

    def objective_energy(x):
        a = x[:, :N]                            
        u = x[:, N:].reshape(P, N, 3)           
        diff = u - mus.unsqueeze(0)             
        sinv_diff = (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1)  # (P, N, 3)
        dist = (diff * sinv_diff).sum(-1) 
        energy = (a * dist).sum(1)            

        s1 = a.sum(1)                           
        s2 = (a ** 2).sum(1) + 1e-6           
        sparsity = sigmaa * (s1 / s2 - 1.0)
        mask = (a ** 2).sum(1) > 1e-10
        sparsity = torch.where(mask, sparsity, torch.zeros_like(sparsity))

        return energy + sparsity

    def f(x, lamb, rho):
        Gx = G(x)
        return objective_energy(x) + (lamb * Gx).sum(1) + 0.5 * rho * (Gx ** 2).sum(1)

    def jacobian(x, lamb, rho):
        a = x[:, :N]                            
        u = x[:, N:].reshape(P, N, 3)            

        s1 = a.sum(1)                             
        s2 = (a ** 2).sum(1) + 1e-6             

        h_color = (a.unsqueeze(-1) * u).sum(1) - c  
        h_alpha = a.sum(1) - 1.0                          

        q = 2.0 * lamb[:, :3] * h_color + 2.0 * rho * (h_color ** 3)  
        p = 2.0 * lamb[:, 3:] * h_alpha.unsqueeze(-1) + 2.0 * rho * (h_alpha ** 3).unsqueeze(-1)  

        diff = u - mus.unsqueeze(0)                              
       
        sinv_diff = (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1) 

        grad_energy_u = 2.0 * a.unsqueeze(-1) * sinv_diff       
        dist = (diff * sinv_diff).sum(-1)  

        mask = (a ** 2).sum(1, keepdim=True) > 1e-10  
        grad_sp = sigmaa * (s2.unsqueeze(1) - 2.0 * a * s1.unsqueeze(1)) / (s2.unsqueeze(1) ** 2)
        grad_sp = torch.where(mask.expand_as(grad_sp), grad_sp, torch.zeros_like(grad_sp))

        grad_energy_a = dist + grad_sp  

        grad_G_a = (q.unsqueeze(1) * u).sum(-1) + p  
        grad_G_u = q.unsqueeze(1) * a.unsqueeze(-1)            

        grad_a = grad_energy_a + grad_G_a       
        grad_u = grad_energy_u + grad_G_u       

        return torch.cat([grad_a, grad_u.reshape(P, N * 3)], dim=1)

    x = x0
    lamb = torch.full((P, 4), 0.1, device=DEVICE, dtype=DTYPE)
    rho = 0.1
    beta = 10.0
    gamma = 0.25
    eps = 1e-6
    maxiter = 10

    G_prev_norm = G(x).norm(dim=1) 

    for k in range(maxiter):
        x_next = ncg(f, jacobian, x, lamb, rho, maxiter=10, gtol=1e-6)
        Gx = G(x_next)
        lamb = lamb + rho * Gx

        G_cur_norm = Gx.norm(dim=1)
        if (G_cur_norm > gamma * G_prev_norm).any():
            rho = beta * rho  

        delta = (x_next - x).norm(dim=1)
        if (delta < eps).all() and (G_cur_norm < eps).all():
            x = x_next
            break
        G_prev_norm = G_cur_norm
        x = x_next

    alphas = x[:, :N]
    colors = x[:, N:].reshape(P, N, 3)
    return alphas, colors

#  Matte regularization 
def matte_regu(im_np, alphas_np, rad=60):

    H, W, N = alphas_np.shape
    scale = np.sqrt((H * W) / 1_000_000.0)
    radius = max(1, int(rad * scale))

    guide = np.clip(im_np * 255.0, 0, 255).astype(np.uint8)
    eps = 0.001 * 255.0 * 255.0

    filtered = np.zeros_like(alphas_np, dtype=np.float32)
    print("Running matte regularization on CPU …")
    for i in range(N):
        alpha_u8 = np.clip(alphas_np[:, :, i] * 255.0, 0, 255).astype(np.uint8)
        f = cv.ximgproc.guidedFilter(guide, alpha_u8, radius, eps)
        filtered[:, :, i] = f.astype(np.float32) / 255.0

    filtered = np.clip(filtered, 0.0, 1.0)
    s = np.sum(filtered, axis=2, keepdims=True)
    s[s <= 1e-8] = 1e-8
    return filtered / s


#  Color refinement 
def color_refine(c, color_distr, alphashat, scu_colors):
    P = c.shape[0]
    N = len(color_distr)

    mus = torch.stack([d['mu'] for d in color_distr])                
    sigma_invs = torch.stack([d['sigma_inv'] for d in color_distr])   

    x0 = scu_colors.reshape(P, N * 3).clone()

    def G(x):
        u = x.reshape(P, N, 3)
        h = (alphashat.unsqueeze(-1) * u).sum(1) - c   
        return h ** 2   
    
    def objective_energy(x):
        u = x.reshape(P, N, 3)
        diff = u - mus.unsqueeze(0)
        dist = (diff * (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return (alphashat * dist).sum(1)

    def f(x, lamb, rho):
        Gx = G(x)
        return objective_energy(x) + (lamb * Gx).sum(1) + 0.5 * rho * (Gx ** 2).sum(1)

    def jacobian(x, lamb, rho):
        u = x.reshape(P, N, 3)
        diff = u - mus.unsqueeze(0)
        sinv_diff = (sigma_invs @ diff.unsqueeze(-1)).squeeze(-1)
        grad_energy_u = 2.0 * alphashat.unsqueeze(-1) * sinv_diff  

        h = (alphashat.unsqueeze(-1) * u).sum(1) - c
        q = 2.0 * lamb * h + 2.0 * rho * (h ** 3)   
        grad_aug_u = q.unsqueeze(1) * alphashat.unsqueeze(-1)   
        return (grad_energy_u + grad_aug_u).reshape(P, N * 3)

    x = x0
    lamb = torch.full((P, 3), 0.1, device=DEVICE, dtype=DTYPE)
    rho = 0.1
    beta = 10.0
    gamma = 0.25
    eps = 1e-6
    maxiter = 10
    G_prev_norm = G(x).norm(dim=1)

    for k in range(maxiter):
        x_next = ncg(f, jacobian, x, lamb, rho, maxiter=10, gtol=1e-6)
        Gx = G(x_next)
        lamb = lamb + rho * Gx
        G_cur_norm = Gx.norm(dim=1)
        if (G_cur_norm > gamma * G_prev_norm).any():
            rho = beta * rho
        delta = (x_next - x).norm(dim=1)
        if (delta < eps).all() and (G_cur_norm < eps).all():
            x = x_next
            break
        G_prev_norm = G_cur_norm
        x = x_next

    colors = x.reshape(P, N, 3)
    return alphashat, colors



#  Helper: convert color_distr from numpy to torch on device
def distr_to_torch(color_distr_np):
    out = []
    for d in color_distr_np:
        out.append({
            'mu': torch.tensor(d['mu'], device=DEVICE, dtype=DTYPE),
            'sigma_inv': torch.tensor(d['sigma_inv'], device=DEVICE, dtype=DTYPE)
        })
    return out

if __name__ == '__main__':
    print(f"Using device: {DEVICE}")

    im = cv.imread('data/img_02.jpg')
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    # Resize
    max_dim = 1024
    h, w = im.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        im = cv.resize(im, (int(w * scale), int(h * scale)), interpolation=cv.INTER_AREA)
    # im = cv.cvtColor(im, cv.COLOR_BGR2RGB)
    color_distr_np = np.load('./color_distribution.npy', allow_pickle=True)
    color_distr = distr_to_torch(color_distr_np)

    im_np = im.astype(np.float64) / 255.0
    H, W, _ = im_np.shape
    N = len(color_distr)
    P = H * W

    c = torch.tensor(im_np.reshape(P, 3), device=DEVICE, dtype=DTYPE)

    # Step 1: SCU (batched)
    print("Running SCU minimization on GPU …")
    alphas_flat, colors_flat = solver_SCU(c, color_distr)

    alphas_np = alphas_flat.cpu().numpy().reshape(H, W, N)
    colors_np = colors_flat.cpu().numpy().reshape(H, W, N, 3)

    # Step 2: matte regularization 
    matreg_np = matte_regu(im_np, alphas_np)

    # Step 3: color refinement on GPU (batched)
    matreg_flat = torch.tensor(matreg_np.reshape(P, N), device=DEVICE, dtype=DTYPE)
    colors_init = torch.tensor(colors_np.reshape(P, N, 3), device=DEVICE, dtype=DTYPE)

    print("Running color refinement on GPU …")
    final_a_flat, final_c_flat = color_refine(c, color_distr, matreg_flat, colors_init)

    final_as = final_a_flat.cpu().numpy().reshape(H, W, N)
    final_colors = final_c_flat.cpu().numpy().reshape(H, W, N, 3)

    # Save as npz

    np.savez_compressed(
        "layers_exact.npz",
        final_as=final_as.astype(np.float32),
        final_colors=final_colors.astype(np.float32),
    )

    # Plot

    fig, axes = plt.subplots(2, N, figsize=(N * 4, 8))
    for i in range(N):
        alpha = np.clip(final_as[:, :, i], 0.0, 1.0)
        color = np.clip(final_colors[:, :, i], 0.0, 1.0)
        axes[0, i].imshow(alpha, cmap='gray', vmin=0, vmax=1)
        axes[0, i].axis('off')
        axes[0, i].set_title(f"Layer {i} Alpha")
        contrib = np.clip(alpha[..., None] * color, 0.0, 1.0)
        axes[1, i].imshow(contrib)
        axes[1, i].axis('off')
        axes[1, i].set_title(f"Layer {i} Color")
    plt.tight_layout()
    plt.show()