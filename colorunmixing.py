import cv2 as cv
import numpy as np
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import multiprocessing

def mahalanobis_dist(x,mu, sigma_inv):

    diff = x - mu

    return float(diff.T @ sigma_inv @ diff)

def ncg(f, jac, x, lamb, rho, maxiter=100, gtol=1e-6):
    def grad_clip(x, grad):
        g = grad.copy()

        # because step is -g
        g[(x <= 0) & (g > 0)] = 0.0
        g[(x >= 1) & (g < 0)] = 0.0

        return g
    
    def line_search(x, g, d):
        a = 1.0
        c1 = 1e-4
        fx = f(x, lamb, rho)
        gd = g @ d

        for _ in range(20):
            x_k = np.clip(x + a * d, 0.0, 1.0)
            if f(x_k, lamb, rho) <= fx + c1 * a * gd:
                return a, x_k
            a *= 0.5

        return a, x_k

    def polak_ribiere(g, g_next):
        numerator = g_next @ (g_next - g)
        denominator = (g @ g) + 1e-16
        beta = numerator / denominator
        return max(beta, 0.0)
    
    x_cur = np.clip(x.astype(np.float64), 0.0, 1.0)

    grad = jac(x_cur, lamb, rho)
    g = grad_clip(x_cur, grad)
    d = -g.copy()

    k = 0
    while k < maxiter:
        if np.linalg.norm(g) < gtol:
            break
        if (g @ d) >= 0:
            d = -g.copy()
        
        a, x_next = line_search(x_cur, g, d)

        if np.linalg.norm(x_next - x_cur) < 1e-8:
            x_cur = x_next
            break
        g_new = grad_clip(x_next, jac(x_next, lamb, rho))
        beta = polak_ribiere(g, g_new)

        d = -g_new + beta * d
        x_cur = x_next
        g = g_new
        k += 1

    return x_cur

def solver_SCU(c, color_distr, sigmaa = 10):
    n = len(color_distr)

    best_first_idx = -1
    min_dist = float('inf')
    # initialize layers
    for i, dist in enumerate(color_distr):
        d = mahalanobis_dist(c, dist['mu'], dist['sigma_inv'])
        if d < min_dist:
            min_dist = d
            best_first_idx = i

    x = np.zeros(4*n, dtype=np.float64)
    for i in range(n):
        if i == best_first_idx:
            x[i] = 1.0 # first alpha = 0
            x[n + i*3 : n + i*3 + 3] = c # set to pixel color
        else: # else The rest of the layers are initialized to zero alpha value and the mean of their distributions as layer colors
            x[i] = 0.0
            x[n + i*3 : n + i*3 + 3] = color_distr[i]['mu']

    # Jacobian
    def jacobian_scu(x,lamb,rho):
        alphas = x[:n]
        colors = x[n:].reshape((n,3))
        grad_alpha = np.zeros(n, dtype=np.float64)
        grad_color = np.zeros((n, 3), dtype=np.float64)

        s1 = np.sum(alphas)
        sum_sq_alphas = np.sum(alphas**2)
        s2 = sum_sq_alphas + 1e-6

        h = np.sum(alphas[:, np.newaxis] * colors, axis=0) - c
        t = np.sum(alphas, axis=0) - 1.0

        q = 2.0 * lamb[:3] * h + 2.0 * rho * (h**3)
        p = 2.0 * lamb[3] * t + 2 * rho * (t**3)

        for i in range(n):
            mu = color_distr[i]['mu']
            sigma_inv = color_distr[i]['sigma_inv']
            diff = colors[i] - mu

            dist = float(diff.T @ sigma_inv @ diff)
            grad_energy_ui = 2.0 * alphas[i] * (sigma_inv @ diff)
            
            if sum_sq_alphas > 1e-10:
                grad_sparsity_ai = sigmaa * (s2 - 2.0 * alphas[i] * s1) / (s2 ** 2)
            else:
                grad_sparsity_ai = 0.0
            grad_energy_ai = dist + grad_sparsity_ai
            
            grad_G_ai = q.T @ colors[i] + p
            grad_G_ui = q*alphas[i]

            grad_alpha[i] = grad_energy_ai + grad_G_ai
            grad_color[i] = grad_energy_ui + grad_G_ui
        return np.concatenate([grad_alpha, grad_color.reshape(-1)])
    # define variabls
    k = 0
    rho = 0.1
    lamb = np.array([0.1, 0.1, 0.1,0.1], dtype=np.float64)
    beta = 10
    gamma = 0.25
    eps = 1e-6
    maxiter = 10
    # obj func
    def objective_energy(x):
        alphas = x[:n]
        colors = x[n:].reshape((n,3))

        energy = 0.0
        for i  in range(n):
            mu = color_distr[i]['mu']
            sigma = color_distr[i]['sigma_inv']

            dist = mahalanobis_dist(colors[i], mu, sigma)
            energy += alphas[i] * dist
        
        sum_alphas = np.sum(alphas)
        sum_sq_alphas = np.sum(alphas**2)

        if sum_sq_alphas > 1e-10:
            sparsity = sigmaa * ((sum_alphas/(sum_sq_alphas + 1e-6))-1.0)
        else:
            sparsity = 0.0

        return energy + sparsity
    
    # constraint

    def G(x):
        alphas    = x[:n]
        colors    = x[n:].reshape(n, 3)

        h_color   = (alphas[:, None] * colors).sum(axis=0) - c  
        h_alpha   = alphas.sum() - 1.0                          

        G_u       = h_color ** 2      
        G_alpha   = h_alpha ** 2      

        return np.concatenate([G_u, [G_alpha]])
    
    def f(x,lamb,rho):
        return objective_energy(x) + lamb @ G(x) + 0.5 * rho * (np.linalg.norm(G(x))**2)

    while k < maxiter:

        # 1. Minimize
        x_next = ncg(f, jacobian_scu, x, lamb, rho, maxiter=10, gtol=1e-6)

        # 2. Update lambda
        lamb = lamb + rho * G(x_next)

        # 3. Update rho
        if np.linalg.norm(G(x_next)) > gamma*np.linalg.norm(G(x)):
            rho = beta * rho
        
        # 4. Stop condition
        if np.linalg.norm(x_next - x) < eps or np.linalg.norm(G(x_next)) < eps:
            x = x_next
            success = True
            break
        x = x_next
        k+=1
    colors = x[n:].reshape((n,3))
    alphas = x[:n]
    return alphas, colors



def matte_regu(im, alphas, rad=60):
    H, W, N = alphas.shape

    # scale radius by image size 
    scale = np.sqrt((H * W) / 1_000_000.0)
    radius = max(1, int(rad * scale))

    guide = np.clip(im * 255.0, 0, 255).astype(np.uint8)

    eps = 0.001 * 255.0 * 255.0

    filtered_alphas = np.zeros((H, W, N), dtype=np.float32)

    print("Running matte regularization")

    for i in range(N):
        alpha = np.clip(alphas[:, :, i] * 255.0, 0, 255).astype(np.uint8)

        # guided filter output 
        f = cv.ximgproc.guidedFilter(guide, alpha, radius, eps)

        # convert back to [0,1]
        filtered_alphas[:, :, i] = f.astype(np.float32) / 255.0

    filtered_alphas = np.clip(filtered_alphas, 0.0, 1.0)

    alpha_sum = np.sum(filtered_alphas, axis=2, keepdims=True)
    alpha_sum[alpha_sum <= 1e-8] = 1e-8

    norm_alphas = filtered_alphas / alpha_sum
    return norm_alphas

def color_refine(c, color_distr, alphashat, scu_layers):
    n = len(color_distr)

    x = scu_layers.flatten().astype(np.float64)

    def objective_energy(x):
        colors = x.reshape((n, 3))
        energy = 0.0
        for i in range(n):
            mu = color_distr[i]['mu']
            sigma = color_distr[i]['sigma_inv']
            dist = mahalanobis_dist(colors[i], mu, sigma)
            energy += alphashat[i] * dist 

        return energy
    # Jacobian
    def jacobian_cf(x, lamb, rho):
        colors = x.reshape((n,3))
        grad_color = np.zeros((n, 3), dtype=np.float64)

        h = np.sum(alphashat[:, np.newaxis] * colors, axis=0) - c

        q = 2.0 * lamb[:3] * h + 2.0 * rho * (h**3)

        for i in range(n):
            mu = color_distr[i]['mu']
            sigma_inv = color_distr[i]['sigma_inv']
            diff = colors[i] - mu

            grad_energy_ui = 2.0 * alphashat[i] * (sigma_inv @ diff)
            
            grad_aug_ui = q * alphashat[i] 
            grad_color[i] = grad_aug_ui + grad_energy_ui
        
        return grad_color.reshape(-1)
    

    # define variabls
    k = 0
    rho = 0.1
    lamb = np.array([0.1, 0.1, 0.1], dtype=np.float64)
    beta = 10
    gamma = 0.25
    eps = 1e-6
    maxiter = 10

    # constraint
    def G(x):
        colors    = x.reshape(n, 3)
        return (np.sum(alphashat[:, None] * colors, axis=0) - c)**2
    
    def f(x,lamb,rho):
        return objective_energy(x) + lamb @ G(x) + 0.5 * rho * (np.linalg.norm(G(x))**2)
    
    while k < maxiter:

        # 1. Minimize
        x_next = ncg(f, jacobian_cf, x, lamb, rho, maxiter=10, gtol=1e-6)

        # 2. Update lambda
        lamb = lamb + rho * G(x_next)

        # 3. Update rho
        if np.linalg.norm(G(x_next)) > gamma*np.linalg.norm(G(x)):
            rho = beta * rho
        
        # 4. Stop condition
        if np.linalg.norm(x_next - x) < eps or np.linalg.norm(G(x_next)) < eps:
            x = x_next
            break
        x = x_next
        k+=1
    colors = x.reshape((n,3))
    return alphashat, colors



def _scu_row(y, im_row, color_distr):
    W = im_row.shape[0]
    N = len(color_distr)
    row_alphas = np.zeros((W, N))
    row_colors = np.zeros((W, N, 3))
    for x in range(W):
        a, u = solver_SCU(im_row[x], color_distr)
        row_alphas[x] = a
        row_colors[x] = u
    return y, row_alphas, row_colors
 
 
def _refine_row(y, im_row, color_distr, matreg_row, colors_row):
    W = im_row.shape[0]
    N = len(color_distr)
    row_alphas = np.zeros((W, N))
    row_colors = np.zeros((W, N, 3))
    for x in range(W):
        a, u = color_refine(im_row[x], color_distr, matreg_row[x], colors_row[x])
        row_alphas[x] = a
        row_colors[x] = u
    return y, row_alphas, row_colors

def run_scu_parallel(im, color_distr, n_jobs=-1):

    H, W, _ = im.shape
    N       = len(color_distr)
    alphas  = np.zeros((H, W, N))
    colors  = np.zeros((H, W, N, 3))
 
    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=1)(delayed(_scu_row)(y, im[y], color_distr) for y in range(H))
    for y, row_a, row_u in results:
        alphas[y] = row_a
        colors[y] = row_u
 
    return alphas, colors
 
 
def run_refine_parallel(im, color_distr, matreg, colors, n_jobs=-1):
    H, W, _ = im.shape
    N       = len(color_distr)
    final_as     = np.zeros((H, W, N))
    final_colors = np.zeros((H, W, N, 3))

 
    results = Parallel(n_jobs=n_jobs, prefer='processes', verbose=1)(delayed(_refine_row)(y, im[y], color_distr, matreg[y], colors[y]) for y in range(H))
    for y, row_a, row_u in results:
        final_as[y]     = row_a
        final_colors[y] = row_u
 
    return final_as, final_colors

if __name__ == '__main__':
    im = cv.imread('albdifdonut.png') 
    im = cv.cvtColor(im,cv.COLOR_BGR2RGB)
    # im = cv.resize(im, (2260,1540), interpolation=cv.INTER_LINEAR)

    im = im / 255.0
    color_distr = np.load('./color_distribution.npy',allow_pickle=True)  
    H, W, _ = im.shape  
    N = len(color_distr)

    print("Running Soft Seg")
    alphas = np.zeros((H, W, N), dtype=np.float64)
    colors = np.zeros((H, W, N, 3), dtype=np.float64)

    print("Running SCU minimization")

    alphas, colors = run_scu_parallel(im, color_distr)

    matreg = matte_regu(im, alphas)

    final_as = np.zeros((H,W,N))
    final_colors = np.zeros((H,W,N,3))
    print("Running color refinement")

    final_as, final_colors = run_refine_parallel(im, color_distr, matreg, colors)
    np.savez_compressed(
        "layers_exact.npz",
        final_as=final_as.astype(np.float32),
        final_colors=final_colors.astype(np.float32)
    )
    for i in range(N):
        alpha = np.clip(final_as[:, :, i], 0.0, 1.0)
        color = np.clip(final_colors[:, :, i], 0.0, 1.0)
        rgba = np.dstack([color, alpha])                 
        rgba8 = (rgba * 255.0).round().astype(np.uint8)
        bgra8 = cv.cvtColor(rgba8, cv.COLOR_RGBA2BGRA)
        cv.imwrite(f"layer_{i:02d}.png", bgra8)
    fig, axes = plt.subplots(2, N, figsize=(N*4, 8))
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