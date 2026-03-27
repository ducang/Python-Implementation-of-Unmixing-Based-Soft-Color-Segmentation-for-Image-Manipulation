import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Helper Func
def mahalanobis_dist(x,mu, sigma):

    sigma1 = sigma + (1e-4) * np.eye(sigma.shape[0])
    diff = x - mu

    return float(diff.T @ np.linalg.inv(sigma1) @ diff)

# Eqt 14
def project_to_planes(c, mu, n):

    ndot = float(np.dot(n,n))

    return c - (np.dot(c- mu, n)/ndot) * n

# Eqt 15
def estimate_alpha(c, u1, u2):

    alpha1 = np.linalg.norm(c-u2)/np.linalg.norm(u1-u2)
    alpha1 = np.clip(alpha1, 0.0, 1.0)
    alpha2 = 1.0 - alpha1

    return alpha1, alpha2

def projectMatrix_3D_norm_to_2D_plane(n):
    n = np.asarray(n, dtype=float)
    n_norm = n / np.linalg.norm(n)

    if abs(n_norm[0]) < 0.9:
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = np.array([0.0, 1.0, 0.0])
    
    u = np.cross(n_norm, r) / np.linalg.norm(np.cross(n_norm, r))
    v = np.cross(n_norm, u)

    P = np.stack([u,v], axis=1) # 3x2

    return P

# Approximate r^p
def approx_repScore_firstTerm(c, color_distr):
    single = []
    for i in color_distr:
        mu_i = i['mu']
        sigma_i = i['sigma']
        # single color mahalanobis dist to distribution
        distance_single = mahalanobis_dist(c,mu_i, sigma_i)
        single.append(distance_single)
    return min(single)

def approx_repScore_secondTerm(c, mu1, mu2, sigma1, sigma2):
    c = np.asarray(c, dtype=float)
    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    sigma1 = np.asarray(sigma1, dtype=float)
    sigma2 = np.asarray(sigma2, dtype=float)

    n = mu1 - mu2

    s1 = np.dot(c-mu1, n)
    s2 = np.dot(c-mu2, n)
    if s1*s2 > 0: return float('inf')

    uhat1 = project_to_planes(c,mu1,n)
    uhat2 = project_to_planes(c,mu2,n)

    a1, a2 = estimate_alpha(c,uhat1,uhat2)

    P = projectMatrix_3D_norm_to_2D_plane(n)

    mu1_2d = P.T @ mu1
    mu2_2d = P.T @ mu2

    sigma1_2d = P.T @ sigma1 @ P
    sigma2_2d = P.T @ sigma2 @ P

    uhat1_2d = P.T @ (uhat1)
    uhat2_2d = P.T @ (uhat2)

    du1 = mahalanobis_dist(uhat1_2d,mu1_2d,sigma1_2d)
    du2 = mahalanobis_dist(uhat2_2d,mu2_2d,sigma2_2d)

    # Eqt 16
    F = a1 * du1 + a2 * du2 
    return float(F)

def approx_RepScore(c, color_distr):

    if len(color_distr) == 0:
        return tau**2 + 1 # so no pixel get removed since rp > t^2 by 1

    if len(color_distr) == 1:
        mu = color_distr[0]['mu']
        sigma = color_distr[0]['sigma']
        return mahalanobis_dist(c, mu, sigma)

    single = approx_repScore_firstTerm(c, color_distr)

    best = float('inf')
    for i in range(len(color_distr)):
        for j in range(i+1, len(color_distr)):
            mu1 = color_distr[i]['mu']
            mu2 = color_distr[j]['mu']
            sigma1 = color_distr[i]['sigma']
            sigma2 = color_distr[j]['sigma']


            fij = approx_repScore_secondTerm(c,mu1,mu2,sigma1,sigma2)
            if fij < best:
                best = fij
    return min(single, best)

# Eqt 10
def get_Vote(gradMag, repScore):
    vp  = np.exp(-gradMag) * (1- np.exp(-repScore))
    return vp

# Eqt 11
def pick_seed(bin_indices, gradMag, best_bin, active_mask):
    H, W, _ = bin_indices.shape
    best_score = -np.inf
    best_p = None

    for y in range(H):
        for x in range(W):
            if not active_mask[y, x]:
                continue
                
            if not (bin_indices[y, x, 0] == best_bin[0] and 
                    bin_indices[y, x, 1] == best_bin[1] and 
                    bin_indices[y, x, 2] == best_bin[2]):
                continue

            y0, y1 = max(0, y - 10), min(H, y + 10)
            x0, x1 = max(0, x - 10), min(W, x + 10)

            window_bins = bin_indices[y0:y1, x0:x1]
            window_active = active_mask[y0:y1, x0:x1]

            S_p = np.sum(
                (window_bins[:, :, 0] == best_bin[0]) & 
                (window_bins[:, :, 1] == best_bin[1]) & 
                (window_bins[:, :, 2] == best_bin[2]) & 
                window_active
            )

            score = S_p * np.exp(-gradMag[y, x])

            if score > best_score:
                best_score = score
                best_p = (y, x)

    return best_p

def estimate_normal_parameter(im,seed, gf):
    H, W, _ = im.shape
    impulse = np.zeros((H, W), dtype=np.float32)
    impulse[seed[0], seed[1]] = 1.0

    weights_full = gf.filter(impulse).astype(np.float64)
    weights_full = np.clip(weights_full, 0.0, None)

    y, x = seed
    y0, y1 = max(0, y - 10), min(H, y + 10)
    x0, x1 = max(0, x - 10), min(W, x + 10)

    patch = im[y0:y1, x0:x1]                    # h x w x 3
    weights_patch = weights_full[y0:y1, x0:x1]  # h x w

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
if __name__ == "__main__":
    im = cv.imread('albdifdonut.png')
    im = cv.cvtColor(im, cv.COLOR_BGR2RGB).astype(np.float64) / 255.0
    # im = cv.resize(im, (im.shape[1]//3, im.shape[0]//3), interpolation=cv.INTER_AREA)
    H, W, _ = im.shape

    gx = cv.Sobel(im, cv.CV_64F, 1, 0)
    gy = cv.Sobel(im, cv.CV_64F, 0, 1)
    gradMag = np.sqrt(np.sum(gx**2 + gy**2, axis=2))

    tau = 5
    color_distr = []

    bin_indices = np.clip((im * 10).astype(np.int32), 0, 9)
    r = bin_indices[:, :, 0]
    g = bin_indices[:, :, 1]
    b = bin_indices[:, :, 2]
    gf = cv.ximgproc.createGuidedFilter(im.astype(np.float32), radius=10, eps=1e-5)
    MIN_VOTES = max(5, (H * W) // 2000)
    represented_mask = np.zeros((H, W), dtype=bool)

    while True:
        repScore = np.zeros((H, W), dtype=float)
        newly_represented = np.zeros((H, W), dtype=bool)
        # 1. Calculate representation score ONLY for unrepresented pixels
        for y in range(H):
            for x in range(W):
                if represented_mask[y, x]:
                    continue 
                
                score = approx_RepScore(im[y, x], color_distr)
                repScore[y, x] = score
                
                if score <= tau**2:
                    newly_represented[y, x] = True
        represented_mask |= newly_represented
        unrepresented_count = np.sum(~represented_mask)

        # 2. Calculate votes only for the unrepresented pixels
        active = ~represented_mask
        vp = np.zeros((H, W), dtype=float)
        vp[active] = get_Vote(gradMag[active], repScore[active])

        # 3. Divide into 10x10x10 
        bins = np.zeros((10, 10, 10), dtype=float)
        np.add.at(bins, tuple(bin_indices[active].T), vp[active])

        best_bin_idx = np.unravel_index(np.argmax(bins), bins.shape)
        
        if bins[best_bin_idx] < MIN_VOTES:
            print(f"Stopping: best bin has {bins[best_bin_idx]:.2f} votes "
                f"(threshold={MIN_VOTES}).")
            break
        seed = pick_seed(bin_indices, gradMag, best_bin_idx, active)
        if seed is None:
            print("Stopping: No valid seed found.")
            break

        mu, sigma = estimate_normal_parameter(im,seed,gf)
        color_distr.append({
            'mu': mu,
            'sigma': sigma,
            'sigma_inv': np.linalg.inv(sigma)
        })
        print(f"Added seed at {seed} with mean {mu}. Unrepresented left: {unrepresented_count}")


    if len(color_distr) == 0:
        print("No colors were extracted. Check your tau value or gradient calculations.")
    else:
        np.save("color_distribution.npy",color_distr)
        extracted_colors = [dist['mu'] for dist in color_distr]
        
        extracted_colors = np.clip(extracted_colors, 0.0, 1.0)

        palette = np.array([extracted_colors])
        
        # Plot it
        plt.figure(figsize=(10, 2))
        plt.imshow(palette)
        plt.axis('off')
        plt.title(f"Extracted Color Model ({len(extracted_colors)} Dominant Colors)")
        plt.show()