import bridgestan as bs
import cmdstanpy as csp
import json
import time
import numpy as np
import scipy as sp
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sewar.full_ref as sfr


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def rmse(X, X_hat):
    return sfr.mse(X, X_hat)  # lower better

def ssim_unit(X, X_hat):
    return sfr.ssim(np.float64(X), np.float64(X_hat), MAX = 1)[0] # higher better


def msssim_unit(X, X_hat):
    return sfr.msssim(np.float64(X), np.float64(X_hat), MAX = 1).real # higher better

def psnr_unit(X, X_hat):
    return sfr.psnr(np.float64(X), np.float64(X_hat), MAX = 1)  # higher better

def logit(u):
    return np.log(u / (1 - u))

seed = 5678

N = 256
d = N
N_p = 1
r = 13
M1 = 2 * N
M2 = 2 * (2 * N + d)
X_src = rgb2gray(mpimg.imread('mimivirus.png'))
R = np.loadtxt('URA.csv', delimiter=",", dtype=int)
X0R = np.concatenate([X_src, np.zeros((N,d)), R], axis=1)
Y = np.abs(np.fft.fft2(X0R, s=(M1, M2))) ** 2
rate = N_p / Y.mean()
Y_tilde = sp.stats.poisson.rvs(rate * Y, random_state=seed)
sigma = 1
data = {
    'N': N,
    'R': R.to_list(),
    'd': N,
    'M1': M1,
    'M2': M2,
    'Y_tilde': Y_tilde.to_list(),
    'r': r,
    'N_p': N_p,
    'sigma': sigma
}

model = csp.CmdStanModel(stan_file = 'holoml.stan')

fit_map = model.optimize(data, inits=1, seed=seed, jacobian=True)
X_hat_map = fit_map.stan_variable('X')

fit_mle = model.optimize(data, inits=1, seed=seed, jacobian=True)
X_hat_mle = fit_mle.stan_variable('X')

fit_sample = model.sample(data, inits=0.5, chains=1, iter_warmup=20, iter_sampling=20, refresh=1, show_console=True)

fit_pf = model.pathfinder(data, inits=1, show_console=True, refresh=1, max_lbfgs_iters=15, num_paths=2, draws=100, num_single_draws=1000)
draws_pf_X = fit_pf.stan_variables()['X']
X_hat_pf = np.mean(draws_pf_X, axis=0)
X_draw_pf = fit_pf.stan_variables()['X'][10, :, :]

start_time = time.perf_counter()
fit_nuts = model.sample(data, inits={'X': X_draw_pf}, chains=2, parallel_chains=2, iter_warmup=200, iter_sampling=200, show_console=True, refresh=1, show_progress=False)
end_time = time.perf_counter()
print(f"Total execution time: {end_time - start_time} seconds")

bs_model = bs.StanModel('holoml.stan', json.dumps(data))
X_src[X_src == 0] = 0.001
X_src_logit = np.array(np.float64(logit(X_src)), order='C')
X_hat_map_logit = np.array(logit(X_hat_map), order='C')
X_hat_mle_logit = np.array(logit(X_hat_mle), order='C')
X_hat_pf_logit = np.array(logit(X_hat_pf), order='C')
X_draw_pf_logit = np.array(logit(X_draw_pf), order='C')
print(f" log posterior: Source: {bs_model.log_density(X_src_logit):10.1f}  MAP: {bs_model.log_density(X_hat_map_logit):10.1f}  MLE: {bs_model.log_density(X_hat_mle_logit):10.1f}  PF(mean): {bs_model.log_density(X_hat_pf_logit):10.1f}  PF(draw): {bs_model.log_density(X_draw_pf_logit):10.1f}")


print(f" RMSE: MAP: {rmse(X_src, X_hat_map):6.3f}  MLE: {rmse(X_src, X_hat_mle):6.3f}  PF: {rmse(X_src, X_hat_pf):6.3f}  (lower better)")
print(f" SSIM: MAP: {ssim_unit(X_src, X_hat_map):6.3f}  MLE: {ssim_unit(X_src, X_hat_mle):6.3f}  PF: {ssim_unit(X_src, X_hat_pf):6.3f}  (higher better)")
print(f"MSSIM: MAP: {msssim_unit(X_src, X_hat_map):6.3f}  MLE: {msssim_unit(X_src, X_hat_mle):6.3f}  PF: {msssim_unit(X_src, X_hat_pf):6.3f}  (higher better)")
print(f" PSNR: MAP: {psnr_unit(X_src, X_hat_map):6.3f}  MLE: {psnr_unit(X_src, X_hat_mle):6.3f}  PF: {psnr_unit(X_src, X_hat_pf):6.3f}  (higher better)")

fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(25, 5))
axes[0].imshow(X_hat_map, cmap='gray', vmin=0, vmax=1)
axes[0].set_title("MAP")
axes[0].axis('off')
axes[1].imshow(X_hat_mle, cmap='gray', vmin=0, vmax=1)
axes[1].set_title("penalized MLE")
axes[1].axis('off')
axes[2].imshow(X_hat_pf, cmap='gray', vmin=0, vmax=1)
axes[2].set_title("Pathfinder (mean)")
axes[2].axis('off')
axes[3].imshow(X_draw_pf, cmap='gray', vmin=0, vmax=1)
axes[3].set_title("Pathfinder (draw)")
axes[3].axis('off')
axes[4].imshow(X_src, cmap='gray', vmin=0, vmax=1)
axes[4].set_title("Sample")
axes[4].axis('off')
plt.tight_layout()
plt.show()
