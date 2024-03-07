import bridgestan as bs
import cmdstanpy as csp
import sewar.full_ref as sfr
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import json
import os


def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray

def rmse(X, X_hat):
    return sfr.mse(np.float64(X), np.float64(X_hat))  # lower better

def ssim_unit(X, X_hat):
    return sfr.ssim(np.float64(X), np.float64(X_hat), MAX = 1)[0] # higher better


def msssim_unit(X, X_hat):
    return sfr.msssim(np.float64(X), np.float64(X_hat), MAX = 1).real # higher better

def psnr_unit(X, X_hat):
    return sfr.psnr(np.float64(X), np.float64(X_hat), MAX = 1)  # higher better

def logit(u):
    return np.log(u / (1 - u))

def fit_optimize(model, data, jacobian, seed):
    fit_mle = model.optimize(data, inits=0.5, seed=seed, jacobian=jacobian)
    X_hat_mle = fit_mle.stan_variable('X')
    return X_hat_mle, fit_mle

def mle(model, data, seed):
    return fit_optimize(model, data, jacobian=False, seed=seed)

def mode(model, data, seed):
    return fit_optimize(model, data, jacobian=True, seed=seed)

def vb(model, data, seed):
    fit_pf = model.pathfinder(data, inits=1,
                              psis_resample=True,
                              num_paths=2,draws=100, num_single_draws=1000,
                              max_lbfgs_iters=20,
                              show_console=True, refresh=1)
    draws_pf_X = fit_pf.stan_variables()['X']
    X_hat_pf = np.mean(draws_pf_X, axis=0)
    X_draw_pf = fit_pf.stan_variables()['X'][0, :, :] # 0 is just first draw
    return X_hat_pf, X_draw_pf, fit_pf

def mcmc(model, data, fit_pf, seed):
    inits_from_pf = fit_pf.create_inits(2)
    fit_sample = model.sample(data, inits=inits_from_pf, chains=2, parallel_chains=2,
                                iter_warmup=200, iter_sampling=200,
                                show_console=True, refresh=5,
                                show_progress=False,
                                seed = seed)
    draws_mcmc_X = fit_sample.stan_variable('X')
    X_hat_mcmc = np.mean(draws_mcmc_X, axis=0)
    X_draw_mcmc = draws_mcmc_X[10, :]



def sim_Y(source_png, N_p, R, r, sigma, seed):
    N, _ = np.shape(R)
    d = N
    M1 = 2 * N
    M2 = 2 * (2 * N + d)
    X_src = rgb2gray(mpimg.imread(source_png))
    X0R = np.concatenate([X_src, np.zeros((N, d)), R], axis=1)
    Y = np.abs(np.fft.fft2(X0R, s=(M1, M2)))**2
    rate = N_p / Y.mean()
    Y_tilde = sp.stats.poisson.rvs(rate * Y, random_state=seed)
    B_cal = np.ones((M1, M2), dtype=int)
    B_cal[M1 // 2 - r + 1: M1 // 2 + r,
              M2 // 2 - r + 1: M2 // 2 + r] = 0
    B_cal = np.fft.ifftshift(B_cal)
    Y_tilde *= B_cal
    data = {
        'N_p': N_p,
        'N': N,
        'R': R,
        'd': N,
        'M1': M1,
        'M2': M2,
        'Y': Y_tilde,
        'r': r,
        'sigma': 10,
        }
    return X_src, data

seed = 15789087
R = np.loadtxt('URA.csv', delimiter=",", dtype=int).tolist()
X_src, data = sim_Y('mimivirus.png', N_p=1, R = R, r=13, sigma=10, seed=567819)

model = csp.CmdStanModel(stan_file = 'holoml.stan')
X_hat_mle, fit_mle = mle(model, data, seed)
X_hat_map, fit_map = mode(model, data, seed)
X_hat_pf, X_draw_pf, fit_pf = vb(model, data, seed)
X_hat_mcmc, X_draw_mcmc, draws_mcmc_X = mcmc(model, data, fit_pf, seed)

bs_model = bs.StanModel('holoml.stan', json.dumps(data))
X_src_copy = X_src.copy()
X_src_copy[X_src_copy == 0] = 0.001  # avoids boundary on log odds scale
X_src_logit = np.array(np.float64(logit(X_src_copy)), order='C')
X_hat_map_logit = np.array(logit(X_hat_map), order='C')
X_hat_mle_logit = np.array(logit(X_hat_mle), order='C')
X_hat_pf_logit = np.array(logit(X_hat_pf), order='C')
X_draw_pf_logit = np.array(logit(X_draw_pf), order='C')

print(f" log posterior: Source: {bs_model.log_density(X_src_logit):10.1f}  MAP: {bs_model.log_density(X_hat_map_logit):10.1f}  MLE: {bs_model.log_density(X_hat_mle_logit):10.1f}  PF(mean): {bs_model.log_density(X_hat_pf_logit):10.1f}  PF(draw): {bs_model.log_density(X_draw_pf_logit):10.1f}  (higher better)")
print(f" SSIM: MAP: {ssim_unit(X_src, X_hat_map):6.3f}  MLE: {ssim_unit(X_src, X_hat_mle):6.3f}  PF(mean): {ssim_unit(X_src, X_hat_pf):6.3f}  PF(draw): {ssim_unit(X_src, X_draw_pf):6.3f}   MCMC(mean): {ssim_unit(X_src, X_hat_mcmc):6.3f}  MCMC(draw): {ssim_unit(X_src, X_draw_mcmc):6.3f}  (higher better)")

print(f"MSSIM: MAP: {msssim_unit(X_src, X_hat_map):6.3f}  MLE: {msssim_unit(X_src, X_hat_mle):6.3f}  PF(mean): {msssim_unit(X_src, X_hat_pf):6.3f}  PF(draw): {msssim_unit(X_src, X_draw_pf):6.3f}  MCMC(mean): {msssim_unit(X_src, X_hat_mcmc):6.3f}  MCMC(draw): {msssim_unit(X_src, X_draw_mcmc):6.3f}  (higher better)")

print(f" PSNR: MAP: {psnr_unit(X_src, X_hat_map):6.3f}  MLE: {psnr_unit(X_src, X_hat_mle):6.3f}  PF(mean): {psnr_unit(X_src, X_hat_pf):6.3f}  PF(draw): {psnr_unit(X_src, X_draw_pf):6.3f}  MCMC(mean): {psnr_unit(X_src, X_hat_mcmc):6.3f}  MCMC(draw): {psnr_unit(X_src, X_draw_mcmc):6.3f}  (higher better)")

print(f" RMSE: MAP: {rmse(X_src, X_hat_map):6.3f}  MLE: {rmse(X_src, X_hat_mle):6.3f}  PF(mean): {rmse(X_src, X_hat_pf):6.3f}  PF(draw): {rmse(X_src, X_draw_pf):6.3f}  MCMC(mean): {rmse(X_src, X_hat_mcmc):6.3f}  MCMC(draw): {rmse(X_src, X_draw_mcmc):6.3f}   (lower better)")

fig = plt.figure()
ax1 = fig.add_subplot(1, 7, 1, title="Source")
ax1.imshow(X_src, cmap="gray", vmin=0, vmax=1)
ax1.axis('off')
ax2 = fig.add_subplot(1, 7, 2, title="MAP")
ax2.imshow(X_hat_map, cmap="gray", vmin=0, vmax=1)
ax2.axis('off')
ax3 = fig.add_subplot(1, 7, 3, title="MLE")
ax3.imshow(X_hat_mle, cmap="gray", vmin=0, vmax=1)
ax3.axis('off')
ax4 = fig.add_subplot(1, 7, 4, title="PF (mean)")
ax4.imshow(X_hat_pf, cmap="gray", vmin=0, vmax=1)
ax4.axis('off')
ax5 = fig.add_subplot(1, 7, 5, title="PF (draw)")
ax5.imshow(X_draw_pf, cmap="gray", vmin=0, vmax=1)
ax5.axis('off')
ax6 = fig.add_subplot(1, 7, 6, title="MCMC (mean)")
ax6.imshow(X_hat_mcmc, cmap="gray", vmin=0, vmax=1)
ax6.axis('off')
ax7 = fig.add_subplot(1, 7, 7, title="MCMC (draw)")
ax7.imshow(X_draw_mcmc, cmap="gray", vmin=0, vmax=1)
ax7.axis('off')
plt.tight_layout()
plt.show()


test_img_dir = "img/usc-sipi"
test_img_paths = [os.path.join(root, file) for root, _, files in os.walk(test_img_dir) for file in files]
test_imgs_src = [rgb2gray(mpimg.imread(path)) for path in test_img_paths]
test_imgs_mles = [model.optimize({data, inits=0.5, seed=seed, jacobian=False).stan_variable('X') for 
