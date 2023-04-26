#%%
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import cmocean
from error import l2norm_err
from train_configs import BG_config
"""
Postprocessing of prediction data 

Return: 
    1. Loss vs Epoch
    2. U contour: Prediction, Reference, L1-norm
    3. V contour: Prediction, Reference, L1-norm
    4. P contour: Prediction, Reference, L1-norm
    5. Print l2-norm error 
"""
# cmp = sns.color_palette('cmo.tarn', as_cmap=True)
# plt.set_cmap(cmp)
plt.rc("font",family = "serif")
plt.rc("font",size = 14)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 14)
plt.rc("ytick",labelsize = 14)
#%%
act = BG_config.act
nn = BG_config.n_neural
nl = BG_config.n_layer
n_adam = BG_config.n_adam
data_name = f"BG_{nn}_{nl}_{act}_{n_adam}_4000_512"
data = np.load(f"pred/res_{data_name}.npz")
d = np.load('data/BurgerX=256T=50.npz')
x = d["T"]
y = d["X"]
u = d["U"].T

# %%
fig, ax = plt.subplots(1,1)
# ax.semilogy(data["hist"][:,0],label="Total")
ax.semilogy(data["hist"][:,1],label="Bc")
ax.semilogy(data["hist"][:,2],label="residual")
plt.legend()
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
plt.savefig(f"figs/{data_name}_Loss")
# %%
pred = data["pred"]
ref = data["ref"]
names = ["U","V","P"]
# %%

fig,axs = plt.subplots(2,2,figsize=(14,7),sharex=True,sharey=True)
c1 =axs[0,0].contourf(x,y,ref[:,:],
                       vmax = ref.max(), vmin = ref.min(),
                        levels = 50, cmap = "cmo.tarn")
c2= axs[0,1].contourf(x,y,pred[:,:], 
                      vmax = ref.max(), vmin = ref.min(),
                      levels = 50, cmap = "cmo.tarn")
c3 = axs[1,1].contourf(x,y,np.abs(ref[:,:]-pred[:,:]), levels = 50, cmap = "cmo.tarn")

cax1 = fig.add_axes([axs[0,1].get_position().x1+0.02,axs[0,1].get_position().y0+0.03,0.01,0.3])
cbar = fig.colorbar(c1, cax=cax1)
cbar.ax.locator_params(nbins = 5,tight=True)

cax2 = fig.add_axes([axs[1,1].get_position().x1+0.02,axs[1,1].get_position().y0+0.03,0.01,0.3])
cbar = fig.colorbar(c3, cax=cax2)
cbar.ax.locator_params(nbins = 5,tight=True)


axs[0,0].set_title("Reference",fontdict={"size":18})
axs[0,1].set_title("Prediction",fontdict={"size":18})
axs[1,1].set_title(r"$\epsilon$ = "+ r"$|u - \tilde{u}|$" ,fontdict={"size":18})
axs[1,0].axis("off")

axs[0,0].set_xlabel("t")
axs[1,1].set_xlabel("t")
axs[0,0].set_xticks([0,0.5,0.99])
axs[1,1].set_xticks([0,0.5,0.99])
axs[0,0].set_yticks([-1,0,1]) 
axs[1,1].set_yticks([-1,0,1]) 
# plt.tight_layout()
plt.savefig(f"figs/{data_name}_BG.pdf",dpi= 300)

# %%
