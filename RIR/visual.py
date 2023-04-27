#%%
import numpy as np 
import matplotlib.pyplot as plt
plt.rc("font",family = "serif")
plt.rc("font",size = 14)
plt.rc("axes",labelsize = 16, linewidth = 2)
plt.rc("legend",fontsize= 12, handletextpad = 0.3)
plt.rc("xtick",labelsize = 14)
plt.rc("ytick",labelsize = 14)
#%%
room = "munin"
res_path = "01_data/" +room+ "/"
file_name = "u3tau4"

file_dir = res_path + file_name 
print(f"The data will be loaded from {file_dir}")

d = np.load(file_dir+".npz")
print(f"INFO: File has been loaded")

print(f"INFO: Visualizing the ")
gt = d["image"]
p = d["image_recov"]
ug = d["image_under"]


T, M0 = gt.shape
dx = 3e-2 # mic array spacing 
fs = 11250 # sampling frequency 
t_idxs = np.arange(0,256) # Define space-time grid

xx,tt = np.meshgrid(dx * np.arange(0,M0), 1000*(t_idxs-1)/fs)

aux = gt[t_idxs,:]
zscale =[np.min(np.real(aux)),
         np.max(np.real(aux))]

fig,axs = plt.subplots(1,3,sharey=True, figsize = (7, 5),dpi = 150)
plt.set_cmap("gist_yarg")
axs[0].contourf(xx,tt,aux,
            vmin =zscale[0], vmax= zscale[1],
            levels = 200,
            )
axs[0].set_title("Reference")

axs[1].contourf(xx,tt,p[t_idxs,:],
            vmin =zscale[0], vmax= zscale[1],
            levels = 200,
            )
axs[1].set_title("Interpolated")


axs[2].contourf(xx,tt,ug[t_idxs,:],
            vmin =zscale[0], vmax= zscale[1],
            levels = 200,
            )
axs[2].set_title("Under-sampled")


[ax.set_aspect(0.5) for ax in axs]
[ax.set_ylim(0,15) for ax in axs]
[ax.set_xlabel(r"x"+" (m)") for ax in axs]
[ax.set_ylabel(r"t"+" (m)") for ax in axs]
plt.savefig("04_figs/" +room + "_" + file_name + '_Image')
# %%

print("INFO: Plot modal assurance criterion (MAC) and its frequency") 
fmac = d["freqMAC"].T
mac = d["MAC"]
print(f"Checking the size of MAC: {mac.shape}")
print(f"Checking the size of freqMAC: {fmac.shape}")

fig,ax = plt.subplots(1,1, figsize = (7,6))
ax.plot(fmac,mac, lw = 2, c = "gray")
ax.set_xlabel(r'$f$ (Hz)'); 
ax.set_ylabel(r'MAC');
plt.savefig("04_figs/" +room + "_" + file_name + '_MAC')