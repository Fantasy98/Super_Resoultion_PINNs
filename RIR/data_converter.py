#%%
import numpy as np 
from scipy.io import loadmat

res_path = "results/munin/"
file_name = "u3tau4"

file_dir = res_path + file_name 
print(f"The data will be loaded from {file_dir}")
#%%
d = loadmat(file_dir+".mat")
print(f"INFO: The structure has been loaded, the keys in dict is {d.keys()}")

d = d["results"][0][0]
print(f"INFO: there are {len(d)} items in the results")

names = ["NMSE", "MAC", "freqMAC",
         "image", "image_recov", "image_under"]

# %%
np.savez_compressed( file_dir+".npz" ,
                    NMSE = d[0],
                    MAC = d[1],
                    freqMAC = d[2],
                    image = d[3],
                    image_recov = d[4],
                    image_under = d[5],)
print("INFO: The data have been saved as npz file!")