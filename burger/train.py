#%%
import numpy as np
from tensorflow.keras import models, layers, optimizers, activations
from PINN_BG import PINNs
from matplotlib import pyplot as plt
from time import time
from train_configs import BG_config
from error import l2norm_err
#%% 
#################
# DATA loading
#################
d = np.load('data/BurgerX=256T=50.npz')

x = d["T"]
y = d["X"]
u = d["U"].T

print(f"INFO: Data loaded, data size,temporal = {x.shape}, spatial = {y.shape}, value = {u.shape}")
# x = x - x.min()
# y = y - y.min()
ref = u
#%% 
#################
# Training Parameters
#################
print("INFO: Loading Training Configuration")
act = BG_config.act
nn = BG_config.n_neural
nl = BG_config.n_layer
n_adam = BG_config.n_adam
cp_step = BG_config.cp_step
bc_step = BG_config.bc_step
#%%
#################
# Training Data
#################
x_f = x.flatten()
y_f = y.flatten()
cp_step = 2000
cp_choose = np.random.randint(low=0, high=x_f.shape[0],size=cp_step)
# Collection points for supervised learning
cp = np.concatenate((x_f[cp_choose].reshape((-1, 1)), 
                     y_f[cp_choose].reshape((-1, 1))), axis = 1)
n_cp = len(cp)
print(n_cp)
print(cp.shape)
#%%
# Boundary points
ind_bc = np.zeros(x.shape, dtype = bool)
ind_bc[[0, 10 , -1],:] = True

print(ind_bc.shape)
x_bc = x[ind_bc]
print(x_bc.shape)
x_bc = x[ind_bc].flatten()
print(x_bc.shape)

y_bc = y[ind_bc].flatten()
print(y_bc.shape)
u_bc = u[ind_bc].flatten()
# v_bc = v[ind_bc].flatten()

bc = np.array([x_bc, y_bc, u_bc]).T
print(bc.shape)
#%%
ni = 2
nv = bc.shape[1] - ni
pp = 1

print(f"Input No = {ni}, Output No = {nv}")
# Randomly select half of Boundary points
# indx_bc = np.random.choice([False, True], len(bc), p=[1 - pp, pp])
# bc = bc[indx_bc]

n_bc = len(bc)
test_name = f'_{nn}_{nl}_{act}_{n_adam}_{n_cp}_{n_bc}'

print(f"There are {n_bc} boundary conditions is used!")
#%%
#################
# Compiling Model
#################

inp = layers.Input(shape = (ni,))
hl = inp
for i in range(nl):
    hl = layers.Dense(nn, activation = act)(hl)
out = layers.Dense(nv)(hl)

model = models.Model(inp, out)
print(model.summary())
lr = 1e-3
opt = optimizers.Adam(lr)
pinn = PINNs(model, opt, n_adam)

#################
# Training Process
#################
print(f"INFO: Start training case : {test_name}")
st_time = time()

hist = pinn.fit(bc, cp)

en_time = time()
comp_time = en_time - st_time
tot_time = comp_time - st_time
print(f"INFO: Training Finished, the computation time is {tot_time}")
# %%
#################
# Prediction
#################
cpp = np.array([x.flatten(), y.flatten()]).T
print(cpp.shape)
pred = pinn.predict(cpp)
print(pred.shape)
u_p = pred[:,1].reshape(u.shape)
# v_p = pred[:, 1].reshape(u.shape)
# p_p = pred[:, 2].reshape(u.shape)

# Shift the pressure to the reference level before calculating the error
# Becacuse we only have pressure gradients in N-S eqs but not pressure itself in BC
fig, ax = plt.subplots(1,1)
cb = ax.contourf(x,y,u_p)
plt.colorbar(cb)
plt.show()
pred = u_p
#%%
#################
# Save prediction and Model
#################
np.savez_compressed('pred/res_BG' + test_name, pred = pred, 
                                                ref = ref,
                                                 x = x, y = y, 
                                                 hist = hist,  
                                                 ct = tot_time)
model.save('models/model_BG' + test_name + '.h5')
print("INFO: Prediction and model have been saved!")