import os
import monai.transforms as mt
import monai.data as md
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
def donkey_noise_like(x, discount=0.8):# lower discount seems better, like 0.8 [0.6 and 0.5 seems even better but unstable training]
    b, c, _w, _h,_d  = x.shape # EDIT: w and h get over-written, rename for a different variant!
    u = torch.nn.Upsample(size=(_w, _h,_d), mode='trilinear')
    noise = torch.randn_like(x)
    for i in range(10):
        r = 2 # Rather than always going 2x, 
        _w, _h,_d  = max(1, int(_w/(r**i))), max(1, int(_h/(r**i))), max(1, int(_d/(r**i)))
        #noise=2*noise#(patch)
        
        noise += u(torch.randn(b, c, _w, _h,_d).to(x)) * discount**i
        if _w==1 or _h==1 or _d==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance
def compare_3d(image_list,silent=False,fixed=True,title=None):
    def show_3d(image):
        _,_,h,w,d=image.shape
        plotting_image_0 = np.concatenate([image[0, 0, :, :, d//2].cpu(), np.flipud(image[0, 0, :, w//2, :].cpu())], axis=1)
        plotting_image_1 = np.concatenate([np.flipud(image[0, 0, int(0.3*h), :, :].cpu()), np.zeros((w, w))], axis=1)
        return np.concatenate([plotting_image_0, plotting_image_1],axis=0).T
    plt.figure(figsize=(20*len(image_list),20),dpi=50)
    result=np.concatenate([show_3d(image) for image in image_list], axis=1)
    ax = plt.gca() #note
    if fixed:
        im=plt.imshow(result, cmap="gray",vmin=0,vmax=1,)
    else:
        im=plt.imshow(result, cmap="gray",)
    sm = plt.cm.ScalarMappable(cmap="gray", norm=im.norm) #note
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01,) #note
    cbar.ax.tick_params(labelsize=min(30*len(image_list),60))
    plt.axis("off")
    if title!=None:
        plt.title(title,fontsize=min(30*len(image_list),60))
    if not silent:
        plt.show()
    return result
def compare_3d_jet(image_list,silent=False,fixed=True,title=None):
    def show_3d(image):
        _,_,h,w,d=image.shape
        plotting_image_0 = np.concatenate([image[0, 0, :, :, d//2].cpu(), np.flipud(image[0, 0, :, w//2, :].cpu())], axis=1)
        plotting_image_1 = np.concatenate([np.flipud(image[0, 0, int(0.3*h), :, :].cpu()), np.zeros((w, w))], axis=1)
        return np.concatenate([plotting_image_0, plotting_image_1],axis=0).T
    plt.figure(figsize=(20*len(image_list),20),dpi=50)
    result=np.concatenate([show_3d(image) for image in image_list], axis=1)
    ax = plt.gca() #note
    if fixed:
        im=plt.imshow(result, cmap="jet",vmin=-1,vmax=1,)
    else:
        im=plt.imshow(result, cmap="jet",)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=im.norm) #note
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01) #note
    cbar.ax.tick_params(labelsize=min(30*len(image_list),60))
    plt.axis("off")
    if title!=None:
        plt.title(title,fontsize=min(30*len(image_list),60))
    if not silent:
        plt.show()
    return result
def compare_3d_rwb(image_list,silent=False,fixed=True,title=None):
    def show_3d(image):
        _,_,h,w,d=image.shape
        plotting_image_0 = np.concatenate([image[0, 0, :, :, d//2].cpu(), np.flipud(image[0, 0, :, w//2, :].cpu())], axis=1)
        plotting_image_1 = np.concatenate([np.flipud(image[0, 0, int(0.3*h), :, :].cpu()), np.zeros((w, w))], axis=1)
        return np.concatenate([plotting_image_0, plotting_image_1],axis=0).T
    from matplotlib.colors import LinearSegmentedColormap
    plt.figure(figsize=(20*len(image_list),20),dpi=50)
    result=np.concatenate([show_3d(image) for image in image_list], axis=1)
    ax = plt.gca() #note
    cmap_name = 'blue_white_red'
    colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # Blue, White, Red
    n_bins = [3]  # Discretizes the interpolation into bins
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    if fixed:
        im=plt.imshow(result, cmap=cmap,vmin=-1,vmax=1,)
    else:
        vmax=max(result.max(),-1*result.min())
        vmin=-vmax
        im=plt.imshow(result, cmap=cmap,vmin=vmin,vmax=vmax,)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=im.norm) #note
    cbar = plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01) #note
    cbar.ax.tick_params(labelsize=min(30*len(image_list),60))
    plt.axis("off")
    if title!=None:
        plt.title(title,fontsize=min(30*len(image_list),60))
    if not silent:
        plt.show()
    return result
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def compare_combined_show(image_list_list, fixed=[True, True, True], min0=[True,True,True],title="."):
    def show_3d(image, cmap, vmin=None, vmax=None, ax=None):
        _, _, h, w, d = image.shape
        plotting_image_0 = np.concatenate([image[0, 0, :, :, d//2].cpu(), np.flipud(image[0, 0, :, w//2, :].cpu())], axis=1)
        plotting_image_1 = np.concatenate([np.flipud(image[0, 0, int(0.3*h), :, :].cpu()), np.zeros((w, w))], axis=1)
        result = np.concatenate([plotting_image_0, plotting_image_1], axis=0).T

        # Plot the result with the specified colormap and normalization
        im = ax.imshow(result, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.axis("off")
        return im

    # Unpack the image lists
    image_list_gray, image_list_jet, image_list_rwb = image_list_list
    all_images = image_list_gray + image_list_jet + image_list_rwb
    n = len(all_images)
    plt.figure(figsize=(20 * n, 20), dpi=50)

    # Custom colormap for RWB
    cmap_rwb = LinearSegmentedColormap.from_list('blue_white_red', [(0, 0, 1), (1, 1, 1), (1, 0, 0)], N=100)
    
    # Adjust fixed settings for each set
    fixed_settings = [fixed[0]] * len(image_list_gray) + [fixed[1]] * len(image_list_jet) + [fixed[2]] * len(image_list_rwb)

    for i, (image, fixed_setting) in enumerate(zip(all_images, fixed_settings)):
        ax = plt.subplot(1, n, i+1)
        if i < len(image_list_gray):
            N=len(image_list_gray)
            cmap = "gray"
            unfixed=(image.min(), image.max()) if min0[0] else (max(image.max(), -1 * image.min()), -max(image.max(), -1 * image.min()))
            vmin, vmax = (0, 1) if fixed_setting else unfixed
        elif i < len(image_list_gray) + len(image_list_jet):
            N=len(image_list_jet)
            cmap = "jet"
            unfixed=(image.min(), image.max()) if min0[1] else (max(image.max(), -1 * image.min()), -max(image.max(), -1 * image.min()))
            vmin, vmax = (-1, 1) if fixed_setting else unfixed
        else:
            N=len(image_list_rwb)
            cmap = cmap_rwb
            unfixed=(image.min(), image.max()) if min0[2] else (max(image.max(), -1 * image.min()), -max(image.max(), -1 * image.min()))
            vmin, vmax = (-1, 1) if fixed_setting else unfixed

        im = show_3d(image, cmap, vmin, vmax, ax=ax)
        # Create colorbar for each subplot
        cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cbar.ax.tick_params(labelsize=min(30*N,60))
    
    if title is not None:
        plt.suptitle(title, fontsize=20*(len(image_list_gray) + len(image_list_jet)+len(image_list_rwb)))
    plt.show()

from tqdm import tqdm
def visualize_stats_discribution(train_loader):
    # Wrap the original DataLoader to apply 'collect_stats' to each batch
    class StatsCollectingDataLoader:
        def __init__(self, dataloader):
            self.dataloader = dataloader

        def __iter__(self):
            return map(collect_stats, self.dataloader)

        def __len__(self):
            return len(self.dataloader)

    # Create an instance of the stats-collecting DataLoader
    stats_collecting_loader = StatsCollectingDataLoader(train_loader)

    # Now, when you iterate over 'stats_collecting_loader', it collects stats
    for batch in tqdm(stats_collecting_loader):
        pass  # Do your processing here

    # Check your collected stats
    import matplotlib.pyplot as plt

    # Assuming stats_collector is populated with the stats
    means = stats_collector['mean']
    stds = stats_collector['std']
    mins = stats_collector['min']
    maxs = stats_collector['max']
    means.sort()
    stds.sort()
    mins.sort()
    maxs.sort()
    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Image Statistics')

    # Mean
    axs[0, 0].plot(means, label='Mean')
    axs[0, 0].set_title('Mean of Images')
    axs[0, 0].set_xlabel('Batch')
    axs[0, 0].set_ylabel('Mean')
    axs[0, 0].legend()

    # Standard Deviation
    axs[0, 1].plot(stds, label='Standard Deviation', color='orange')
    axs[0, 1].set_title('Standard Deviation of Images')
    axs[0, 1].set_xlabel('Batch')
    axs[0, 1].set_ylabel('Std')
    axs[0, 1].legend()

    # Min
    axs[1, 0].plot(mins, label='Min', color='green')
    axs[1, 0].set_title('Min of Images')
    axs[1, 0].set_xlabel('Batch')
    axs[1, 0].set_ylabel('Min')
    axs[1, 0].legend()

    # Max
    axs[1, 1].plot(maxs, label='Max', color='red')
    axs[1, 1].set_title('Max of Images')
    axs[1, 1].set_xlabel('Batch')
    axs[1, 1].set_ylabel('Max')
    axs[1, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
def hist_match(source,target):
    oldshape=source.shape
    source=source.view(-1).cpu().numpy()
    target=target.view(-1).cpu().numpy()
    # transform source form [0,1] to [0,255] integer
    source=[int(i*255) for i in source]
    target=[int(i*255) for i in target]
    s_values,bin_idx,s_counts=np.unique(source,return_inverse=True,return_counts=True)
    t_values,t_counts=np.unique(target,return_counts=True)
    s_quantiles=np.cumsum(s_counts).astype(np.float64)
    s_quantiles/=s_quantiles[-1]
    t_quantiles=np.cumsum(t_counts).astype(np.float64)
    t_quantiles/=t_quantiles[-1]
    interp_t_values=np.interp(s_quantiles,t_quantiles,t_values)
    result=interp_t_values[bin_idx].reshape(oldshape)/255
    return torch.from_numpy(result).float().cuda()