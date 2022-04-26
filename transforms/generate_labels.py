import numpy as np
from skimage.transform import resize, downscale_local_mean
from skimage.measure import block_reduce
import math
import matplotlib.pyplot as plt

def generate_heatmaps(landmarks, image_size, sigma, num_res_levels, lambda_scale=100, dtype=np.float32):
    heatmap_list = []

    num_heatmaps = len(landmarks)
    resizings = [[(num_heatmaps, int(image_size[0]/(2**x)), int(image_size[1]/(2**x)))] for x in range(num_res_levels)]
    resizing_factors = [[2**x, 2**x] for x in range(num_res_levels)]

    for size_f in resizing_factors:
        intermediate_heatmaps = []
        for idx, lm in enumerate(landmarks):
            lm = np.round(lm / size_f)
            downsample_size = [image_size[0] / size_f[0], image_size[1] / size_f[1]]
            down_sigma = sigma[idx]/ size_f[0]
            intermediate_heatmaps.append(gaussian_gen(lm, downsample_size, 1, down_sigma, dtype, lambda_scale))
        heatmap_list.append(np.array(intermediate_heatmaps))


    # fig, ax = plt.subplots(nrows=3, ncols=len(heatmap_list))

    # for i,map in enumerate(heatmap_list):
    #     print("this map shape, ", map.shape)
    #     for c in range(num_heatmaps):
    #         ax[c, i].imshow(map[c])
    
    # plt.show()
    return heatmap_list[::-1]

#generate Guassian with center on landmark. sx and sy are the std.
def gaussian_gen(landmark, resolution, step_size, std, dtype=np.float32, lambda_scale=100):

    sx = std
    sy = std

    x = resolution[0]//step_size
    y = resolution[1]//step_size

    mx = np.floor(landmark[0]/step_size)
    my = np.floor(landmark[1]/step_size)

   

    x = np.arange(x)
    y = np.arange(y)

    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D

    #define guassian 
    g = (1) / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.))) 

    #normalise between 0 and 1
    g *= 1.0/g.max() * lambda_scale
    # g[g<=0]=-1


    #transpose becasue x and y are the wrong way round
    # g = np.transpose(g)
    #add extra dimension for later calculations in Loss
    # g =  np.expand_dims(g, axis=0)


    return g

def get_downsampled_heatmaps(heatmaps, num_res_levels):

    print("og shape ", heatmaps.shape) #1,3,512,512

    original_size = heatmaps.shape[2:]
    num_heatmaps = heatmaps.shape[1]
    resizings = [[(num_heatmaps, int(original_size[0]/(2**x)), int(original_size[1]/(2**x)))] for x in range(num_res_levels)]
    resizing_factors = [[(1, 2**x, 2**x)] for x in range(num_res_levels)]

    print("resizings: ", resizings)
    print("resizing_factors: ", resizing_factors)

    z = [x[0] for x in resizings]
    print(z[0], " and ", z[-1])
    

    print("asd", np.squeeze(heatmaps).shape ,(3,256,256))
    y = resize(np.squeeze(heatmaps),resizings[-1][0], mode="edge", clip=True, anti_aliasing=False)
    print("ressss", y.shape)

    # heatmaps_to_return = [np.expand_dims(resize(np.squeeze(heatmaps), resizings[x][0], mode="edge", clip=True, anti_aliasing=False), axis=0) for x in range(len(resizings))]
    # heatmaps_to_return = [np.expand_dims(downscale_local_mean(np.squeeze(heatmaps), resizings[x][0]), axis=0) for x in range(len(resizings))]
    # heatmaps_to_return = [np.expand_dims(block_reduce(np.squeeze(heatmaps), resizings[x][0], func=np.mean), axis=0) for x in range(len(resizings))]
    heatmaps_to_return = [np.expand_dims(downscale_local_mean(np.squeeze(heatmaps), resizing_factors[x][0]), axis=0) for x in range(len(resizing_factors))]

    print("heatmaps to return shape@ ", len(heatmaps_to_return))
    fig, ax = plt.subplots(nrows=3, ncols=len(heatmaps_to_return))

    for i,map in enumerate(heatmaps_to_return):
        print("this map shape, ", map.shape)
        for c in range(num_heatmaps):
            ax[c, i].imshow(map[0,c])
    plt.show()



    