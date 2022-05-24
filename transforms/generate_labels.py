import numpy as np
from skimage.transform import resize, downscale_local_mean
from skimage.measure import block_reduce
import math
import matplotlib.pyplot as plt
from utils.im_utils.heatmap_manipulation import get_coords
from visualisation import visualize_image_trans_coords, visualize_imageNcoords_cropped_imgNnormcoords
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import matplotlib.patches as patchesplt
import time

class LabelGenerator(ABC):
    """ Super class that defines some methods for generating landmark labels.
    """

    def __init__(self, full_res_size, network_input_size):
        self.full_res_size = full_res_size
        self.network_input_size = network_input_size

    @abstractmethod
    def generate_labels(self, landmarks, x_y_corner_patch, landmarks_in_indicator, input_size, hm_sigmas, num_res_supervisions, hm_lambda_scale):
        """ generates heatmaps for given landmarks of size input_size, using sigma hm_sigmas.
            Generates int(num_res_supervisions) heatmaps, each half the size as previous.
            The hms are scaled by float hm_lambda_scale

        Args:
            landmarks [int, int]: list of landmarks to gen heatmaps for
            input_size [int, int]: size of first heatmap
            hm_sigmas [float]: gaussian sigmas of heatmaps, 1 for each landmark
            num_res_supervisions int: number of heatmaps to generate, each half resolution of previous.
            hm_lambda_scale float: value to scale heatmaps by.

        """
    @abstractmethod
    def debug_sample(self, sample_dict, landmarks, image):
        """ Visually debug a sample. Provide logging and visualisation of the sample.

        Args:
            sample_dict (dict): dict of sample info returned by __get_item__ method
            landmarks [int, int]: list of the original landmarks, before any augmentation (same as those in sample_dict if no aug used).
            image [float, float]: original input image before augmentation  (same as those in sample_dict if no aug used).
        """

    @abstractmethod
    def stitch_heatmap(self, patch_predictions, stitching_info):
        '''
        Use model outputs from a patchified image to stitch together a full resolution heatmap
        
        '''

class UNetLabelGenerator(LabelGenerator):
    """ Generates target heatmaps for the U-Net network training scheme
    """
    def __init__(self):
        super(LabelGenerator, self).__init__()


    

    def generate_labels(self, landmarks, x_y_corner_patch, landmarks_in_indicator, image_size, sigmas, num_res_levels, lambda_scale=100, dtype=np.float32, to_tensor=True):
        heatmap_list = []

        resizing_factors = [[2**x, 2**x] for x in range(num_res_levels)]

        for size_f in resizing_factors:
            intermediate_heatmaps = []
            for idx, lm in enumerate(landmarks):

                lm = np.round(lm / size_f)
                downsample_size = [image_size[0] / size_f[0], image_size[1] / size_f[1]]
                down_sigma = sigmas[idx]/ size_f[0]
                if landmarks_in_indicator[idx] == 1:

                    intermediate_heatmaps.append(gaussian_gen(lm, downsample_size, 1, down_sigma, dtype, lambda_scale))
                else:
                    # print("downsample size: ", downsample_size, (int(downsample_size[0]), int(downsample_size[1])))
                    intermediate_heatmaps.append(np.zeros((int(downsample_size[0]), int(downsample_size[1]))))
            heatmap_list.append(np.array(intermediate_heatmaps))

        hm_list = heatmap_list[::-1]

        if to_tensor:
            all_seg_labels = []
            for maps in hm_list:
                all_seg_labels.append(torch.from_numpy(maps).float())

            hm_list = all_seg_labels


        return hm_list

    def stitch_heatmap(self, patch_predictions, stitching_info):
        '''
        Use model outputs from a patchified image to stitch together a full resolution heatmap
        
        '''
        

    

    def debug_sample(self, sample_dict, untrans_image, untrans_coords):
        """ Visually debug a sample. Provide logging and visualisation of the sample.

        """

        # print("before coords: ", landmarks)
        # print("og image sahpe: ", image.shape, "trans image shape", sample_dict["image"].shape, "trans targ coords: ", sample_dict["target_coords"])
        # print("len of hetamps ", len(sample_dict["label"]), " and shape: ", sample_dict["label"][-1].shape, " and hm exp shape ", np.expand_dims(sample_dict["label"][-1], axis=0).shape)
        landmarks_from_label = get_coords(torch.from_numpy(np.expand_dims(sample_dict["label"][-1], axis=0)))
        print("landmarks reverse engineered from heatmap label: ", landmarks_from_label)

        # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
        visualize_image_trans_coords(untrans_image[0], untrans_coords, sample_dict["image"][0] , sample_dict["target_coords"])


    def debug_crop(self, original_im, cropped_im, original_lms, normalized_lms, lms_indicators):
            """ Visually debug a cropped sample. Provide logging and visualisation of the sample.

            """

            print("before coords: ", original_lms)
            print("normalized lms: ", normalized_lms)
            print("landmark indicators", lms_indicators)
        

            # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
            visualize_imageNcoords_cropped_imgNnormcoords(original_im[0], cropped_im[0] ,original_lms, normalized_lms, lms_indicators)



class PHDNetLabelGenerator(LabelGenerator):
    """   Generates target heatmaps and displacements for the PHD-Net network training scheme
    """
    def __init__(self, maxpool_factor, full_heatmap_resolution, class_label_scheme, sample_grid_size ):
        super(LabelGenerator, self).__init__()
        # self.sampling_bias = sampling_bias
        self.maxpool_factor = maxpool_factor
        self.full_heatmap_resolution = full_heatmap_resolution
        self.class_label_scheme = class_label_scheme
        self.sample_grid_size = sample_grid_size


    def stitch_heatmap(self, patch_predictions, stitching_info):
        '''
        Use model outputs from a patchified image to stitch together a full resolution heatmap
        
        '''

    

    def generate_labels(self, landmarks, xy_patch_corner, landmarks_in_indicator, image_size, sigmas, num_res_levels, lambda_scale=100, dtype=np.float32, to_tensor=True):
        heatmap_list = []
    # landmark, xy_patch_corner, class_loss_scheme, grid_size, full_heatmap_resolution, maxpooling_factor, sigma, lambda_scale=100, debug=False):

        # sub_image, sub_patch_displacements, sub_class, sub_patch_cent, guassian_weights = gen_patch_displacements_heatmap(
        #     landmark, self.class_label_scheme, self.sample_grid_size, self.full_image_resolution, self.maxpool_factor, sigma, debug=True)
    
        intermediate_heatmaps = []
        for idx, lm in enumerate(landmarks):
            sigma = sigmas[idx]
            # intermediate_heatmaps.append(gaussian_gen(lm, downsample_size, 1, down_sigma, dtype, lambda_scale))
#  (landmark: Any, xy_patch_corner: Any, class_loss_scheme: Any, grid_size: Any, 
# full_heatmap_resolution: Any, maxpooling_factor: Any, sigma: Any, lambda_scale: int = 100, debug: bool = False)

            sub_patch_displacements, sub_class, guassian_weights = gen_patch_displacements_heatmap(
                lm, xy_patch_corner, self.class_label_scheme, self.sample_grid_size, self.full_heatmap_resolution, self.maxpool_factor, sigma, lambda_scale, debug=True)
            
            # heatmap_list.append(np.array(intermediate_heatmaps))

        hm_list = heatmap_list[::-1]

        if to_tensor:
            all_seg_labels = []
            for maps in hm_list:
                all_seg_labels.append(torch.from_numpy(maps).float())

            hm_list = all_seg_labels


        return hm_list

    def debug_sample(self, sample_dict, landmarks, image):
        """ Visually debug a sample. Provide logging and visualisation of the sample.

        """

        print("before coords: ", landmarks)
        print("og image sahpe: ", image.shape, "trans image shape", sample_dict["image"].shape, "trans targ coords: ", sample_dict["target_coords"])
        print("len of hetamps ", len(sample_dict["label"]), " and shape: ", sample_dict["label"][-1].shape, " and hm exp shape ", np.expand_dims(sample_dict["label"][-1], axis=0).shape)
        landmarks_from_label = get_coords(torch.from_numpy(np.expand_dims(sample_dict["label"][-1], axis=0)))
        print("landmarks reverse engineered from heatmap label: ", landmarks_from_label)

        # visualize_image_trans_target(np.squeeze(image), sample["image"][0], heatmaps[-1])
        visualize_image_trans_coords(image[0], sample_dict["image"][0] , sample_dict["target_coords"])


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
    # g[g<=0]=-1hm_lambda_scale


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


#  image, landmark, self.class_label_scheme, self.sample_grid_size, self.full_image_resolution, self.maxpool_factor, sigma, debug=True
def gen_patch_displacements_heatmap(landmark, xy_patch_corner, class_loss_scheme, grid_size, full_heatmap_resolution, maxpooling_factor, sigma, lambda_scale=100, debug=False):
    # Don't worry about sampling here, just generate the heatmaps.
    # 1) Generate displacements for the patch 
    # 2) generate full heatmap and slice the heatmap
        # Get the landmark on the full image by doing lm = normalized_lm + xy_patch 

    s = time.time()
   
   
    # need to find sub image grid now so 8x8 grid of this. so 64 patches of 16x16
    # need this grid so i can find center of each patch and if the landmark is in it

    # loop from top left as 0,0 down so like the convolutions go to match.
    # go from the randomly generated y to that + grid_size in steps
    # (8 for grid size 128)
    patches = [grid_size[0]//2**maxpooling_factor, grid_size[1]//2**maxpooling_factor]
    step_size = 2**maxpooling_factor

    x_y_displacements = np.zeros((patches))
    
    for x in range(0, grid_size[0], step_size):
        for y in range(0, grid_size[0], step_size):
        
            center_xy = [x+(step_size//2),y+(step_size//2)]

            # find log of displacements accounting for orientation

            distance_y = abs(landmark[1] - center_xy[1])
            if (landmark[1] > center_xy[1]):
                displace_y = math.log(distance_y, 2)
            elif (landmark[1] < center_xy[1]):
                displace_y = -math.log(distance_y, 2)
            else:
                displace_y = 0

            distance_x = abs(landmark[0] - center_xy[0])
            if (landmark[0] > center_xy[0]):
                    displace_x = math.log(distance_x, 2)
            elif (landmark[0] < center_xy[0]):
                displace_x = -math.log(distance_x, 2)
            else:
                displace_x = 0

            x_y_displacements[x,y] = [displace_x, displace_y]



    ###########Gaussian weights #############
    #Generate guassian weights for weighted loss
    full_resolution_lm = landmark + xy_patch_corner
    gaussian_weights_full = gaussian_gen(full_resolution_lm, full_heatmap_resolution, step_size, sigma, lambda_scale)


    x_block_start = int(np.floor(xy_patch_corner[0]/step_size))
    x_block_end = int(np.floor((xy_patch_corner[0] + grid_size)/step_size))
    y_block_start = int(np.floor(xy_patch_corner[1]/step_size))
    y_block_end = int(np.floor((xy_patch_corner[1]+ grid_size)/step_size))
    gaussian_weights =  gaussian_weights_full[:, x_block_start:x_block_end, y_block_start:y_block_end]
    # print("after guassian shape@ ", guassian_weights.shape)
  

    #Classification labels. Can be gaussian heatmap or binary.
    if class_loss_scheme == 'gaussian':
        sub_class = gaussian_weights
    else:
        raise NotImplementedError("only multi-branch labels are currently implemented. try with PHDNET.BRANCH_SCHEME as \"multi\"")
        sub_class = np.zeros([1,patches, patches])
        if landmark[0] >= x_rand and landmark[0] <= x_rand+grid_size and landmark[1] >= y_rand and landmark[1] <=y_rand+grid_size:
            pos_patches = []
        #  print("sample contains patch")
            #find positive patches
            patch_xy = landmark[:2].cpu().detach().numpy()- [x_rand, y_rand]
            patch_xy = (patch_xy/8).clip(0,patches-0.001) #clip when landmark is on far right border. cant be whole num due to border check

            #if landmark is on the border then patch before is also classed positive

            #case 1: X and Y not on border (X,Y)
            #Case 2: X is on the border, then (X,Y), (X-1, Y)
            #Case 3: Y is on the border (X,Y), (X, Y-1)
            #Case 4: X and Y is on the border  (X,Y), (X-1, Y),  (X-1,Y-1), (X, Y-1)

            #Also need to worry if border is at 0 edge or far edge.
            #if landmark is on the left of patch zero, just use patch 0. solved by additional clause below.
            #if landmark is on right of patch 16 (idx 15) then just use patch idx 15. Solved by above clip.
            border_x = float(patch_xy[0]).is_integer() and  patch_xy[0] != 0
            border_y = float(patch_xy[1]).is_integer() and  patch_xy[1] != 0

            x_rounded = np.floor(patch_xy[0]).astype('int')
            y_rounded = np.floor(patch_xy[1]).astype('int')

            if not border_x and not border_y:
                pos_patches.append([x_rounded,y_rounded ])

            elif border_x and not border_y:
                pos_patches.append([x_rounded,y_rounded ])
                pos_patches.append([x_rounded-1,y_rounded ])

            elif not border_x and border_y:
                pos_patches.append([x_rounded,y_rounded ])
                pos_patches.append([x_rounded,y_rounded-1 ])
            else: # border_x and border_y
                pos_patches.append([x_rounded,y_rounded ])
                pos_patches.append([x_rounded-1,y_rounded ])
                pos_patches.append([x_rounded,y_rounded-1 ])
                pos_patches.append([x_rounded-1,y_rounded-1 ])
            
            pos_patches = np.array(pos_patches)

            try:
                sub_class[:,pos_patches[:,1],pos_patches[:,0]] = 1
            except:
                print("error")
                print("landmark:", landmark)
                print("x range: ", x_rand, x_rand+grid_size, "y range: ", y_rand, y_rand+grid_size)
                print("patch xy", patch_xy)
                print("pose patches", pos_patches)
                exit() 


    # ####################### DEBUGGING VISUALISATION ##############


    if debug:
        print("normalized landmark: ", landmark)
        print("reconstructed full landmark: ", full_resolution_lm)
        print("full gauss shape and sliced gauss shape ", gaussian_weights_full.shape, gaussian_weights.shape)


        fig, ax = plt.subplots(nrows=3, ncols=1)
        ax[0].imshow(gaussian_weights_full[0])
        ax[1].imshow(gaussian_weights[0])
        ax[2].imshow(gaussian_weights[0])

        rect0 = patchesplt.Rectangle(( full_resolution_lm[0], full_resolution_lm[1]) ,6,6,linewidth=2,edgecolor='g',facecolor='none') 
        ax[0].add_patch(rect0)

        rect1 = patchesplt.Rectangle(( landmark[0], landmark[1]) ,6,6,linewidth=2,edgecolor='m',facecolor='none') 
        ax[1].add_patch(rect1)

        rect2 = patchesplt.Rectangle(( landmark[0], landmark[1]) ,6,6,linewidth=2,edgecolor='m',facecolor='none') 
        ax[2].add_patch(rect2)


        for x in range(0, grid_size[0], step_size):
            for y in range(0, grid_size[0], step_size):
        
                center_xy = [x+(step_size//2),y+(step_size//2)]
                x_disp = np.sign(x_y_displacements[x,y,0]) * (2**(abs(x_y_displacements[x,y,0])))
                y_disp = np.sign(x_y_displacements[x,y,1]) * (2**(abs(x_y_displacements[x,y,1])))
            #  print(x_cent, y_cent, x_disp, y_disp)
                ax[2].arrow(center_xy[0], center_xy[1], x_disp, y_disp)



        plt.show()

    # # ############################# end of DEBUGGING VISUALISATION #################
    

    # e = time.time()
    return x_y_displacements, sub_class, gaussian_weights

    