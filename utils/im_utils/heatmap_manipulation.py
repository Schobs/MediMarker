import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patchesplt
import time
import scipy.optimize as opt
# from inference.fit
def twoD_Gaussian(xdata_tuple, amplitude, xo, yo, sigma_x, sigma_y, theta, offset):
    (x, y) = xdata_tuple                                                        
    xo = float(xo)
    yo = float(yo)    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                            + c*((y-yo)**2)))
    return g.ravel()

def get_coords_fit_gauss(images, predicted_coords_all, visualize=False):
    all_fitted_dicts = []
    all_hm_maxes = []
    all_final_coords = []
    for hm_stack_idx, hm_Stack in enumerate(images):
        fitted_dicts = []
        hm_maxes = []
        final_coords = []

        for hm_idx, hm in enumerate(hm_Stack):
            predicted_heatmap = hm.detach().cpu().numpy()
            predicted_coords = predicted_coords_all[hm_stack_idx][hm_idx].detach().cpu().numpy()

            # print("predicted_heatmap shape: ", predicted_heatmap.shape)
            # print("predicted_coords shape: ", predicted_coords.shape)
            # Create x and y indices
            x = np.linspace(0, predicted_heatmap.shape[1], predicted_heatmap.shape[1])
            y = np.linspace(0, predicted_heatmap.shape[0], predicted_heatmap.shape[0])
            x, y = np.meshgrid(x, y)

            
            # initial_guess = (3,predicted_heatmap.shape[0]/2,predicted_heatmap.shape[1]/2,3,3,0,0)
            initial_guess = (predicted_heatmap.max(),predicted_coords[0], predicted_coords[1],6,6,0,0)
            bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.pi,-np.inf] , [np.inf,np.inf,np.inf,np.inf,np.inf, np.pi,np.inf])
            # bounds = ([-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf] , [np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])

            popt, pcov = opt.curve_fit(twoD_Gaussian, (x, y), predicted_heatmap.ravel(), p0=initial_guess, bounds=bounds)
            data_fitted = twoD_Gaussian((x, y), *popt)

            sigma_prod= popt[3] * popt[4]
            sigma_ratio = max(popt[3], popt[4]) /min(popt[3], popt[4])

            # print("prod and ratio: ", sigma_prod, sigma_ratio)
            if visualize:
                fig, ax = plt.subplots(1, 3)

                data_fitted_notheta = twoD_Gaussian((x, y), *[popt[0],popt[1],popt[2],popt[3],popt[4],0,popt[6]])
                ax[0].imshow(predicted_heatmap, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
                ax[0].contour(x, y, data_fitted_notheta.reshape(predicted_heatmap.shape[0], predicted_heatmap.shape[1]), 8, colors='w')

                data_fitted_halftheta = twoD_Gaussian((x, y), *[popt[0],popt[1],popt[2],popt[3],popt[4],popt[5]/2,popt[6]])
                ax[1].imshow(predicted_heatmap, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
                ax[1].contour(x, y, data_fitted_halftheta.reshape(predicted_heatmap.shape[0], predicted_heatmap.shape[1]), 8, colors='w')

                ax[2].imshow(predicted_heatmap, cmap=plt.cm.jet, origin='lower', extent=(x.min(), x.max(), y.min(), y.max()))
                ax[2].contour(x, y, data_fitted.reshape(predicted_heatmap.shape[0], predicted_heatmap.shape[1]), 8, colors='w')

                print("Pred coords: ", predicted_coords)
                print("Covariance: Amplitude %s, coords: (%s,%s) sigmaXY (%s, %s), theta %s, offset %s " % (popt[0],popt[1],popt[2],popt[3],popt[4],np.degrees(popt[5]),popt[6]))
                plt.show()
                plt.close()
            del predicted_heatmap, data_fitted
            print("initial guess: ", initial_guess, "and fitted: ",  [popt[1],popt[2]])
            final_coords.append([popt[1], popt[2]])
            hm_maxes.append(popt[0])
            fitted_dicts.append({"amplitude": popt[0], "mean": [popt[1],popt[2]],"sigma": [popt[3],popt[4]],"theta": np.degrees(popt[5]), "offset": popt[6]})
        all_fitted_dicts.append(fitted_dicts)
        all_final_coords.append(final_coords)
        all_hm_maxes.append(hm_maxes)
    return all_final_coords, all_hm_maxes, all_fitted_dicts


def get_coords(images):

    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    # print("score map shape:", images.shape)  #  torch.Size([1, 3, 64, 64])
    assert images.dim() == 4, 'Score maps should be 4-dim not shape %s' % images.shape
    # print("score map shape:", images.shape)  #  torch.Size([1, 3, 64, 64])
    maxval, idx = torch.max(images.view(images.size(0), images.size(1), -1), 2)
    # print("maxval & idx is: ", maxval, idx) #  tensor([[0.0394, 0.0333, 0.0242]]) tensor([[2207, 1695, 2071]])


    maxval = maxval.view(images.size(0), images.size(1), 1)
    # print("maxval again is: ", maxval) # tensor([[[0.0394],[0.0333],[0.0242]]])


    idx = idx.view(images.size(0), images.size(1), 1) +1
    # print("idx again is: ", idx) #([[[2208],[1696],[2072]]])

    preds = idx.repeat(1, 1, 2).float() 
    # print("preds shape and preds is", preds.shape) # torch.Size([1, 3, 2])
    # print(preds) # tensor([[[2208., 2208.], [1696., 1696.],[2072., 2072.]]])

    #ok so i think the index is the actual value from the flattened array,
    #this is taking that value and finding the row and column of that.
    preds[:,:,0] = (preds[:,:,0] - 1) % images.size(3) 
    preds[:,:,1] = torch.floor((preds[:,:,1] - 1) / images.size(3))

    # print("preds again with first and second 3rd dimension")
    # print(preds[:,:,0]) #tensor([[32., 32., 24.]])
    # print(preds[:,:,1]) # tensor([[35., 27., 33.]])


    

    pred_mask = maxval.gt(0).repeat(1, 1, 2).float()
    # print("preds mask shape n value: ", pred_mask.shape, pred_mask) # torch.Size([1, 3, 2])
    preds *= pred_mask
    # print("preds multiplied by mask: ", preds) #tensor([[[32., 35.], [32., 27.], [24., 33.]]])
    preds = torch.round(preds)
    return preds, maxval



def candidate_smoothing(output, full_resolution, maxpool_factor,  log_displacement_bool=True, smooth_factor=4, return_cropped_im=False, debug=False, save_fig=False):
    import transforms.generate_labels as gl

	# print("output and input andshape, ", output[0].shape, output[1].shape, input.shape)
	# print("fr, ", full_resolution)

    predicted_heatmap = output[0]
    predicted_disps = output[1]

    step_size = 2**maxpool_factor
    # print("predicted_heatmap shape ", predicted_heatmap.shape)
    # print("predicted_disps shape ", predicted_disps.shape)
    # print("full resolution ", full_resolution)
    
    smoothed_heatmaps = []
    for lm in range(output[0].shape[0]):
        upscaled_hm =  np.broadcast_to(predicted_heatmap[lm][:,None,:,None], (predicted_heatmap[lm].shape[0], step_size, predicted_heatmap[lm].shape[1], step_size)).reshape(full_resolution)



        all_locs = []
        for x_idx, x in enumerate(range(0, full_resolution[0], step_size)):
            for y_idx, y in enumerate(range(0, full_resolution[1], step_size)):

                # center_xy = [x+(step_size//2),y+(step_size//2)]
                # #REMEMBER TO MINUS 1 TO REVERSE THE LOG SHIFT WHEN CALCULATING THE LABELS!
                # x_disp = np.sign(predicted_disps[lm,0,x_idx,y_idx]) * (2**(abs(predicted_disps[lm,0,x_idx,y_idx]))-1)
                # y_disp = np.sign(predicted_disps[lm,1, x_idx,y_idx]) * (2**(abs(predicted_disps[lm,1,x_idx,y_idx]))-1)
                # loc = [center_xy[0]+x_disp, center_xy[1]+y_disp]

                center_xy = [x+(step_size//2),y+(step_size//2)]
                #REMEMBER TO MINUS 1 TO REVERSE THE LOG SHIFT WHEN CALCULATING THE LABELS!
                if log_displacement_bool:
                    x_disp = np.sign(predicted_disps[lm,0,x_idx,y_idx]) * (2**(abs(predicted_disps[lm,0,x_idx,y_idx]))-1)
                    y_disp = np.sign(predicted_disps[lm,1, x_idx,y_idx]) * (2**(abs(predicted_disps[lm,1,x_idx,y_idx]))-1)
                else:
                    x_disp = predicted_disps[lm,0,x_idx,y_idx]
                    y_disp = predicted_disps[lm,1, x_idx,y_idx]

                loc = [center_xy[0]+x_disp, center_xy[1]+y_disp]
                
                # loc = [center_xy[1]+y_disp, center_xy[0]+x_disp]

                if 0 <= loc[0] <=  full_resolution[0] and 0 <= loc[1] <= full_resolution[1]:
                    all_locs.append(loc)

        s= time.time()
        vote_heatmap = gl.gaussian_gen_fast(all_locs, full_resolution, sigma=1)

        # vote_heatmap = gl.gaussian_gen_alternate(all_locs, full_resolution, sigma=1)
        # vote_heatmap = np.zeros(full_resolution)
        # for vote in all_locs:
        #     vote_heatmap += gl.gaussian_gen(vote,full_resolution, 1, 1, lambda_scale=1)
            
        # print("time to gen all candidate points: ", time.time()-s)
        # print("vote heatmap shape max: ", vote_heatmap.shape, vote_heatmap.max())

        smoothed_heatmap = vote_heatmap * upscaled_hm
        smoothed_heatmaps.append(torch.tensor(smoothed_heatmap))

        if debug:

            # 1. show the upscaled heatmap with the extracted coords from hm branch only
            # 2. show displacements as arrows
            # 3. show displacement map of blobs
            # 4. show smoothed heatmap 

            #1

            fig, ax = plt.subplots(nrows=2, ncols=2)
            coords_from_uhm, arg_max_uhm = get_coords(torch.tensor(np.expand_dims(np.expand_dims(upscaled_hm, axis=0), axis=0)))
            coords_from_uhm =coords_from_uhm.detach().cpu().numpy()[0,0]

            print("get_coords from upscaled_hm: ", coords_from_uhm)

            ax[0,0].imshow(upscaled_hm)
            rect4 = patchesplt.Rectangle(( (coords_from_uhm[0]), (coords_from_uhm[1])) ,6,6,linewidth=2,edgecolor='m',facecolor='none') 
            ax[0,0].add_patch(rect4)

            #2
            ax[0,1].imshow(upscaled_hm)
            for x_idx, x in enumerate(range(0, full_resolution[0], step_size)):
                for y_idx, y in enumerate(range(0, full_resolution[1], step_size)):

                    center_xy = [x+(step_size//2),y+(step_size//2)]
                    #REMEMBER TO ADD 1 TO REVERSE THE LOG SHIFT WHEN CALCULATING THE LABELS!
                    if log_displacement_bool:
                        x_disp = np.sign(predicted_disps[lm,0,x_idx,y_idx]) * (2**(abs(predicted_disps[lm,0,x_idx,y_idx]))-1)
                        y_disp = np.sign(predicted_disps[lm,1, x_idx,y_idx]) * (2**(abs(predicted_disps[lm,1,x_idx,y_idx]))-1)
                    else:
                        x_disp = predicted_disps[lm,0,x_idx,y_idx]
                        y_disp = predicted_disps[lm,1, x_idx,y_idx]

                    ax[0,1].arrow(center_xy[0], center_xy[1], x_disp, y_disp)
            print("average location: ", np.mean(all_locs, axis=0))

            ax[1,0].imshow(vote_heatmap)

            coords_from_cm, arg_max_cm = get_coords(torch.tensor(np.expand_dims(np.expand_dims(vote_heatmap, axis=0), axis=0)).contiguous())
            coords_from_cm =coords_from_cm.detach().cpu().numpy()[0,0]
            print("get_coords from vote_heatmap: ", coords_from_cm, "max: ", arg_max_cm)

            ax[1,1].imshow(smoothed_heatmap)

            coords_from_sm, arg_max_vm = get_coords(torch.tensor(np.expand_dims(np.expand_dims(smoothed_heatmap, axis=0), axis=0)))
            coords_from_sm =coords_from_sm.detach().cpu().numpy()[0,0]
            print("get_coords from smoothed_heatmap: ", coords_from_sm, "max: ", arg_max_vm)



            plt.show()
        

    
    return torch.stack(smoothed_heatmaps)




