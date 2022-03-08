import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

def visualize_image_target(image, target):
    fig, ax = plt.subplots(nrows=len(target)+2, ncols=target[0].shape[0], squeeze=False)
    
    print("mean and std of image:", np.round(np.mean(image),5), np.round(np.std(image)), "and the image size@ ", image.shape)

    print("shapes ", len(target),target[0].shape[0])
    for cols in range(target[0].shape[0]):
        ax[0, cols].imshow(image)
        ax[1, cols].imshow(image)
        ax[1, cols].imshow(target[-1][cols], cmap='Oranges', alpha=0.5)
        for row in range(len(target)):
            ax[row+2, cols].imshow(target[row][cols])
          

    plt.show()
    # plt.close()

        # all_targets = []
        # for a in range((target.shape[0])):

        #     if target[a].ndim == 3:
        #         ax[a+2].imshow(np.squeeze(target[a]))
        #         ax[1].imshow(np.squeeze(target[a]), cmap='Oranges', alpha=0.5)
        #     else:
        #         ax[a+2].imshow(target[a])
        #         ax[1].imshow(target[a], cmap='Oranges', alpha=0.5)
                



def visualize_image_trans_target(og_image, trans_image, target):
    '''
    visualize an image and the same image after it has been transformed. also shows the genreated heatmap.
    
    '''
    fig, ax = plt.subplots(nrows=1, ncols=target.shape[0]+3, squeeze=False)
    
    print("mean and std of OG image:", np.round(np.mean(og_image),5), np.round(np.std(og_image)), "and the image size@ ", og_image.shape)
    print("mean and std of trans image:", np.round(np.mean(trans_image.cpu().numpy()),5), np.round(np.std(trans_image.cpu().numpy())), "and the image size@ ", trans_image.shape)
    print("mean and std of trans image:", target.shape, np.round(np.mean(target.cpu().numpy()),5), np.round(np.std(target.cpu().numpy())), "and the image size@ ", target.shape)

    ax[0,0].imshow(og_image, cmap='gray')
    ax[0,1].imshow(trans_image, cmap='gray')
    ax[0,2].imshow(trans_image, cmap='gray')


    all_targets = []
    for a in range((target.shape[0])):

        if target[a].ndim == 3:
            ax[0,a+3].imshow(np.squeeze(target[a]))
            ax[0,2].imshow(np.squeeze(target[a]), cmap='Oranges', alpha=0.5)
        else:
            ax[0,a+3].imshow(target[a])
            ax[0,2].imshow(target[a], cmap='Oranges', alpha=0.5)
            


    plt.show()
    plt.close()

# a row for each predicted heatmap (3) and a column for each resolution of predicted heatmap
def visualize_predicted_heatmaps(heatmaps, predicted_coords, target_coords):
    fig, ax = plt.subplots(nrows=len(heatmaps), ncols=(predicted_coords.shape[1]), squeeze=False)

    # print("heatmaps len: ", len(heatmaps))
    # print("predicted coords shape: ", predicted_coords.shape)
    # print("target coords shape: ", target_coords.shape)

    for a in range(len(heatmaps)):
        for b in range(predicted_coords.shape[1]):
            # print(" Heatmap shape ", heatmaps[a].shape)
            ax[a, b].imshow(heatmaps[a][0][b])

            if a == len(heatmaps)-1:
                rect1 = patches.Rectangle((int(predicted_coords[0,b,0]), int(predicted_coords[0,b,1])),3,3,linewidth=2,edgecolor='r',facecolor='none')
                ax[a, b].add_patch(rect1)

                rect2 = patches.Rectangle((int(target_coords[0,b,0]), int(target_coords[0,b,1])),3,3,linewidth=2,edgecolor='g',facecolor='none')
                ax[a, b].add_patch(rect2)

    
    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()

    # plt.show()
    # plt.close()
