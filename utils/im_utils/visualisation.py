import logging
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from scipy.stats import multivariate_normal
import matplotlib.patches as patchesplt


def visualize_image_target(image, target, coordinates):

    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(image)

    for coord in coordinates:
        rect1 = patches.Rectangle(
            (int(coord[0]), int(coord[1])),
            3,
            3,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        ax.add_patch(rect1)

    plt.show()
    plt.close()


def visualize_image_multiscaleheats_pred_coords(
    og_image, og_heatmap, upscaled_heatmap, pred_coords, targ_coords
):
    """
    visualize an image, the predicted heatmap, the upscaled heatmap. Show the predicted coords and target coords

    """

    # print("Shapes: ogimage %s, ogheatmap %s, upscaled heatmap %s, pred coords %s and targ coords %s " % (og_image.shape, og_heatmap.shape, upscaled_heatmap.shape, pred_coords.shape, targ_coords.shape))
    print("Shapes: ogimage", og_image.shape)
    print("Shapes: og_heatmap", og_heatmap.shape)
    print("Shapes: upscaled_heatmap", upscaled_heatmap.shape)
    print("Shapes: pred_coords", pred_coords.shape)
    print("Shapes: targ_coords", targ_coords.shape)

    fig, ax = plt.subplots(nrows=1, ncols=4, squeeze=False)
    additive_og_hm = np.sum(og_heatmap, axis=0)
    # additive_og_hm = np.zeros(og_heatmap[0].shape)
    # additive_og_hm = [np.add(additive_og_hm,x) for x in og_heatmap]

    # print(additive_og_hm)
    # print("lens ", len(additive_og_hm))
    # print("and shape: ", additive_og_hm.shape)

    # additive_us_hm = np.zeros(upscaled_heatmap[0].shape)
    # additive_us_hm = [np.add(additive_us_hm,x) for x in upscaled_heatmap]

    additive_us_hm = np.sum(upscaled_heatmap, axis=0)

    print(
        "additive  additive_og_hm and additive_us_hmshapes: ",
        additive_og_hm.shape,
        additive_us_hm.shape,
    )
    ax[0, 0].imshow(og_image, cmap="gray")
    ax[0, 0].set_title("Input Image")

    ax[0, 1].imshow(og_image, cmap="gray")
    ax[0, 1].set_title("Input Image with pred(r)/targ(g) coords")

    ax[0, 3].imshow(additive_us_hm, cmap="gray")
    ax[0, 3].set_title("Upscaled Predicted Heatmaps with pred(r)/targ(g) coords")

    ax[0, 2].imshow(additive_og_hm, cmap="gray")
    ax[0, 2].set_title("Predicted Heatmap")

    # ax[0,2].imshow(trans_image, cmap='gray')
    for co in range(len(pred_coords)):
        rect11 = patches.Rectangle(
            (int(pred_coords[co][0]), int(pred_coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )
        rect12 = patches.Rectangle(
            (int(pred_coords[co][0]), int(pred_coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )

        ax[0, 1].add_patch(rect11)
        ax[0, 1].text(int(pred_coords[co][0]), int(pred_coords[co][1]), co)
        ax[0, 3].add_patch(rect12)

        rect21 = patches.Rectangle(
            (int(targ_coords[co][0]), int(targ_coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )
        rect22 = patches.Rectangle(
            (int(targ_coords[co][0]), int(targ_coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )

        ax[0, 1].add_patch(rect21)
        ax[0, 1].text(int(targ_coords[co][0]), int(targ_coords[co][1]), co)

        ax[0, 3].add_patch(rect22)

    plt.show()
    plt.close()


def visualize_heat_pred_coords(og_image, pred_coords, targ_coords):
    """
    visualize an image, the predicted heatmap, the upscaled heatmap. Show the predicted coords and target coords

    """

    # print("Shapes: ogimage %s, ogheatmap %s, upscaled heatmap %s, pred coords %s and targ coords %s " % (og_image.shape, og_heatmap.shape, upscaled_heatmap.shape, pred_coords.shape, targ_coords.shape))
    print("Shapes: ogimage", og_image.shape)
    print("Shapes: pred_coords", pred_coords.shape)
    print("Shapes: targ_coords", targ_coords.shape)

    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
    # additive_og_hm = np.zeros(og_heatmap[0].shape)
    # additive_og_hm = [np.add(additive_og_hm,x) for x in og_heatmap]

    # print(additive_og_hm)
    # print("lens ", len(additive_og_hm))
    # print("and shape: ", additive_og_hm.shape)

    # additive_us_hm = np.zeros(upscaled_heatmap[0].shape)
    # additive_us_hm = [np.add(additive_us_hm,x) for x in upscaled_heatmap]

    ax[0, 0].imshow(og_image, cmap="gray")
    ax[0, 0].set_title("Input Image with pred(r)/targ(g) coords")

    # ax[0,2].imshow(trans_image, cmap='gray')
    for co in range(len(pred_coords)):
        rect11 = patches.Rectangle(
            (int(pred_coords[co][0]), int(pred_coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )

        ax[0, 0].add_patch(rect11)
        ax[0, 0].text(int(pred_coords[co][0]), int(pred_coords[co][1]), co)

        rect21 = patches.Rectangle(
            (int(targ_coords[co][0]), int(targ_coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="g",
            facecolor="none",
        )

        ax[0, 0].add_patch(rect21)
        ax[0, 0].text(int(targ_coords[co][0]), int(targ_coords[co][1]), co)

    plt.show()
    plt.close()


def visualize_image_all_coords(image, coords):
    """
    visualize an image and all landmark labels, with each landmark labelled by index.

    """

    # print("Shapes: ogimage %s, ogheatmap %s, upscaled heatmap %s, pred coords %s and targ coords %s " % (og_image.shape, og_heatmap.shape, upscaled_heatmap.shape, pred_coords.shape, targ_coords.shape))
    print("Shapes: image", image.shape)
    print("len: coords", len(coords))

    fig, ax = plt.subplots(nrows=1, ncols=1, squeeze=False)
    ax[0, 0].imshow(image, cmap="gray")
    ax[0, 0].set_title("Input Image with all coordinates labelled by index.")

    # ax[0,2].imshow(trans_image, cmap='gray')
    for co in range(len(coords)):
        rect11 = patches.Rectangle(
            (int(coords[co][0]), int(coords[co][1])),
            3,
            3,
            linewidth=2,
            edgecolor="r",
            facecolor="none",
        )

        ax[0, 0].add_patch(rect11)
        ax[0, 0].text(int(coords[co][0]), int(coords[co][1]), co)

    plt.show()
    plt.close()


def visualize_image_trans_coords(
    untrans_image, untrans_coords, trans_image, trans_coords
):
    """
    visualize an image and the same image after it has been transformed. also shows the coordinates on the trans image and untrans image

    """
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)

    print("og im shape@ ", untrans_image.shape)

    print(
        "mean and std of OG image:",
        np.round(np.mean(untrans_image), 5),
        np.round(np.std(untrans_image)),
        "and the image size@ ",
        untrans_image.shape,
    )
    print(
        "mean and std of trans image:",
        np.round(np.mean(trans_image.cpu().numpy()), 5),
        np.round(np.std(trans_image.cpu().numpy())),
        "and the image size@ ",
        trans_image.shape,
    )
    print("Num coords:", len(untrans_coords))

    ax[0, 0].imshow(untrans_image, cmap="gray")
    ax[0, 1].imshow(trans_image, cmap="gray")
    # ax[0,2].imshow(trans_image, cmap='gray')
    for coord_idx, co in enumerate(untrans_coords):
        # rect1 = patches.Rectangle((int(co[0]), int(co[1])),9,9,linewidth=2,edgecolor='r',facecolor='none')
        # ax[0,0].add_patch(rect1)
        ax[0, 0].plot(int(co[0]), int(co[1]), marker="+", mew=3, ms=10, color="red")
        ax[0, 1].plot(
            int(trans_coords[coord_idx][0]),
            int(trans_coords[coord_idx][1]),
            marker="+",
            markersize=20,
            linewidth=8,
            color="red",
        )

        text = ax[0, 0].text(
            int(co[0]) - 30,
            int(co[1]) - 10,  # Position
            r"$L_{{{}}}$".format(str(coord_idx + 1)),  # Text
            verticalalignment="bottom",  # Centered bottom with line
            horizontalalignment="center",  # Centered with horizontal line
            fontsize=15,  # Font size
            color="white",  # Color
        )

        text2 = ax[0, 1].text(
            int(trans_coords[coord_idx][0]) - 30,
            int(trans_coords[coord_idx][1]) - 10,  # Position
            r"$L_{{{}}}$".format(str(coord_idx + 1)),  # Text
            verticalalignment="bottom",  # Centered bottom with line
            horizontalalignment="center",  # Centered with horizontal line
            fontsize=15,  # Font size
            color="white",  # Color
        )
        # plt.axis('off')
        # text.set_path_effects([path_effects.Stroke(linewidth=3, foreground='black'),
        #                path_effects.Normal()])
    # for co in trans_coords:
    #     rect2 = patches.Rectangle((int(co[0]), int(co[1])),12,12,linewidth=2,edgecolor='g',facecolor='none')
    #     ax[0,1].add_patch(rect2)

    plt.show()
    plt.close()


def visualize_patch(
    full_im, full_lm, padded_image, padded_lm, image_patch, landmarks, padding_slice
):
    """
    visualize a full image with a landmark, and the image patch along with the same landmark.

    """
    assert len(full_lm) == len(landmarks) == 2
    logger = logging.getLogger()

    fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False)

    logger.info("full im shape: %s, patch shape: %s ", full_im.shape, image_patch.shape)

    ax[0, 0].imshow(full_im, cmap="gray")
    rect1 = patches.Rectangle(
        (int(full_lm[0]), int(full_lm[1])),
        3,
        3,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax[0, 0].add_patch(rect1)

    ax[0, 1].imshow(padded_image, cmap="gray")
    rect_patch = patches.Rectangle(
        (int(padding_slice[0]), int(padding_slice[1])),
        image_patch.shape[0],
        image_patch.shape[1],
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax[0, 1].add_patch(rect_patch)
    rect_plm = patches.Rectangle(
        (int(padded_lm[0]), int(padded_lm[1])),
        3,
        3,
        linewidth=2,
        edgecolor="r",
        facecolor="none",
    )
    ax[0, 1].add_patch(rect_plm)

    ax[0, 2].imshow(image_patch, cmap="gray")

    rect2 = patches.Rectangle(
        (int(landmarks[0]), int(landmarks[1])),
        3,
        3,
        linewidth=2,
        edgecolor="g",
        facecolor="none",
    )
    ax[0, 2].add_patch(rect2)

    plt.show()
    plt.close()


def visualize_centre_patch(
    full_im, full_landmarks, image_patch, patch_landmarks
):
    """
    visualize a full image with a landmark, and the image patch along with the same landmark.
    full_landmarks include: original center, jittered center, full groundtruth
    patch_landmarks include :  norm og center, norm, jittered center, norm groundtruth

    """
    assert len(patch_landmarks) == len(full_landmarks) == 3

    logger = logging.getLogger()

    fig, ax = plt.subplots(nrows=1, ncols=3, squeeze=False)

    logger.info("full image shape: %s, patch shape: %s ",
                full_im.shape, image_patch.shape)
    logger.info("MAGENTA = original centre, ORANGE = jittered centre, GREEN = ground Truth")

    # Original Image
    ax[0, 0].imshow(full_im, cmap="gray")
    rect1 = patches.Rectangle(
        (int(full_landmarks[0][0]), int(full_landmarks[0][1])),
        3,
        3,
        linewidth=2,
        edgecolor="m",
        facecolor="none",
    )
    ax[0, 0].add_patch(rect1)
    rect11 = patches.Rectangle(
        (int(full_landmarks[1][0]), int(full_landmarks[1][1])),
        3,
        3,
        linewidth=2,
        edgecolor="green",
        facecolor="none",
    )
    ax[0, 0].add_patch(rect11)

    rect13 = patches.Rectangle(
        (int(full_landmarks[2][0]), int(full_landmarks[2][1])),
        3,
        3,
        linewidth=2,
        edgecolor="orange",
        facecolor="none",
    )
    ax[0, 0].add_patch(rect13)

    ax[0, 1].imshow(image_patch, cmap="gray")
    rect20 = patches.Rectangle(
        (int(patch_landmarks[0][0]), int(patch_landmarks[0][1])),
        3,
        3,
        linewidth=2,
        edgecolor="m",
        facecolor="none",
    )
    ax[0, 1].add_patch(rect20)

    rect2 = patches.Rectangle(
        (int(patch_landmarks[1][0]), int(patch_landmarks[1][1])),
        3,
        3,
        linewidth=2,
        edgecolor="green",
        facecolor="none",
    )
    ax[0, 1].add_patch(rect2)
    rect22 = patches.Rectangle(
        (int(patch_landmarks[2][0]), int(patch_landmarks[2][1])),
        3,
        3,
        linewidth=2,
        edgecolor="orange",
        facecolor="none",
    )
    ax[0, 1].add_patch(rect22)

    k1 = multivariate_normal(mean=[int(full_landmarks[0][0]), int(full_landmarks[0][1])], cov=[[200, 0], [0, 200]])

    # create a grid of (x,y) coordinates at which to evaluate the kernels
    heatmap_shape = full_im.shape
    xlim = (0, heatmap_shape[0])
    ylim = (0, heatmap_shape[1])
    xres = int(heatmap_shape[0])
    yres = int(heatmap_shape[1])
    x = np.linspace(xlim[0], xlim[1], xres)
    y = np.linspace(ylim[0], ylim[1], yres)
    xx, yy = np.meshgrid(x, y)

    # evaluate kernels at grid points
    xxyy = np.c_[xx.ravel(), yy.ravel()]
    zz = (k1.pdf(xxyy))

    # reshape heatmap
    heatmap = np.array(zz.reshape((xres, yres)), dtype=np.float32)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255.0
    heatmap = np.clip(heatmap, 0, 255)
    heatmap = heatmap.astype(np.uint8)

    # create 3-channel image with heatmap in channel 0
    image = np.zeros((heatmap_shape[0], heatmap_shape[1], 3), dtype=np.uint8)
    image[:, :, 0] = heatmap
    ax[0, 2].imshow(image[:, :, 0])
    plt.show()
    plt.close()


def visualize_imageNcoords_cropped_imgNnormcoords(
    og_image, cropped_im, og_coords, norm_coords, lm_indicators
):
    """
    visualize an image, the cropped (with pad image), original coords and normalized coords to the crop.

    """
    fig, ax = plt.subplots(nrows=1, ncols=2, squeeze=False)

    # print("mean and std of OG image:", np.round(np.mean(og_image),5), np.round(np.std(og_image)), "and the image size@ ", og_image.shape)
    # print("mean and std of trans image:", np.round(np.mean(trans_image.cpu().numpy()),5), np.round(np.std(trans_image.cpu().numpy())), "and the image size@ ", trans_image.shape)
    # print("Num coords:", len(coords))

    ax[0, 0].imshow(og_image, cmap="gray")
    ax[0, 1].imshow(cropped_im, cmap="gray")
    # ax[0,2].imshow(trans_image, cmap='gray')
    for co in og_coords:
        rect1 = patches.Rectangle(
            (int(co[0]), int(co[1])), 3, 3, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax[0, 0].add_patch(rect1)

    for co in norm_coords:
        rect1 = patches.Rectangle(
            (int(co[0]), int(co[1])), 3, 3, linewidth=2, edgecolor="r", facecolor="none"
        )
        ax[0, 1].add_patch(rect1)

    plt.show()
    plt.close()


def visualize_image_trans_target(og_image, trans_image, target):
    """
    visualize an image and the same image after it has been transformed. also shows the genreated heatmap.

    """
    fig, ax = plt.subplots(nrows=1, ncols=target.shape[0] + 3, squeeze=False)
    logger = logging.getLogger()
    logger.info(
        "mean and std of OG image %s +/- %s.And the image size %s ",
        np.round(np.mean(og_image), 5),
        np.round(np.std(og_image)),
        og_image.shape,
    )
    logger.info(
        "mean and std of trans image: %s +/- %s. And the image size: %s",
        np.round(np.mean(trans_image.cpu().numpy()), 5),
        np.round(np.std(trans_image.cpu().numpy())),
        trans_image.shape,
    )
    logger.info(
        "mean and std of target landmark: %s +/- %s.And the target shape: %s ",
        np.round(np.mean(target.cpu().numpy()), 5),
        np.round(np.std(target.cpu().numpy())),
        target.shape,
    )

    ax[0, 0].imshow(og_image, cmap="gray")
    ax[0, 1].imshow(trans_image, cmap="gray")
    ax[0, 2].imshow(trans_image, cmap="gray")

    all_targets = []
    for a in range((target.shape[0])):

        if target[a].ndim == 3:
            ax[0, a + 3].imshow(np.squeeze(target[a]))
            ax[0, 2].imshow(np.squeeze(target[a]), cmap="Oranges", alpha=0.5)
        else:
            ax[0, a + 3].imshow(target[a])
            ax[0, 2].imshow(target[a], cmap="Oranges", alpha=0.5)

    plt.show()
    plt.close()


# a row for each predicted heatmap (3) and a column for each resolution of predicted heatmap
def visualize_predicted_heatmaps(heatmaps, predicted_coords, target_coords):
    fig, ax = plt.subplots(
        nrows=len(heatmaps), ncols=(predicted_coords.shape[1]), squeeze=False
    )

    # print("heatmaps len: ", len(heatmaps))
    # print("predicted coords shape: ", predicted_coords.shape)
    # print("target coords shape: ", target_coords.shape)

    for a in range(len(heatmaps)):
        for b in range(predicted_coords.shape[1]):
            # print(" Heatmap shape ", heatmaps[a].shape)
            ax[a, b].imshow(heatmaps[a][0][b])

            if a == len(heatmaps) - 1:
                rect1 = patches.Rectangle(
                    (int(predicted_coords[0, b, 0]), int(predicted_coords[0, b, 1])),
                    3,
                    3,
                    linewidth=2,
                    edgecolor="r",
                    facecolor="none",
                )
                ax[a, b].add_patch(rect1)

                rect2 = patches.Rectangle(
                    (int(target_coords[0, b, 0]), int(target_coords[0, b, 1])),
                    3,
                    3,
                    linewidth=2,
                    edgecolor="g",
                    facecolor="none",
                )
                ax[a, b].add_patch(rect2)

    plt.subplots_adjust(wspace=0, hspace=0)

    plt.show()

    # plt.show()
    # plt.close()


# def multi_variate_hm(data_dict, y_mean, cov_matr, heatmap_shape, noise=None, plot_targ=False):


#     transformed_input_image = data_dict["image"]
#     all_hms = []
#     for sample_idx, ind_sample_image in enumerate(transformed_input_image):
#         # extra_info = {"lower": lower, "upper": upper, "cov_matr": cov_matr}

#         # extra_info = ind_sample["extra_info"]
#         # create  kernel
#         m1 = y_mean[sample_idx]
#         s1 = cov_matr[sample_idx,:,sample_idx,:]

#         if noise is not None:
#             s1 += [[noise, 0], [0, noise]]

#         k1 = multivariate_normal(mean=m1, cov=s1)


#         # create a grid of (x,y) coordinates at which to evaluate the kernels
#         xlim = (0, heatmap_shape[0])
#         ylim = (0, heatmap_shape[1])
#         xres = int(heatmap_shape[0])
#         yres = int(heatmap_shape[1])

#         x = np.linspace(xlim[0], xlim[1], xres)
#         y = np.linspace(ylim[0], ylim[1], yres)
#         xx, yy = np.meshgrid(x, y)

#         # evaluate kernels at grid points
#         xxyy = np.c_[xx.ravel(), yy.ravel()]
#         zz = k1.pdf(xxyy)

#         # reshape and plot image
#         img = np.array(zz.reshape((xres, yres))*100, dtype=np.float16)

#         img = generate_heatmap_and_landmark(k1,data_dict["label"]["landmarks"][sample_idx] )

#         all_hms.append(np.expand_dims(img,axis=0))
#     return np.array(all_hms)

def multi_variate_hm(data_dict, y_mean, cov_matr, heatmap_shape, noise=None, plot_targ=False, plot_wo_noise_extra=False):
    transformed_input_image = data_dict["image"]
    all_hms = []
    all_hms_wo_noise = []
    all_full_covs = []

    for sample_idx, _ in enumerate(transformed_input_image):
        m1 = y_mean[sample_idx]

        s1 = cov_matr[sample_idx, :, sample_idx, :]
        if noise is not None:
            if isinstance(noise, list):
                s1 += [[noise[0], 0], [0, noise[1]]]
            elif noise.ndim == 0:
                s1 += [[noise, 0], [0, noise]]
            else:
                raise ValueError("Noise must be a scalar or a vector of length 2")
        all_full_covs.append(s1)
        covariances = [multivariate_normal(mean=m1, cov=s1)]

        if plot_wo_noise_extra:
            covariances.append(multivariate_normal(mean=m1, cov=cov_matr[sample_idx, :, sample_idx, :]))

        for idx, k1 in enumerate(covariances):
            # create a grid of (x,y) coordinates at which to evaluate the kernels
            xlim = (0, heatmap_shape[0])
            ylim = (0, heatmap_shape[1])
            xres = int(heatmap_shape[0])
            yres = int(heatmap_shape[1])
            x = np.linspace(xlim[0], xlim[1], xres)
            y = np.linspace(ylim[0], ylim[1], yres)
            xx, yy = np.meshgrid(x, y)

            # evaluate kernels at grid points
            xxyy = np.c_[xx.ravel(), yy.ravel()]
            zz = (k1.pdf(xxyy))

            # reshape heatmap
            heatmap = np.array(zz.reshape((xres, yres)), dtype=np.float32)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255.0
            heatmap = np.clip(heatmap, 0, 255)
            heatmap = heatmap.astype(np.uint8)

            # create 3-channel image with heatmap in channel 0
            image = np.zeros((heatmap_shape[0], heatmap_shape[1], 3), dtype=np.uint8)
            image[:, :, 0] = heatmap

        #     data = distr.k1(size = 5000)
        #     plt.plot(data[:,0],data[:,1], 'o')
            if plot_targ:
                # add green cross at landmark location
                if "landmarks_unstandardized" in data_dict["label"].keys():
                    landmark = data_dict["label"]["landmarks_unstandardized"][sample_idx][0]
                else:
                    landmark = data_dict["label"]["landmarks"][sample_idx]

                cross_size = 6
                # add intersection over the landmark
                x_min = max(0, int(landmark[0] - cross_size / 2))
                x_max = min(heatmap_shape[0], int(landmark[0] + (cross_size / 2)+1))

                y_min = max(0, int(landmark[1] - cross_size / 2))
                y_max = min(heatmap_shape[1], int(landmark[1] + (cross_size / 2)+1))
                # image[x_min:x_max, int(landmark[1]), 1] = 255  # green line
                # image[int(landmark[0]), y_min:y_max, 1] = 255  # green line

                image[y_min:y_max, int(landmark[0]), 1] = 255  # green line
                image[int(landmark[1]), x_min:x_max, 1] = 255  # green line

            if idx == 0:
                all_hms.append(np.expand_dims(image, axis=0))
            else:
                all_hms_wo_noise.append(np.expand_dims(image, axis=0))

    return np.array(all_hms),  np.array(all_hms_wo_noise), np.array(all_full_covs)


def plot_target(heatmaps, targets):
    assert heatmaps.shape[1] == 1, "Heatmaps must be of shape (batch_size, 1, height, width). 1 means 1 landmark training only. Can only plot one target currently."
    heatmaps_with_target = []
    for heatmap, target in zip(heatmaps, targets):
        target = target[0]
        heatmap_shape = heatmap.shape[-2:]
        image = np.zeros((heatmap_shape[0], heatmap_shape[1], 3), dtype=np.uint8)
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min()) * 255.0

        image[:, :, 0] = heatmap
        cross_size = 6
        # add intersection over the landmark
        x_min = max(0, int(target[0] - cross_size / 2))
        x_max = min(heatmap_shape[0], int(target[0] + (cross_size / 2)+1))

        y_min = max(0, int(target[1] - cross_size / 2))
        y_max = min(heatmap_shape[1], int(target[1] + (cross_size / 2)+1))
        # image[x_min:x_max, int(landmark[1]), 1] = 255  # green line
        # image[int(landmark[0]), y_min:y_max, 1] = 255  # green line

        image[y_min: y_max, int(target[0]), 1] = 255  # green line
        image[int(target[1]), x_min: x_max, 1] = 255  # green line
        heatmaps_with_target.append(np.expand_dims(image, axis=0))

    return np.array(heatmaps_with_target)
