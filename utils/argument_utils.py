

def get_evaluation_mode(eval_mode, og_im_size, inp_size):
    if eval_mode == 'use_input_size':
        use_full_res_coords =False
        resize_first = False
    elif og_im_size == inp_size: 
        print("your DATASET.ORIGINAL_IMAGE_SIZE == DATASET.INPUT_SIZE, therefore defaulting evaluation mode to \"use_input_size\"")
        use_full_res_coords =False
        resize_first = False
    elif eval_mode== 'scale_heatmap_first':
        use_full_res_coords =True
        resize_first = True
    elif eval_mode == 'scale_pred_coords':
        use_full_res_coords =True
        resize_first = False
    else:
        raise ValueError("value for cg.INFERENCE.EVALUATION_MODE not recognised. Choose from: scale_heatmap_first, scale_pred_coords, use_input_size")
    return use_full_res_coords, resize_first