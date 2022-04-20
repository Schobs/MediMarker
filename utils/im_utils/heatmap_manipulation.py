import torch

def get_coords(images):

    ''' get predictions from score maps in torch Tensor
        return type: torch.LongTensor
    '''
    assert images.dim() == 4, 'Score maps should be 4-dim'
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
    return preds