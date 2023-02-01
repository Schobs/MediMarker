

class GPFlowLoss():
    """
    Gaussian Process loss

    """

    def __init__(self, loss_func):
        self.loss = loss_func
        self.loss_seperated_keys = ["all_loss_all"]

    def forward(self, x, y, sigmas):

        l = -self.loss(x, y["landmarks"])

        loss_seperated = {"all_loss": l}

        return l, loss_seperated
