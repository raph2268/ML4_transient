import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

# Loss functions
def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
    loss_1 = F.cross_entropy(y_1, t, reduce = False)
    ind_1_sorted = np.argsort(loss_1.data).cuda()
    loss_1_sorted = loss_1[ind_1_sorted]

    loss_2 = F.cross_entropy(y_2, t, reduce = False)
    ind_2_sorted = np.argsort(loss_2.data).cuda()
    loss_2_sorted = loss_2[ind_2_sorted]

    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]])/float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]])/float(num_remember)

    ind_1_update=ind_1_sorted[:num_remember]
    ind_2_update=ind_2_sorted[:num_remember]
    # exchange
    loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update])
    loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update])

    return torch.sum(loss_1_update)/num_remember, torch.sum(loss_2_update)/num_remember, pure_ratio_1, pure_ratio_2



# Adjusted loss_coteaching function for binary classification
def loss_coteaching_binary(y_1, y_2, t, forget_rate, ind): #, noise_or_not):
    # Convert t to float for binary cross entropy loss
    t = t.float()

    # Calculate binary cross entropy loss without reduction
    loss_1 = F.binary_cross_entropy_with_logits(y_1.squeeze(), t, reduction='none')
    loss_2 = F.binary_cross_entropy_with_logits(y_2.squeeze(), t, reduction='none')

    # Sort losses
    ind_1_sorted = torch.argsort(loss_1.data)
    ind_2_sorted = torch.argsort(loss_2.data)

    # Sort losses by index
    loss_1_sorted = loss_1[ind_1_sorted]
    loss_2_sorted = loss_2[ind_2_sorted]

    # Determine the number of samples to remember
    remember_rate = 1 - forget_rate
    num_remember = int(remember_rate * len(loss_1_sorted))

    """# Compute pure ratios
    pure_ratio_1 = np.sum(noise_or_not[ind[ind_1_sorted[:num_remember]]]) / float(num_remember)
    pure_ratio_2 = np.sum(noise_or_not[ind[ind_2_sorted[:num_remember]]]) / float(num_remember)"""

    # Update indices
    ind_1_update = ind_1_sorted[:num_remember]
    ind_2_update = ind_2_sorted[:num_remember]

    # Calculate updated losses based on exchanged indices
    loss_1_update = F.binary_cross_entropy_with_logits(y_1[ind_2_update].squeeze(), t[ind_2_update], reduction='none')
    loss_2_update = F.binary_cross_entropy_with_logits(y_2[ind_1_update].squeeze(), t[ind_1_update], reduction='none')

    # Average the losses for the batch
    return torch.mean(loss_1_update), torch.mean(loss_2_update)#, pure_ratio_1, pure_ratio_2# Adjusted loss_coteaching function for binary classification


def loss_coteaching_binary_asym(y_1, y_2, t, forget_rate_0, forget_rate_1, ind):
    # Convert t to float for binary cross entropy loss
    t = t.float()

    # Calculate binary cross entropy loss without reduction
    loss_1 = F.binary_cross_entropy_with_logits(y_1.squeeze(), t, reduction='none')
    loss_2 = F.binary_cross_entropy_with_logits(y_2.squeeze(), t, reduction='none')

    # Sort losses for both networks
    ind_1_sorted = torch.argsort(loss_1.data)
    ind_2_sorted = torch.argsort(loss_2.data)

    # Sort losses by index
    loss_1_sorted = loss_1[ind_1_sorted]
    loss_2_sorted = loss_2[ind_2_sorted]

    # Separate indices for class 0 and class 1
    ind_class_0 = (t == 0).nonzero(as_tuple=True)[0]
    ind_class_1 = (t == 1).nonzero(as_tuple=True)[0]

    # Apply sorting to class-specific indices
    ind_1_class_0_sorted = ind_1_sorted[torch.isin(ind_1_sorted, ind_class_0)]
    ind_1_class_1_sorted = ind_1_sorted[torch.isin(ind_1_sorted, ind_class_1)]
    ind_2_class_0_sorted = ind_2_sorted[torch.isin(ind_2_sorted, ind_class_0)]
    ind_2_class_1_sorted = ind_2_sorted[torch.isin(ind_2_sorted, ind_class_1)]

    # Determine number of samples to remember for each class
    remember_rate_0 = 1 - forget_rate_0
    remember_rate_1 = 1 - forget_rate_1
    num_remember_0 = int(remember_rate_0 * len(ind_class_0))
    num_remember_1 = int(remember_rate_1 * len(ind_class_1))

    # Select the indices to update based on class-specific remember rates
    ind_1_update = torch.cat((ind_1_class_0_sorted[:num_remember_0], ind_1_class_1_sorted[:num_remember_1]))
    ind_2_update = torch.cat((ind_2_class_0_sorted[:num_remember_0], ind_2_class_1_sorted[:num_remember_1]))

    # Calculate updated losses based on exchanged indices
    loss_1_update = F.binary_cross_entropy_with_logits(y_1[ind_2_update].squeeze(), t[ind_2_update], reduction='none')
    loss_2_update = F.binary_cross_entropy_with_logits(y_2[ind_1_update].squeeze(), t[ind_1_update], reduction='none')

    # Average the losses for the batch
    return torch.mean(loss_1_update), torch.mean(loss_2_update)
