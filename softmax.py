import numpy as np
import torch
import torch.nn.functional as F

def softmax(scores):
    #scores = np.asarray(scores, dtype=np.float64)

    e_x = np.exp(scores - np.max(scores))
    output_custom1 = e_x / e_x.sum(axis=0)

    # maxes = torch.mean(torch.as_tensor(scores), 1, keepdim=True)[0]
    # x_exp = torch.exp(torch.as_tensor(scores) - maxes)
    # x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    # output_custom2 = x_exp / x_exp_sum
    # output_custom2 = output_custom2.numpy()

    torch_output = F.softmax(torch.as_tensor(scores), 0).numpy()

    difference = (output_custom1 - torch_output).sum()



    return output_custom1

