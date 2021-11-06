import numpy as np
import torch
import torch.nn.functional as F

def softmax(scores):
    #scores = np.asarray(scores, dtype=np.float64)

    #np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)

    e_x = np.exp(scores) # - np.max(scores))
    result = e_x / e_x.sum(axis=0)

    # OLD VERSION
    # e_x2 = np.exp(scores - np.max(scores))
    # result2 = e_x2 / e_x2.sum(axis=0)



    # test = np.sum(result[:, 2])

    # maxes = torch.mean(torch.as_tensor(scores), 1, keepdim=True)[0]
    # x_exp = torch.exp(torch.as_tensor(scores) - maxes)
    # x_exp_sum = torch.sum(x_exp, 1, keepdim=True)
    # output_custom2 = x_exp / x_exp_sum
    # output_custom2 = output_custom2.numpy()

    torch_output = F.softmax(torch.as_tensor(scores), 0)
    # test2 = torch.sum(torch_output[:, 0])
    # difference = (output_custom1 - torch_output).sum()

    return result

