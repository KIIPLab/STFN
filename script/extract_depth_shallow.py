import torch

def get_depth_feature(save_output):
    #print('vit1:',save_output.outputs)

    feat = torch.cat(
        (
            #save_output.outputs[7][:, 1:, :],
            #save_output.outputs[8][:, 1:, :],
            save_output.outputs[9][:, 1:, :],
            save_output.outputs[10][:, 1:, :],
            save_output.outputs[11][:, 1:, :],
        ),
        dim=2
    )
    #print(save_output.outputs[11].shape)
    #print("depth:",feat.shape)

    return feat

def get_shallow_feature(save_output):

    #print('vit2 lengh:',len(save_output.outputs))

    feat = torch.cat(
        (
            save_output.outputs[0][:,1:,:],
            save_output.outputs[1][:,1:,:],
            save_output.outputs[2][:,1:,:],
            #save_output.outputs[3][:,1:,:],
            #save_output.outputs[4][:,1:,:],
        ),
        dim=2
    )
    #print("shallow:", feat.shape)
    #print(save_output.outputs[4].shape)

    return feat