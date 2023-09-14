import torch

def get_resnet_feature(save_output):
    #for i in range(len((save_output.outputs))):
    #    print(save_output.outputs[i].shape)
    feat = torch.cat(
        (
            save_output.outputs[0], # output을 3,4,5 layer에서 가져온 경우의 feature :([batch, 512, 28, 28])
            save_output.outputs[1],
            save_output.outputs[2],
        ), #outputs[3][4][5]

        dim=1
    )
    # resnet layer 중 3개 layer를 사용했음을 볼 수 있음.
    #print('feat of resnet:', feat.shape) # 각 레이어는 (batch, 256, 56, 56) 만큼의 shape를 가지게 되며
    # cat시킬 경우, 최종 feat은  (batch,768,56,56)의 shape를 갖게 된다.
    return feat

def get_vit_feature(save_output):
    #for i in range(len((save_output.outputs))):
    #    print(save_output.outputs[i].shape)
    feat = torch.cat(
        (
            save_output.outputs[0][:,1:,:],
            save_output.outputs[1][:,1:,:],
            save_output.outputs[2][:,1:,:],
            save_output.outputs[3][:,1:,:],
            save_output.outputs[4][:,1:,:],
        ),
        dim=2
    )
    # vit의 12개 블록 중 5번째 블록까지만 사용. 또한 각 블록의 결과를 cat시킨 것으로 보아
    # vit의 최종 결과까지 가지 않고, 중간 블록 중 일부의 output을 사용했음을 볼 수 있음
    #print('shape if vit',save_output.outputs[11].shape)
    #print('feat of vit:',feat.shape)
    return feat

def get_inception_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[0],
            save_output.outputs[2],
            save_output.outputs[4],
            save_output.outputs[6],
            save_output.outputs[8],
            save_output.outputs[10]
        ),
        dim=1
    )
    return feat

def get_resnet152_feature(save_output):
    feat = torch.cat(
        (
            save_output.outputs[3],
            save_output.outputs[4],
            save_output.outputs[6],
            save_output.outputs[7],
            save_output.outputs[8],
            save_output.outputs[10]
        ),
        dim=1
    )
    return feat

def get_feature_swin(save_output):
    # swin은 block수가 24개
    # (4, 56*56, 128)
    # (4, 56*56, 128)

    feat = torch.cat(
        (
            save_output.outputs[0][:, :, :],
            save_output.outputs[1][:, :, :],

        ),
        dim=2
    )
    #print(feat.shape)

    return feat