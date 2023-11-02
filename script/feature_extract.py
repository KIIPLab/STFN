import torch

def get_first_feature_swin(save_output):
    # swin은 layer수가 총 24개, block수가 총 4개
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

def get_second_feature_swin(save_output) :
    # swin transformer block별 layer shape 변화 추이
    # stage 0~1: (-, 56*56 , 128)
    # stage 2~3: ( - , 28*28, 256)
    # stage 4~21: ( - , 14*14, 512)
    # stage 22~23: ( - , 7*7, 1024)
    # layer 생김새는 ViT보단 오히려 ResNet과 유사

    feat = torch.cat(
        (
        save_output.outputs[2][:, :, :],
        save_output.outputs[3][:, :, :],
        ),
        dim=2
    )
    return feat

def get_third_feature_swin(save_output) :
    # swin transformer block별 layer shape 변화 추이
    # stage 0~1: (-, 56*56 , 128)
    # stage 2~3: ( - , 28*28, 256)
    # stage 4~21: ( - , 14*14, 512)
    # stage 22~23: ( - , 7*7, 1024)
    # layer 생김새는 ViT보단 오히려 ResNet과 유사

    feat = torch.cat(
        (
        save_output.outputs[4][:, :, :],
        save_output.outputs[5][:, :, :]
        ),
        dim=2
    )
    return feat


def get_forth_feature_swin(save_output) :
    # swin transformer block별 layer shape 변화 추이
    # stage 0~1: (-, 56*56 , 128)
    # stage 2~3: ( - , 28*28, 256)
    # stage 4~21: ( - , 14*14, 512)
    # stage 22~23: ( - , 7*7, 1024)
    # layer 생김새는 ViT보단 오히려 ResNet과 유사

    feat = torch.cat(
        (
        save_output.outputs[22][:, :, :],
        save_output.outputs[23][:, :, :],
        ),
        dim=2
    )
    return feat


