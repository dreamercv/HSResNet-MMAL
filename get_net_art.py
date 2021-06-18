# encoding utf-8
"""
    @author: Binge.Van
    @input:
    #outut:
    @desc:
        resnet模型与hs模型合并生成新模型
        
"""
import torch
import os
from collections import OrderedDict


def get_acc(nohup_path, acc_path):
    nohup_info = open(nohup_path, "r", encoding="utf-8")
    lines = nohup_info.readlines()
    acc_info = open(acc_path, "w", encoding="utf-8")
    keywords = ["Train set:", "Test set:", "Training"]
    for line in lines:
        for keyword in keywords:
            if keyword in line:
                acc_info.write(line + "\n")
                break
    acc_info.close()
    nohup_info.close()


def merge_pth(pth_root: str, epoch_list: list, save_path):
    epoch_list.sort()

    new_epoch = torch.load(os.path.join(pth_root, f"air_hs_epoch{epoch_list[0]}.pth"))
    new_state_dict = new_epoch["model_state_dict"]

    num_epoch = len(epoch_list)
    for i in range(1,num_epoch):
        pth_path = os.path.join(pth_root, f"air_hs_epoch{epoch_list[i]}.pth")
        weight = torch.load(pth_path)
        model_state_dict = weight["model_state_dict"]
        for key, value in model_state_dict.items():
            if "num_batches_tracked" in key:
                # print(epoch_list[i])
                # print(model_state_dict[key])
                continue
            if key in new_state_dict.keys():
                new_state_dict[key] += value
            else:
                new_state_dict[key] = value

    for n_key in new_state_dict.keys():
        # print(n_key)
        if "num_batches_tracked" in n_key:
            continue
        # print(new_state_dict[n_key])
        new_state_dict[n_key] /= num_epoch
    print(new_epoch["epoch"])
    print(new_epoch["learning_rate"])
    torch.save({
        'epoch': new_epoch["epoch"],
        'model_state_dict': new_state_dict,
        'learning_rate':  new_epoch["learning_rate"]}, os.path.join(save_path, f"air_hs_epoch{new_epoch['epoch']}.pth"))
    # print(new_state_dict)


if __name__ == '__main__':

    # get_acc("/home/MMAL-Net/nohup_hs.out","/home/MMAL-Net/hs_resnet.out")
    pth_root = "/home/MMAL-Net/checkpoint/aircraft_hs"
    epoch_list = [68, 69, 70, 61, 62]
    save_path = "/home/MMAL-Net/checkpoint/aircraft_hs_avg/"
    if os.path.exists(save_path):
        os.system(f"rm -rf {save_path}")
        os.mkdir(save_path)
    else:
        os.mkdir(save_path)
    merge_pth(pth_root, epoch_list, save_path)

    # mmalnet_model_path = "/home/MMAL-Net/models/air_epoch146.pth"
    # mmalnet_hs_model_root = "/home/MMAL-Net/checkpoint/aircraft/"
    # save_model_path = os.path.join("/home/MMAL-Net/models/air_hs_epoch1.pth")
    # mmalnet_model = torch.load(mmalnet_model_path)
    # mmalnet_hs_model_001 = torch.load(os.path.join(mmalnet_hs_model_root, "epoch1.pth"))
    # mmalnet_hs_model_157 = torch.load((os.path.join(mmalnet_hs_model_root, "epoch157.pth")))
    # print("Loading OK ...")
    # # 修改epoch/lr/model_state_dict的值
    # new_state_dict = OrderedDict()
    # for key in mmalnet_hs_model_157['model_state_dict'].keys():
    #     name = key
    #     if key in mmalnet_model['model_state_dict'].keys():
    #         value = mmalnet_model['model_state_dict'][key]
    #     else:
    #         value = mmalnet_hs_model_157['model_state_dict'][key]
    #     new_state_dict[name] = value
    # torch.save({
    #     'epoch': mmalnet_hs_model_001['epoch'],
    #     'model_state_dict': new_state_dict,
    #     'learning_rate': mmalnet_hs_model_001['learning_rate'],
    # }, save_model_path)
    # # print(new_state_dict)
    # print("Re_write OK ..")

# import GPUtil
# import time
# if __name__ == '__main__':
#     while True:
#         Gpus = GPUtil.getGPUs()
#         for gpu in Gpus:
#             print(gpu.id,
#                   gpu.name,
#                   gpu.serial,
#                   gpu.uuid,
#                   gpu.load*100,
#                   gpu.memoryUtil*100,
#                   gpu.memoryTotal,
#                   gpu.memoryUsed,
#                   gpu.memoryFree,
#                   gpu.display_mode,
#                   gpu.display_active)
#         time.sleep(2)
#         print(type(GPUtil.showUtilization(all=True)))
# '''
# pip install GPUtil
# '''
