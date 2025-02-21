# import torch
# pretrained_weights = torch.load('./detr-r50-e632da11.pth')

# num_class = 8
# pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1, 256)
# pretrained_weights["model"]["class_embed.weight"].resize_(num_class+1)
# torch.save(pretrained_weights,"detr-r50_%d.path" %num_class)


import torch
import torch.nn.functional as F

#pretrained_weights = torch.load('./detr-r50-e632da11.pth')
pretrained_weights = torch.load('./detr-r101-2c7b67e5.pth')

num_class = 17

# Adjust class_embed.weight
old_weight = pretrained_weights["model"]["class_embed.weight"]
new_weight = F.pad(old_weight, (0, 0, 0, num_class + 1 - old_weight.shape[0]))
pretrained_weights["model"]["class_embed.weight"] = new_weight

# Adjust class_embed.bias
old_bias = pretrained_weights["model"]["class_embed.bias"]
new_bias = F.pad(old_bias, (0, num_class + 1 - old_bias.shape[0]))
pretrained_weights["model"]["class_embed.bias"] = new_bias

#torch.save(pretrained_weights, "detr-r50_%d.pth" % num_class)
torch.save(pretrained_weights, "detr-r101_%d.pth" % num_class)
