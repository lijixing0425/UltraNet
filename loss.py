import torch


def BinaryDiceLoss(input, targets):
    assert input.shape == targets.shape
    input_flat = input.reshape(targets.size()[0], -1)
    targets_flat = targets.reshape(targets.size()[0], -1)
    intersection = input_flat * targets_flat
    N_dice_eff = (2. * intersection.sum(1) + 1) / (input_flat.sum(1) + targets_flat.sum(1) + 1)
    loss = 1 - N_dice_eff.sum() / targets.size()[0]
    return loss


def MultiClassDiceLoss(input, target):
    target = torch.nn.functional.one_hot(target.long(), input.shape[1])
    total_loss = 0
    logits = torch.nn.functional.softmax(input, dim=1)
    for i in range(logits.shape[1]):
        dice_loss = BinaryDiceLoss(logits[:, i, :, :], target[..., i])
        total_loss += dice_loss

    return total_loss / logits.shape[1]

def point_map_loss(input, target, weight_target):

    target = target.long()
    temp = torch.ones_like(target)
    target = torch.where(target > temp, temp, target)

    rate = target.sum() / (target.shape[0] * target.shape[1] * target.shape[2])

    class_mask = torch.where(target == 1, 1 - rate, rate)

    if input.shape[-1] != target.shape[-1]:
        input = torch.nn.Upsample(scale_factor=target.shape[-1]//input.shape[-1], mode='bilinear', align_corners=True)(input)
    input = input[:, 0, :, :]

    assert input.shape == target.shape
    weight_mask = weight_target * class_mask
    bceloss = torch.nn.BCELoss(weight=weight_mask)
    loss1 = bceloss(input, target.float())
    loss2 = BinaryDiceLoss(input, target)
    return loss1, loss2

def BinaryDice(input, targets):
    assert input.shape == targets.shape
    input_flat = input.reshape(targets.size()[0], -1)
    targets_flat = targets.reshape(targets.size()[0], -1)
    intersection = input_flat * targets_flat
    N_dice_eff = (0.1 + 2. * intersection.sum(1)) / (0.1 + input_flat.sum(1) + targets_flat.sum(1))

    loss = N_dice_eff.sum() / targets.size()[0]
    return loss


def MultiClassDice(input, target):
    target = torch.nn.functional.one_hot(target.long(), input.shape[1])
    total_loss = 0
    logits = torch.argmax(input, dim=1)
    logits = torch.nn.functional.one_hot(logits.long(), input.shape[1])
    for i in range(input.shape[1]):
        dice_loss = BinaryDice(logits[:, :, :, i], target[:, :, :, i])
        total_loss += dice_loss

    return total_loss / input.shape[1]


def Binaryiou(input, targets):
    assert input.shape == targets.shape
    input_flat = input.reshape(targets.size()[0], -1)
    targets_flat = targets.reshape(targets.size()[0], -1)
    intersection = input_flat * targets_flat
    N_dice_eff = (0.1 + intersection.sum(1)) / (0.1 + input_flat.sum(1) + targets_flat.sum(1) - intersection.sum(1))
    loss = N_dice_eff.sum() / targets.size()[0]
    return loss


def MultiClassiou(input, target):
    target = torch.nn.functional.one_hot(target.long(), input.shape[1])
    total_loss = 0
    logits = torch.argmax(input, dim=1)
    logits = torch.nn.functional.one_hot(logits.long(), input.shape[1])
    for i in range(input.shape[1]):
        dice_loss = Binaryiou(logits[:, :, :, i], target[:, :, :, i])
        total_loss += dice_loss

    return total_loss / input.shape[1]