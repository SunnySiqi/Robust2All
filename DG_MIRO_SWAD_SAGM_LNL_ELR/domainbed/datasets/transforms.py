from torchvision import transforms as T
from .tps_transform import TPSTransform

basic = T.Compose(
    [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
aug = T.Compose(
    [
        T.RandomResizedCrop(224, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.3, 0.3, 0.3, 0.3),
        T.RandomGrayscale(p=0.1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

CP_weak = T.Compose(
    [
        T.Resize(256),
        T.RandomResizedCrop(
            224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        T.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084)),
    ]
)


CP_strong = T.Compose(
    [
        T.Resize(256),
        T.RandomResizedCrop(
            224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(),
        TPSTransform(p=1),
        T.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084)),
    ]
)

CP_test = T.Compose(
    [
        T.Resize(256),
        T.CenterCrop(224),
        T.Normalize((0.09957533, 0.19229738, 0.16250879, 0.18240248, 0.14978176),
(0.077283904, 0.074369825, 0.06784963, 0.066472545, 0.068180084)),
    ]
)
