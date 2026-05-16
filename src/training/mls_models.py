import torch.nn as nn
from torchvision import models

class SliceSelectorModel(nn.Module):
    def __init__(self):
        super(SliceSelectorModel, self).__init__()
        # برای طبقه بندی 0 و 1 (این اسلایس هدف هست یا نه)، ResNet18 کافی و سریع است
        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # ورودی ما 3 کاناله (PNG) است، پس نیاز به تغییر لایه اول نیست
        in_features = self.backbone.fc.in_features
        # خروجی 1 نورون برای مسئله Binary
        self.backbone.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        return self.backbone(x)

class KeypointModel(nn.Module):
    def __init__(self):
        super(KeypointModel, self).__init__()
        # برای رگرسیون دقیق کی‌پوینت‌ها، ResNet34 پایدارتر است
        self.backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 6), # 3 نقطه * 2 مختصات (X,Y)
            nn.Sigmoid()       # چون مختصات را بین 0 و 1 نرمال کرده‌ایم
        )

    def forward(self, x):
        return self.backbone(x)