class ColorizationNet(nn.Module):
  def __init__(self, input_size=128):
    super(ColorizationNet, self).__init__()
    self.l_cent = 50.
    self.l_norm = 100.
    self.ab_norm = 110.

    ## First half: ResNet
    resnet = models.resnet18(num_classes=313) 
    # Change first conv layer to accept single-channel (grayscale) input
    resnet.conv1.weight = nn.Parameter(resnet.conv1.weight.sum(dim=1).unsqueeze(1)) 
    # Extract midlevel features from ResNet-gray
    self.midlevel_resnet = nn.Sequential(*list(resnet.children())[0:6])

    ## Second half: Upsampling
    self.upsample = nn.Sequential(
      nn.Upsample(scale_factor=2, mode='bilinear')
      nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(),
      nn.BatchNorm2d(256),
      nn.Conv2d(256, 313, kernel_size=1, stride=1, padding=0)
      nn.Softmax(dim=1)
      nn.Conv2d(313, 2, kernel_size=1, padding=0, dilation=1, stride=1)
      nn.Upsample(scale_factor=4, mode='bilinear'))
 
  def unnormalize_ab(self, in_ab):
		return in_ab*self.ab_norm
        
        
  def forward(self, input):

    # Pass input through ResNet-gray to extract features
    midlevel_features = self.midlevel_resnet(input)

    # Upsample to get colors
    output = self.upsample(midlevel_features)
    return output