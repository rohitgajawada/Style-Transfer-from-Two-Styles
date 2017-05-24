import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.models as models
import copy

dtype = torch.cuda.FloatTensor

imsize = 256

loader = transforms.Compose([
    transforms.Scale(imsize),
    transforms.ToTensor()
])

def image_loader(img):
    image = Image.open(img)
    image = Variable(loader(image))
    image = image.unsqueeze(0)
    return image

style1img = image_loader("picasso.jpg").type(dtype)
style2img = image_loader("style.jpg").type(dtype)
contentimg = image_loader("newme.jpg").type(dtype)

assert style1img.size() == contentimg.size()
assert style1img.size() == style2img.size()

class Contentloss(nn.Module):

    def __init__(self, target, weight):
        super(Contentloss, self).__init__()
        self.target = weight * target.detach()
        self.weight = weight
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        self.output = input
        return self.output

    def backward(self, retain_variables = True):
        self.loss.backward(retain_variables = retain_variables)
        return self.loss

class GramMatrix(nn.Module):

    def forward(self, input):
        a, b, c, d = input.size()
        features = input.view(a * b, c* d)
        G = torch.mm(features, features.t())
        return G.div(a * b * c * d)

class StyleLoss(nn.Module):

    def __init__(self, target, weight):
        super(StyleLoss, self).__init__()
        self.target = weight * target.detach()
        self.weight = weight
        self.gram = GramMatrix()
        self.criterion = nn.MSELoss()

    def forward(self, input):
        self.output = input.clone()
        self.stygram = self.gram(input)
        self.stygram.mul_(self.weight)
        self.loss = self.criterion(self.stygram * self.weight, self.target)

    def backward(self, retain_variables = True):
        self.loss.backward(retain_variables = retain_variables)
        return self.loss

cnn = models.vgg19(pretrained = True).features

cnn = cnn.cuda()

needed_contentlayers = ['conv_4']
needed_stylelayers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def genModelandloss(cnn, style1img, style2img, content_img, style_weight = 700, content_weight = 1, content_layers = needed_contentlayers, style_layers = needed_stylelayers):

    cnn = copy.deepcopy(cnn)
    content_losses = []
    style_losses =[]

    model = nn.Sequential()
    gram = GramMatrix()

    model = model.cuda()
    gram = gram.cuda()

    i = 1
    for layer in list(cnn):

        if isinstance(layer, nn.Conv2d):
            name = "conv_" + str(i)
            model.add_module(name, layer)

            if name in content_layers:
                target = model(content_img).clone()
                content_loss = Contentloss(target, content_weight)
                model.add_module("content_loss_" + str(i), content_loss)
                content_losses.append(content_loss)

            if name in style_layers:
                print(model)
                print("output1:", model(style1img))
                # print("output2:", model(style2img))

                x = model(style1img)
                # y = model(style2img)

                target_feature1 = x.clone()
                # target_feature2 = y.clone()

                target_featuregram1 = gram(target_feature1)
                # target_featuregram2 = gram(target_feature2)

                style_loss1 = StyleLoss(target_featuregram1, style_weight)
                # style_loss2 = StyleLoss(target_featuregram2, style_weight)

                model.add_module("style_loss1_" + str(i), style_loss1)
                style_losses.append(style_loss1)
                # model.add_module("style_loss2_" + str(i), style_loss2)
                # style_losses.append(style_loss2)


        if isinstance(layer, nn.ReLU):
            name = "relu_" + str(i)
            model.add_module(name, layer)
            i += 1

        if isinstance(layer, nn.MaxPool2d):
            name = "pool_" + str(i)
            model.add_module(name, layer)

    return model, style_losses, content_losses

input_img = contentimg.clone()

def run_style_transfer(cnn, contentimg, styleimg1, styleimg2, input_img, num_steps = 10, style_weight = 700, content_weight = 1):

    print("Running rohit's next level style transfer")

    model, style_losses, content_losses = genModelandloss(cnn, styleimg1, styleimg2, contentimg, style_weight, content_weight)

    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            input_param.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_param)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.backward()
            for cl in content_losses:
                content_score += cl.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                style_score.data[0], content_score.data[0]))
                print()

            return style_score + style_score

        optimizer.step(closure)

    input_param.data.clamp_(0, 1)

    return input_param.data

output = run_style_transfer(cnn, contentimg, style1img, style2img, input_img)

unloader = transforms.ToPILImage()
plt.ion()

def imshow(tensor, title=None):
    image = tensor.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)

    plt.pause(0.001)

plt.figure()
imshow(output, title='Output Image')

# sphinx_gallery_thumbnail_number = 4
plt.ioff()
plt.show()
