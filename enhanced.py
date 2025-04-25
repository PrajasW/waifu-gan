from flask import Flask, render_template, send_file
import torch
from torch import nn
import random
from torchvision.utils import save_image
from io import BytesIO
from PIL import Image
from collections import OrderedDict

# Model Definitions
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)
class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),  # 1st layer
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),  # 2nd layer
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),  # 3rd layer
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),  # 4th layer
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),  # Final layer
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)

# Helper function to remove "module." prefix from DataParallel models
def remove_data_parallel_prefix(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    return new_state_dict

# Flask app definition
app = Flask(__name__)

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nz = 100
netG = Generator().to(device)
netD = Discriminator().to(device)

# Load pre-trained model states
state_dict_G = torch.load("generator.pth", map_location=device)
state_dict_D = torch.load("discriminator.pth", map_location=device)

# Remove 'module.' prefix if model was saved with DataParallel
state_dict_G = remove_data_parallel_prefix(state_dict_G)
state_dict_D = remove_data_parallel_prefix(state_dict_D)

# Load state dict into models
netG.load_state_dict(state_dict_G)
netD.load_state_dict(state_dict_D)

netG.eval()
netD.eval()

# Helper function to generate image and check with discriminator
def generate_and_check():
    while True:
        # Generate random noise and pass it through the generator
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake_image = netG(noise).cpu()

        # Pass generated image through the discriminator
        output = netD(fake_image)
        
        # If the discriminator classifies it as real (output close to 1), return it
        if output.item() > 0.5:
            return fake_image
        # Otherwise, generate a new image

@app.route('/')
def index():
    rand_value = random.randint(0, 99999)
    return render_template("index.html", rand=rand_value)

@app.route("/generate")
def generate():
    # Generate and check the image using the function above
    fake_image = generate_and_check()

    # Save to BytesIO buffer
    buffer = BytesIO()
    save_image(fake_image, buffer, format="PNG", normalize=True)
    buffer.seek(0)

    return send_file(buffer, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
