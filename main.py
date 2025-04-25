from flask import Flask, render_template, send_file
import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.utils as vutils
import requests
from PIL import Image
import io
import random
from torchvision.utils import save_image
from io import BytesIO

# Model Definition
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

app = Flask(__name__)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nz = 100
netG = Generator().to(device)
state_dict = torch.load("generator.pth", map_location=device)
if "module." in list(state_dict.keys())[0]:
    # If saved with DataParallel, remove 'module.' prefix
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace("module.", "")
        new_state_dict[new_key] = v
    state_dict = new_state_dict

netG.load_state_dict(state_dict)

netG.eval()

from flask import Flask, render_template
import random

app = Flask(__name__)

@app.route('/')
def index():
    rand_value = random.randint(0, 99999)
    return render_template("index.html", rand=rand_value)

@app.route("/generate")
def generate():
    with torch.no_grad():
        noise = torch.randn(1, nz, 1, 1, device=device)
        fake = netG(noise).cpu()

        # Save to BytesIO buffer
        buffer = BytesIO()
        save_image(fake, buffer, format="PNG", normalize=True)
        buffer.seek(0)

        return send_file(buffer, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
