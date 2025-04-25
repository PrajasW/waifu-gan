# Waifu GAN

This project allows you to generate random waifu images using a DCGAN Architecture. You can run the generator in a standard mode or with an enhanced version that uses a Discriminator to verify the generated images before serving them to the user.

## Prerequisites

1. **Python 3.x** (preferably Python 3.7+)
2. **PyTorch** (with CUDA support if using GPU)
3. **Flask** for serving the web interface
4. **Other Dependencies**: Listed below

### Required Libraries

Before running the code, make sure you install all the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

1. Clone this repository to your local machine.
2. Install the required Python packages using the command above.
3. Make sure your trained model weights (`generator.pth` and `discriminator.pth`) are located in the same directory as the Python scripts.

### Project Structure

```
.
├── generator.pth            # Trained Generator model
├── discriminator.pth        # Trained Discriminator model (if using enhanced mode)
├── main.py                  # Standard Generator script (Flask app)
├── enhanced.py              # Enhanced Generator with Discriminator check (Flask app)
├── requirements.txt         # List of required packages
├── static/                  # Static files (images, stylesheets, etc.)
├── templates/               # HTML templates for the UI
│   └── index.html           # Main webpage for generating images
└── README.md                # Project README
```

## Running the Application

### 1. **Normal Generator** (without Discriminator Check)

To run the standard waifu image generator (without a discriminator to verify the generated image), use the following command:

```bash
python main.py
```

This will start a Flask web server, and you can view the web UI by visiting `http://localhost:5000` in your browser. The app will generate and display random waifu images on each page refresh.

### 2. **Enhanced Generator** (with Discriminator Check)

If you'd like to include a Discriminator to verify the authenticity of generated images, use the following command:

```bash
python enhanced.py
```

In this mode, after generating an image, the Discriminator will evaluate the generated image. If the image passes the check, it will be displayed to the user; otherwise, the generator will try a different combination.

### 3. **Web Interface**

After running either of the scripts, the web UI will be accessible at `http://localhost:5000`. You will be able to:

- **Generate Random Waifu Images**: Each time you refresh the page, a new random waifu image will be displayed.
- **View the Generated Image**: The image will be shown on the page, and you can refresh to generate a new one.

### Example Image Generation

After running the app, the URL for the generated image might look like:
<p float="left">
  <img src="https://github.com/user-attachments/assets/0d8bc53f-4538-4004-9089-950fc4515c02" width="200" height="200" />
  <img src="https://github.com/user-attachments/assets/7e75f331-3625-4c95-86af-3adaea963b32" width="200"  height="200"/>
  <img src="https://github.com/user-attachments/assets/75bf97a9-e59f-49e7-a607-33282926f781" width="200"  height="200"/>
</p>

This image will be shown on the web page.
![image](https://github.com/user-attachments/assets/aa7c3989-d111-44ac-bf05-0de79139e6f2)

## Kaggle Notebook Reference

For reference on how to train the GAN model used in this project, you can check the Kaggle notebook:

[Kaggle GAN Training Notebook](https://www.kaggle.com/code/prajaspw/anime-gan)

This notebook covers the training process, the architecture used for both the generator and discriminator, and how to save the trained models.

---

## Contributing

Feel free to fork the repository, create issues, and submit pull requests for improvements.

---

Let me know if you'd like to add anything else to the README!
