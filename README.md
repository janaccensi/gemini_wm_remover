# 🧹 Gemini Watermark Remover

A lightweight, efficient Python tool to automatically remove watermarks from images using a fixed binary mask. Right now it's using the gemini binary mask, but it can use any mask you want. This tool utilizes OpenCV's inpainting algorithms to seamlessly clean your images.

## ✨ Features

- **Automated Removal**: Uses a pre-defined mask to target watermark locations.
- **Smart Inpainting**: Applies OpenCV's Telea algorithm for natural-looking restoration.
- **Debug Mode**: Visualizes the mask overlay and processing steps.
- **Highlight Mode**: Option to simply highlight the detected area without modifications.

## 🚀 Getting Started

### Prerequisites

Ensure you have Python installed. Then, install the required dependencies:

```bash
pip install -r requirements.txt
```

### 📂 Directory Structure

The project is structured for simplicity:

```
.
├── Input/              # Place your source images here
├── Output/             # Processed images will appear here
├── bin_mask.png        # The binary mask used for watermark detection
├── wm_remover.py       # Main script
└── requirements.txt    # Dependencies
```

### 🛠️ Usage

1.  Place the image you want to clean in the `input` folder and name it `image.jpg` (or update the script to match your filename).
2.  Run the script:

    ```bash
    python wm_remover.py
    ```

3.  Check the `output` folder for your cleaned image!

## ⚙️ Customization

You can adjust the mask position by modifying the `padding` and `mask_offset` variables in the `wm_remover.py` file to perfectly align with specific watermarks.

## 📝 License

Free to use for your projects. Enjoy clean images!
