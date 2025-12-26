import cv2
import numpy as np
import os
from typing import Tuple, Optional

def apply_fixed_mask_inpaint(
    image_path: str,
    mask_path: str = 'bin_mask.png',
    output_path: str = 'output_fixed_mask.jpg',
    mask_offset: Tuple[int, int] = (7, 7),
    debug: bool = False,
    highlight_only: bool = False
) -> None:
    """
    Applies a fixed mask inpainting to an image to remove watermarks or specific regions.

    The function loads a source image and a binary mask. It calculates the position of the mask
    relative to the bottom-right corner of the image (with a fixed padding), creates a full-size
    mask, and applies OpenCV's Telea inpainting algorithm.

    Args:
        image_path (str): Path to the input image file.
        mask_path (str): Path to the binary mask image file (default: 'bin_mask.png').
        output_path (str): Path to save the resulting image (default: 'output_fixed_mask.jpg').
        mask_offset (Tuple[int, int]): (x, y) offset to fine-tune the mask position (default: (7, 7)).
        debug (bool): If True, displays visualization windows of the process (default: False).
        highlight_only (bool): If True, draws a red rectangle around the watermark area instead of removing it.
    """
    
    # --- 1. Load Resources ---
    if not os.path.exists(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return
    if not os.path.exists(mask_path):
        print(f"Error: Mask '{mask_path}' not found.")
        return

    img = cv2.imread(image_path)
    mask_img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"Error: Failed to load image '{image_path}'.")
        return
    if mask_img is None:
        print(f"Error: Failed to load mask '{mask_path}'.")
        return

    # --- 2. Setup Dimensions ---
    img_h, img_w = img.shape[:2]
    mask_h, mask_w = mask_img.shape[:2]
    
    # Fixed padding from the bottom-right corner
    padding = 69 

    # --- 3. Calculate Position ---
    # Position relative to bottom-right corner: X = Width - Padding - Mask Width + Offset X
    x = img_w - padding - mask_w + mask_offset[0]
    y = img_h - padding - mask_h + mask_offset[1]

    # Bounds check
    if x < 0 or y < 0 or x + mask_w > img_w or y + mask_h > img_h:
        print(f"Error: Mask position ({x}, {y}) is out of global image bounds ({img_w}x{img_h}).")
        return

    print(f"Applying mask at coordinates: ({x}, {y})")

    # --- 4. Create Global Mask ---
    # Create the full-size mask (black background)
    full_mask = np.zeros((img_h, img_w), dtype="uint8")
    
    # Threshold the local mask to ensure binary (0 or 255)
    _, binary_local_mask = cv2.threshold(mask_img, 127, 255, cv2.THRESH_BINARY)
    
    # Place the local mask into the full-size mask
    full_mask[y:y+mask_h, x:x+mask_w] = binary_local_mask

    # --- 5. Apply Inpainting OR Highlighting ---
    if highlight_only:
        # Draw a red rectangle around the calculated mask area
        result = img.copy()
        cv2.rectangle(result, (x, y), (x + mask_w, y + mask_h), (0, 0, 255), 5)
        # Also draw the mask itself in semi-transparent red to see exact pixels
        overlay = result.copy()
        overlay[y:y+mask_h, x:x+mask_w, 2] = np.maximum(overlay[y:y+mask_h, x:x+mask_w, 2], binary_local_mask)
        cv2.addWeighted(overlay, 0.5, result, 0.5, 0, result)
        print("Debug: Highlighted watermark area instead of removing it.")
    else:
        # Radius of 3 pixels is standard for removing small text/watermarks
        inpaint_radius = 3
        result = cv2.inpaint(img, full_mask, inpaint_radius, cv2.INPAINT_TELEA)

    # --- 6. Save Output ---
    cv2.imwrite(output_path, result)
    print(f"Success: Saved result to '{output_path}'")

    # --- 7. Debug Visualization (Optional) ---
    if debug:
        _show_debug_visualization(img, full_mask, result, x, y, mask_w, mask_h, binary_local_mask)

def _show_debug_visualization(
    original: np.ndarray, 
    full_mask: np.ndarray, 
    result: np.ndarray, 
    x: int, y: int, w: int, h: int, 
    local_mask: np.ndarray
) -> None:
    """Helper function to show debug windows."""
    # Overlay region visualization
    preview = original.copy()
    
    # 1. Draw bounding box
    cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
    # 2. Highlight mask area in green
    overlay_color = np.zeros_like(original)
    overlay_color[y:y+h, x:x+w, 1] = local_mask # Green channel
    
    cv2.addWeighted(preview, 0.7, overlay_color, 0.3, 0, preview)

    cv2.imshow("Debug: Original with Overlay", preview)
    cv2.imshow("Debug: Full Mask", full_mask)
    cv2.imshow("Debug: Final Result", result)
    
    print("Press any key to close debug windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Example usage
    # Place your image in the 'input' folder and name it 'image.jpg', or change the path below.
    INPUT_FILE = "input/image.jpg"
    MASK_FILE = "bin_mask.png"
    OUTPUT_FILE = "output/cleaned_image.jpg"
    
    if os.path.exists(INPUT_FILE):
        # Set debug=True to see the windows, debug=False for headless execution
        apply_fixed_mask_inpaint(INPUT_FILE, mask_path=MASK_FILE, output_path=OUTPUT_FILE, debug=False)
    else:
        print(f"Welcome to WM Remover!")
        print(f"To test, place an image at '{INPUT_FILE}' and run this script again.")