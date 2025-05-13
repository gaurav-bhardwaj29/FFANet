import cv2
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from skimage.metrics import structural_similarity as ssim
total_ssim = 0.0
count = 0
import image_dehazer

# Check for Metal support (Apple GPU)
def check_metal_support():
    build_info = cv2.getBuildInformation()
    has_metal = "Metal:" in build_info and "YES" in build_info.split("Metal:")[1].split("\n")[0]
    return has_metal

def load_dataset(dataset_path):
    """
    Loads images from the specified dataset folder.
    Supported image formats: .jpg, .png.
    """
    image_files = glob.glob(os.path.join(dataset_path, "*.jpg")) + \
                  glob.glob(os.path.join(dataset_path, "*.png"))
    print(f"Found {len(image_files)} images")
    return image_files  # Return only the file paths, load images later

def compute_luminance(I):
    B, G, R = cv2.split(I.astype(np.float32))
    H = 0.299 * R + 0.587 * G + 0.114 * B
    return H

def contrast_enhancement(I, b_factor=2.0):
    H = compute_luminance(I)
    l = np.mean(H)
    channels = cv2.split(I.astype(np.float32))
    enhanced_channels = []
    for ch in channels:
        enhanced = b_factor * (ch - l) + l
        enhanced = np.clip(enhanced, 0, 255)
        enhanced_channels.append(enhanced.astype(np.uint8))
    enhanced_img = cv2.merge(enhanced_channels)
    return enhanced_img

def estimate_transmission(I, k=1.0):
    window_size = 15
    dark_channel = cv2.erode(cv2.min(cv2.min(I[:,:,0], I[:,:,1]), I[:,:,2]),
                             np.ones((window_size, window_size), np.uint8))
    T = np.exp(-k * (dark_channel / 255.0))
    T = np.clip(T, 0.1, 1.0)
    return T

def fast_visibility_recovery(I, T, A1=None):
    window_size = 15
    dark_channel = cv2.erode(cv2.min(cv2.min(I[:,:,0], I[:,:,1]), I[:,:,2]),
                             np.ones((window_size, window_size), np.uint8))
    num_pixels = dark_channel.size
    num_bright = max(1, int(num_pixels * 0.001))
    bright_indices = np.argsort(dark_channel.ravel())[-num_bright:]
    A1 = np.max(I.reshape(-1,3)[bright_indices], axis=0)

    X = np.zeros_like(I, dtype=np.float32)
    for i in range(3):
        X[:,:,i] = A1[i] * (1 - T)

    I_float = I.astype(np.float32)
    denominator = 1 - (X[:,:,0] / (A1[0] + 1e-6))
    denominator = np.clip(denominator, 0.1, None)

    F = (I_float - X) / denominator[:,:,None]
    F = np.clip(F, 0, 255).astype(np.uint8)
    return F, A1

def joint_bilateral_filter(I, d=9, sigmaColor=75, sigmaSpace=75):
    I_float = I.astype(np.float32)
    filtered = cv2.bilateralFilter(I_float, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return filtered.astype(np.uint8)

def luminance_blending(J, L, P=15, Q=236):
    J_ycc = cv2.cvtColor(J, cv2.COLOR_BGR2YCrCb)
    L_ycc = cv2.cvtColor(L, cv2.COLOR_BGR2YCrCb)
    Y_J, Cr_J, Cb_J = cv2.split(J_ycc)
    Y_L, Cr_L, Cb_L = cv2.split(L_ycc)

    GJ = cv2.GaussianBlur(Y_J, (15,15), 0)
    GL = cv2.GaussianBlur(Y_L, (15,15), 0)

    weight = np.where(GL < 128, 1.0, 0.5)

    Y_blend = (weight * Y_J + (1 - weight) * Y_L).astype(np.float32)

    Y_min = np.min(Y_blend)
    Y_max = np.max(Y_blend)
    Y_stretched = P + (Y_blend - Y_min) * (Q - P) / (Y_max - Y_min + 1e-6)
    Y_stretched = np.clip(Y_stretched, 0, 255).astype(np.uint8)

    Cr_blend = ((Cr_J.astype(np.float32) + Cr_L.astype(np.float32)) / 2).astype(np.uint8)
    Cb_blend = ((Cb_J.astype(np.float32) + Cb_L.astype(np.float32)) / 2).astype(np.uint8)

    blended_ycc = cv2.merge([Y_stretched, Cr_blend, Cb_blend])
    blended_rgb = cv2.cvtColor(blended_ycc, cv2.COLOR_YCrCb2BGR)
    return blended_rgb

def dehaze_frame(I, b_factor=2.0, k=1.0):
    L = contrast_enhancement(I, b_factor=b_factor)
    T = estimate_transmission(I, k=k)
    F, A1 = fast_visibility_recovery(I, T, None)
    J = joint_bilateral_filter(F)
    output = luminance_blending(J, L)
    return output

def process_image(args):
    """Process a single image for multiprocessing"""
    img_path, output_path, b_factor, k = args
    
    # Read image
    img = cv2.imread(img_path)
    if img is None:
        return False
    
    # Process image
    dehazed = dehaze_frame(img, b_factor, k)
    
    # Save result
    cv2.imwrite(output_path, dehazed)
    return True

if __name__ == "__main__":
    # Get the user's home directory for Mac-friendly paths
    home_dir = os.path.expanduser("~")
    
    # Example: dataset in Documents folder - modify as needed
    dataset_path = os.path.join(home_dir, "Desktop/pur")
    output_folder = os.path.join(home_dir, "Desktop/dehazed1")
    
    # Create output folder
    os.makedirs(output_folder, exist_ok=True)
    
    # Parameters for dehazing
    b_factor = 2.0  # Contrast enhancement factor
    k = 1.0         # Transmission estimate factor
    
    # Check if OpenCV supports Metal
    has_metal = check_metal_support()
    if has_metal:
        print("Metal GPU acceleration is available and will be used!")
        # Enable OpenCV Metal backend
        cv2.ocl.setUseOpenCL(True)
    else:
        print("Metal GPU acceleration not available. Using CPU only.")
    
    # Load image paths
    image_files = load_dataset(dataset_path)
    
    if not image_files:
        print("No images found in the dataset path. Please check the path and image files.")
    else:
        # Number of images to show as examples
        show_limit = 5
        
        # Prepare arguments for multiprocessing
        process_args = []
        for idx, img_path in enumerate(image_files):
            # Create Mac-friendly output path
            # Assuming img and dehazed_img are NumPy arrays in BGR or RGB format
            
            img = cv2.imread(img_path)
            if img is None:
                continue    
            dehazed_img, _ = image_dehazer.remove_haze(img)
            score = ssim(img, dehazed_img, channel_axis=2)   # compare color images
            total_ssim += score
            count += 1

            filename = os.path.basename(img_path)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_folder, f"dehazed_{base_name}.png")
            
            # Store arguments
            process_args.append((img_path, output_path, b_factor, k))
        if count > 0:
            print(f"Average SSIM: {total_ssim/count:.4f}")

        # Show a few example results
        print("\nProcessing sample images for preview...")
        for i in range(min(show_limit, len(image_files))):
            img_path = image_files[i]
            img = cv2.imread(img_path)
            if img is not None:
                # dehazed = dehaze_frame(img, b_factor, k)
                
                dehazed, _ = image_dehazer.remove_haze(img)
                # Display the results
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                dehazed_rgb = cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB)
                
                fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                axes[0].imshow(img_rgb)
                axes[0].set_title(f"Original {i+1}")
                axes[0].axis("off")
                axes[1].imshow(dehazed_rgb)
                axes[1].set_title(f"Dehazed {i+1}")
                axes[1].axis("off")
                plt.tight_layout()
                plt.show()
        
        # Process all images with multiprocessing
        print(f"\nProcessing all {len(image_files)} images in parallel...")
        
        # Use number of cores minus 1 to keep system responsive
        num_processes = max(1, multiprocessing.cpu_count() - 1)
        print(f"Using {num_processes} processes")
        
        # Process images in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            results = list(tqdm(executor.map(process_image, process_args), 
                               total=len(process_args), 
                               desc="Dehazing Images"))
        
        # Report results
        successful = sum(1 for r in results if r)
        print(f"\nProcessing complete! Successfully processed {successful} out of {len(image_files)} images.")
        print(f"Dehazed images saved to: {output_folder}")




