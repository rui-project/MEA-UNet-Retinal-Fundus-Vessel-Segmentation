import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm
import warnings
import shutil
warnings.filterwarnings("ignore")

# saving preprocessed images and mask
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def unique_paths(paths):
    seen = set()
    unique = []
    for path in sorted(paths):
        normalized_path = os.path.normcase(os.path.abspath(path))
        if normalized_path not in seen:
            seen.add(normalized_path)
            unique.append(path)
    return unique

# loading original images, manual and FOV masks
def load_data(dataset_name):

    train_images = []
    train_masks = []
    train_fov_masks = []
    test_images = []
    test_masks = []
    test_fov_masks = []
    
    if dataset_name == "DRIVE":
        # DRIVE (including train and test datasets)
        train_image_paths = sorted(glob("DRIVE/training/images/*.tif"))
        train_mask_paths = sorted(glob("DRIVE/training/1st_manual/*.gif"))
        train_fov_paths = sorted(glob("DRIVE/training/mask/*.gif"))
        
        test_image_paths = sorted(glob("DRIVE/test/images/*.tif"))
        test_mask_paths = sorted(glob("DRIVE/test/1st_manual/*.gif"))
        test_fov_paths = sorted(glob("DRIVE/test/mask/*.gif"))
        
      
        train_image_basenames = [os.path.splitext(os.path.basename(p))[0].split("_")[0] for p in train_image_paths]
        train_mask_basenames = [os.path.splitext(os.path.basename(p))[0].split("_")[0] for p in train_mask_paths]
        
 
        print("Examples of basenames for training images:", train_image_basenames[:3])
        print("Examples of basenames for training mask:", train_mask_basenames[:3])
        
      
        adjusted_train_mask_paths = []
        for img_path in train_image_paths:
            img_base = os.path.splitext(os.path.basename(img_path))[0].split("_")[0]
            mask_path = None
            for m_path in train_mask_paths:
                m_base = os.path.splitext(os.path.basename(m_path))[0].split("_")[0]
                if img_base == m_base:
                    mask_path = m_path
                    break
            adjusted_train_mask_paths.append(mask_path)
        
       
        adjusted_test_mask_paths = []
        for img_path in test_image_paths:
            img_base = os.path.splitext(os.path.basename(img_path))[0].split("_")[0]
            mask_path = None
            for m_path in test_mask_paths:
                m_base = os.path.splitext(os.path.basename(m_path))[0].split("_")[0]
                if img_base == m_base:
                    mask_path = m_path
                    break
            adjusted_test_mask_paths.append(mask_path)
        
        train_images = train_image_paths
        train_masks = adjusted_train_mask_paths
        train_fov_masks = train_fov_paths
        
        test_images = test_image_paths
        test_masks = adjusted_test_mask_paths
        test_fov_masks = test_fov_paths
        
    elif dataset_name == "CHASEDB1":
       
        all_image_paths = sorted(glob("CHASEDB1/Image_*L.jpg") + glob("CHASEDB1/Image_*R.jpg"))
        all_mask_paths = sorted(glob("CHASEDB1/Image_*1stHO.png"))
        
        
        np.random.seed(42)
        indices = np.random.permutation(len(all_image_paths))
        split_idx = int(len(all_image_paths) * 0.8)
        
        train_indices = indices[:split_idx]
        test_indices = indices[split_idx:]
        
        train_images = [all_image_paths[i] for i in train_indices]
        train_masks = [all_mask_paths[i] for i in train_indices]
        train_fov_masks = [None] * len(train_images)
        
        test_images = [all_image_paths[i] for i in test_indices]
        test_masks = [all_mask_paths[i] for i in test_indices]
        test_fov_masks = [None] * len(test_images)
        
    elif dataset_name == "HRF":

        all_image_paths = unique_paths(glob("HRF/images/*.jpg") + glob("HRF/images/*.JPG"))
        all_mask_paths = unique_paths(glob("HRF/manual1/*.tif") + glob("HRF/manual1/*.TIF"))
        all_fov_paths = unique_paths(glob("HRF/mask/*.tif") + glob("HRF/mask/*.TIF"))

        def stem(path):
            return os.path.splitext(os.path.basename(path))[0].lower()

        mask_by_stem = {stem(path): path for path in all_mask_paths}
        fov_by_stem = {stem(path).replace("_mask", ""): path for path in all_fov_paths}

        image_to_mask = {
            img_path: mask_by_stem.get(stem(img_path))
            for img_path in all_image_paths
        }

        image_to_fov = {
            img_path: fov_by_stem.get(stem(img_path))
            for img_path in all_image_paths
        }

        missing_masks = [path for path, mask in image_to_mask.items() if mask is None]
        missing_fovs = [path for path, fov in image_to_fov.items() if fov is None]
        if missing_masks:
            print(f"Warning: {len(missing_masks)} HRF images have no matching vessel mask.")
        if missing_fovs:
            print(f"Warning: {len(missing_fovs)} HRF images have no matching FOV mask.")
        
        
        dr_images = []
        g_images = []
        h_images = []
        
        for img_path in all_image_paths:
            img_name = stem(img_path)
            if img_name.endswith("_dr"):
                dr_images.append(img_path)
            elif img_name.endswith("_g"):
                g_images.append(img_path)
            elif img_name.endswith("_h"):
                h_images.append(img_path)
        
        print(f"Number of images of DR (diabetic retinopathy): {len(dr_images)}") # 糖尿病视网膜病变
        print(f"Number of images of Glaucoma: {len(g_images)}") # 青光眼
        print(f"Number of images of Healthy Fundus: {len(h_images)}")   # 健康眼底
        
       
        total_classified = len(dr_images) + len(g_images) + len(h_images)
        if total_classified != len(all_image_paths):
            print(f"Warning: having{len(all_image_paths) - total_classified}images has not been correctly classified.")
        
       
        np.random.seed(42)
        
        def split_group(images, train_ratio=0.6):
            if not images:
                return [], []
            indices = np.random.permutation(len(images))
            split_idx = int(len(images) * train_ratio)
        
            split_idx = max(1, min(split_idx, len(images) - 1))
            return [images[i] for i in indices[:split_idx]], [images[i] for i in indices[split_idx:]]
        
  
        train_images = []
        test_images = []
        
        dr_train, dr_test = split_group(dr_images)
        g_train, g_test = split_group(g_images)
        h_train, h_test = split_group(h_images)
        
        train_images.extend(dr_train + g_train + h_train)
        test_images.extend(dr_test + g_test + h_test)
        
        print(f"Train images: {len(train_images)} (DR: {len(dr_train)}, G: {len(g_train)}, H: {len(h_train)})")
        print(f"Test images: {len(test_images)} (DR: {len(dr_test)}, G: {len(g_test)}, H: {len(h_test)})")
        
    
        intersection = set(train_images).intersection(set(test_images))
        if intersection:
            print(f"Error: train and test sets have{len(intersection)}duplicate images")
        
            train_images = [img for img in train_images if img not in intersection]
        
       
        train_masks = [image_to_mask.get(img_path) for img_path in train_images]
        train_fov_masks = [image_to_fov.get(img_path) for img_path in train_images]
        
        test_masks = [image_to_mask.get(img_path) for img_path in test_images]
        test_fov_masks = [image_to_fov.get(img_path) for img_path in test_images]
    
   
    print(f"\n{dataset_name} Dataset:")
    print(f"Number of images in the training set: {len(train_images)}")
    print(f"Number of labels in the training set: {len([p for p in train_masks if p is not None])}")
    print(f"Number of images in the test set: {len(test_images)}")
    print(f"Number of labels in the test set: {len([p for p in test_masks if p is not None])}")
    
    return train_images, train_masks, train_fov_masks, test_images, test_masks, test_fov_masks

# The Core Function
def preprocess_image(image_path, mask_path=None, fov_path=None, target_size=(584, 565), resize=True):

    try:
        if image_path.lower().endswith('.tif'):
            
            original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        else:
            original_image = cv2.imread(image_path)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"读取图像出错: {image_path}, 错误: {e}")
     
        return None, None, None, None, None
    
   
    if resize:
        image = cv2.resize(original_image, target_size)
    else:
        image = original_image.copy()
    
    
    original_mask = None
    mask = None
    if mask_path is not None:
        try:
            if mask_path.lower().endswith('.tif'):
               
                original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            elif mask_path.lower().endswith('.png') or mask_path.lower().endswith('.gif'):
                original_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
       
                if original_mask is None:
                    try:
                        from PIL import Image
                        pil_image = Image.open(mask_path)
                        original_mask = np.array(pil_image)
                        print(f"Successfully read the mask using PIL: {mask_path}")
                    except Exception as e:
                        print(f"Error reading mask using PIL: {mask_path}, Error: {e}")
            
            if original_mask is not None:
            
                if original_mask.max() > 1:
                    original_mask = (original_mask > 127).astype(np.uint8) * 255
                if resize:
                    mask = cv2.resize(original_mask, target_size, interpolation=cv2.INTER_NEAREST)
                else:
                    mask = original_mask.copy()
              
                print(f"Successfully read and processed the mask: {mask_path}, Shape: {original_mask.shape}, Adjusted shape: {mask.shape}, Value range: [{mask.min()}, {mask.max()}]")
        except Exception as e:
            print(f"Error reading annotations: {mask_path}, Error: {e}")
    
   
    fov_mask = None
    if fov_path is not None:
        try:
            if fov_path.lower().endswith('.tif'):
                fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)
            elif fov_path.lower().endswith('.gif'):
                fov_mask = cv2.imread(fov_path, cv2.IMREAD_GRAYSCALE)

                if fov_mask is None:
                    try:
                        from PIL import Image
                        pil_image = Image.open(fov_path)
                        fov_mask = np.array(pil_image)
                    except Exception as e:
                        print(f"Error reading FOV mask using PIL: {fov_path}, Error: {e}")
            
            if fov_mask is not None:
          
                if fov_mask.max() > 1:
                    fov_mask = (fov_mask > 127).astype(np.uint8) * 255
                if resize:
                    fov_mask = cv2.resize(fov_mask, target_size, interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            print(f"Error reading FOV mask: {fov_path}, Error: {e}")
    elif mask_path is not None:  
        if resize:
            h, w = target_size[::-1]
        else:
            h, w = image.shape[:2]
        center = (w // 2, h // 2)
        radius = min(w, h) // 2 - 10
        fov_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(fov_mask, center, radius, 255, -1)
    
    # Pre-processing steps
    # 1. Extract the green channel (the channel with the highest contrast).
    green_channel = image[:, :, 1]
    
    # 2. Apply CLAHE (对比度受限的自适应直方图均衡化)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(green_channel)
    
    # 3. Gaussian blur denoising
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
    
    # 4. normalized to [0,1]
    normalized = blurred / 255.0
    
    # 5. Apply FOV mask (if available)
    if fov_mask is not None:
      
        fov_mask_norm = fov_mask / 255.0
     
        normalized = normalized * fov_mask_norm
    
 
    processed_image = np.stack([normalized, normalized, normalized], axis=2)
    
    return processed_image, original_image, mask, original_mask, fov_mask

def preprocess_dataset(dataset_name, save_dir, sample_display=3):
    """
    Preprocess the entire dataset and save the results, maintaining the split between the training and test sets.
    """
    print(f"\nStarting preprocessing {dataset_name} datasets...")
    
    # Loading datasets
    train_images, train_masks, train_fov_masks, test_images, test_masks, test_fov_masks = load_data(dataset_name)
    
    # Create save directory
    dataset_save_dir = os.path.join(save_dir, dataset_name)
    if os.path.exists(dataset_save_dir):
        shutil.rmtree(dataset_save_dir)
    create_dir(os.path.join(dataset_save_dir, "train", "images"))
    create_dir(os.path.join(dataset_save_dir, "train", "masks"))
    create_dir(os.path.join(dataset_save_dir, "test", "images"))
    create_dir(os.path.join(dataset_save_dir, "test", "masks"))
    
    # Determine whether resizing is necessary
    resize = True
    if dataset_name == "HRF":
        resize = False
        print("\nNote: The HRF dataset will remain at its original size and will not be scaled.")
    
    # Process the training set
    print("\nProcessing training set...")
    for i, (image_path, mask_path, fov_path) in enumerate(tqdm(zip(train_images, train_masks, train_fov_masks), 
                                                             total=len(train_images), 
                                                             desc="Preprocess training set")):
        processed_image, original_image, mask, original_mask, fov_mask = preprocess_image(
            image_path, mask_path, fov_path, resize=resize
        )
        
        if processed_image is None:
            print(f"Failed to process training images: {image_path}")
            continue
        
        # Save the processed image
        image_filename = os.path.basename(image_path).split(".")[0] + ".png"
        cv2.imwrite(
            os.path.join(dataset_save_dir, "train", "images", image_filename),
            (processed_image * 255).astype(np.uint8)[:, :, 0]
        )
        
        # Save annotations (if any)
        if mask is not None:
            mask_filename = image_filename
            if np.any(mask):
                mask_save_path = os.path.join(dataset_save_dir, "train", "masks", mask_filename)
                cv2.imwrite(mask_save_path, mask)
                print(f"Successfully saved the training set mask: {mask_save_path}")
    
    # Processing test set
    print("\nProcessing test set...")
    for i, (image_path, mask_path, fov_path) in enumerate(tqdm(zip(test_images, test_masks, test_fov_masks), 
                                                             total=len(test_images), 
                                                             desc="Preprocess test set")):
        processed_image, original_image, mask, original_mask, fov_mask = preprocess_image(
            image_path, mask_path, fov_path, resize=resize
        )
        
        if processed_image is None:
            print(f"Failed to process test images: {image_path}")
            continue
        
        # Save the processed image
        image_filename = os.path.basename(image_path).split(".")[0] + ".png"
        cv2.imwrite(
            os.path.join(dataset_save_dir, "test", "images", image_filename),
            (processed_image * 255).astype(np.uint8)[:, :, 0]
        )
        
      
        if mask is not None:
            mask_filename = image_filename
            if np.any(mask):
                mask_save_path = os.path.join(dataset_save_dir, "test", "masks", mask_filename)
                cv2.imwrite(mask_save_path, mask)
                print(f"Successfully saved the test set mask: {mask_save_path}")
    
   
    info = {
        "dataset": dataset_name,
        "train_images": len(train_images),
        "train_masks": len([p for p in train_masks if p is not None]),
        "test_images": len(test_images),
        "test_masks": len([p for p in test_masks if p is not None])
    }
    
    with open(os.path.join(dataset_save_dir, "dataset_info.txt"), "w") as f:
        for key, value in info.items():
            f.write(f"{key}: {value}\n")
    
    print(f"{dataset_name} Dataset preprocessing complete!！\n")

def main():
   
    np.random.seed(42)
    
  
    save_dir = "preprocessed_data"
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir) 
    create_dir(save_dir)
    
    # Preprocessing DRIVE
    preprocess_dataset("DRIVE", save_dir)
    
    # Preprocessing CHASEDB1
    preprocess_dataset("CHASEDB1", save_dir)
    
    # Preprocessing HRF
    preprocess_dataset("HRF", save_dir)
    
    print("All dataset preprocessing is complete!")

if __name__ == "__main__":
    main() 
