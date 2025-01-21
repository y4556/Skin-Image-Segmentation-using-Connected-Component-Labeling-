# Skin Image Segmentation using Connected Component Labeling
<img src="https://github.com/user-attachments/assets/b7fe9638-62e8-4ea7-8132-a689659663db?raw=true" alt="Skin Tissue" width="300"/> <img src="https://github.com/y4556/Skin-Image-Segmentation-using-Connected-Component-Labeling-/blob/main/1.%5B512x4864%5D.png?raw=true" alt="Tissue Mask" width="300"/>


## Abstract  
This project implements a method for segmenting tissue types in medical images using **color-based segmentation** and **Connected Component Analysis (CCA)**. The segmentation focuses on accurately identifying layers such as the dermis (DRM), epidermis (EPI), dermal-epidermal junction (DEJ), and keratin (KER). The segmentation accuracy is evaluated using the Dice coefficient.  
## Technologies Used  

- **NumPy**: Numerical operations and array manipulation.  
- **OpenCV**: Image processing tasks like reading, editing, and saving images.  
- **Matplotlib**: Visualization of images and results.
- **OS**: File and folder operations.  

## Features  

- **Color-Based Segmentation**: Defines intensity ranges (V_SET) for accurate tissue segmentation.  
- **Connected Component Labeling**: Labels connected components in segmented images.  
- **Dice Coefficient**: Evaluates segmentation accuracy.  
- **Visualization**: Displays original images, ground truth masks, and segmented results.  

## Code Overview  

### Key Functions  

1. **`in_v_set(pixel, v_set)`**  
   Checks if a pixel belongs to a specific intensity range.  

2. **`CCA_eight(v_set, image, color)`**  
   Performs eight-connected component analysis to label connected regions.  

3. **`color_largest_component(mask, label)`**  
   Identifies and colors the largest connected component in a binary mask.  

4. **`creating_masks(img, color, v_set)`**  
   Generates a mask for each tissue type using CCA and color assignment.  

5. **`masks(img)`**  
   Iterates over predefined tissue types, generating masks for each and displaying results.  

6. **`merge_masks(masks)`**  
   Combines multiple masks into a single mask using bitwise operations.  

7. **`dice_coefficient(merged_mask, original_mask)`**  
   Computes the Dice coefficient to evaluate segmentation accuracy.  
