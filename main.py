import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
def in_v_set(pixel, v_set):
    b, g, r = pixel  # Extracting the B, G, R channels
    for k, v in v_set.items():
        if k == 'B':
            b_range = v
        elif k == 'G':
            g_range = v
        elif k == 'R':
            r_range = v
    if b_range[0] <= b <= b_range[1] and g_range[0] <= g <= g_range[1] and r_range[0] <= r <= r_range[1]:
        return True
    return False
def CCA_eight(v_set, image, color):
    img = image
    mask = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    r, c, _ = image.shape
    label = color

    for i in range(r):
        for j in range(c):
            if i == 0 and j == 0: #top left corner
                if in_v_set(img[i, j], v_set):
                    mask[i, j] = label
            elif i == 0 and j > 0:#top row
                if in_v_set(img[i, j], v_set):
                    if in_v_set(img[i, j - 1],v_set):
                        mask[i, j] = label
            elif i > 0 and j == 0:#left column
                if in_v_set(img[i, j], v_set):
                    up = img[i - 1, j]
                    up_right = img[i - 1, j + 1]
                    if in_v_set(up, v_set) or in_v_set(up_right, v_set):
                        mask[i, j] = label
            elif i>0 and j==c-1:#right column
                if in_v_set(img[i,j],v_set):
                   left=img[i,j-1]
                   up=img[i-1,j]
                   up_left=img[i-1,j-1]
                   if in_v_set(up,v_set) or in_v_set(left,v_set) or in_v_set(up_left,v_set):
                          mask[i,j]=label
            elif i > 0 and j > 0:#rest of the image
                if in_v_set(img[i, j], v_set):
                    up = mask[i - 1, j]
                    left = mask[i, j - 1]
                    up_left = mask[i - 1, j - 1]
                    up_right = mask[i - 1, j + 1]
                    neighbors = [up, left, up_left, up_right]
                    neighbors = [neighbor for neighbor in neighbors if np.any(neighbor != 0)]
                    if len(neighbors) > 0:
                        min_neighbor = min([np.min(neighbor) for neighbor in neighbors])
                        mask[i, j] = min_neighbor
                    else:
                        mask[i, j] = label
    return mask, label
def color_largest_component(mask, label):
    num_labels, labeled_mask = cv.connectedComponents(mask)
    areas = [np.sum(labeled_mask == label) for label in range(1, num_labels)]
    largest_label = np.argmax(areas) + 1
    largest_component_mask = np.where(labeled_mask == largest_label, label, 0)
    largest_component_mask = np.uint8(largest_component_mask)
    return largest_component_mask
def color_mask(mask, color):
    colored_mask = np.ones((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            if mask[i, j] == 255:
                colored_mask[i, j] = color
    return colored_mask
def creating_masks(img,color,v_set):
    proccessed_mask,l=CCA_eight(v_set,img,255)
    colored_mask1 = color_largest_component(proccessed_mask, 255)
    colored_mask=color_mask(proccessed_mask,color)
    return colored_mask
def masks(img):
    #Defining V_SET
    v_sets = {
        'BKG': {'B': (240, 255), 'G': (240, 255), 'R': (240, 255.0)},
        'DEJ': {'B': (100.86, 250.92), 'G': (100.13, 221.19), 'R': (221.52, 255.0)},
        'EPI': {'B': (100, 255), 'G': (0, 200), 'R': (150, 255)},
        'DRM': {'B': (178.21, 255), 'G': (138.12, 250.45), 'R': (220.55, 255.0)},
        'KER': {'B': (30, 200), 'G': (40, 200), 'R': (80, 180.5)}
    }
    tissue_colors = {
    'DEJ': [255, 172, 255],
    'DRM': [0, 255,190],
    'EPI': [160, 48, 112],
    'KER': [224, 224, 224],
    'BKG': [255, 255, 255]
    }
    masks=[]
    for tissue_type in v_sets:
        mask = creating_masks(img, tissue_colors[tissue_type], v_sets[tissue_type])
        masks.append(mask)
    for mask in masks:
        plt.imshow(cv.cvtColor(mask, cv.COLOR_BGR2RGB))
        plt.show()
    return masks
def merge_masks(masks):
    merged_mask = np.zeros_like(masks[0])
    for i,mask in enumerate(masks):
        if i == 0:
        #mask = np.bitwise_not(mask)
            continue
        N_black_pixels = np.any(mask != [0, 0, 0], axis=-1)
        merged_mask[N_black_pixels] |= mask[N_black_pixels]
    return merged_mask


def dice_coefficient(gen_mask, org_mask):
    cv.imshow("Generated Mask", gen_mask)
    cv.imshow("Original Mask", org_mask)
    cv.waitKey()
    true_pixels = 0
    false_pixels = 0

    # Loop through each pixel
    for i in range(gen_mask.shape[0]):
        for j in range(gen_mask.shape[1]):
            # Get the label of the generated and original masks
            gen_label = get_label({
                'B': gen_mask[i, j, 0],
                'G': gen_mask[i, j, 1],
                'R': gen_mask[i, j, 2]
            })
            org_label = get_label({
                'B': org_mask[i, j, 0],
                'G': org_mask[i, j, 1],
                'R': org_mask[i, j, 2]
            })

            if gen_label == org_label:
                true_pixels += 1
            else:
                false_pixels += 1

    total_true_pixels = np.count_nonzero(org_mask != 0)
    print(true_pixels)
    dice_coeff = (2.0 * true_pixels) / total_true_pixels
    # Print Dice coefficient
    print("Dice Coefficient for these two masks is:", dice_coeff)

def get_label(pixel_color):
    tissue_colors = {
        'DEJ': {'B': (254, 256), 'G': (171, 173), 'R': (254, 256)},
        'DRM': {'B': (0, 256), 'G': (254, 256), 'R': (189, 191)},
        'EPI': {'B': (159, 161), 'G': (47, 49), 'R': (111, 113)},
        'KER': {'B': (223, 225), 'G': (223, 225), 'R': (223, 225)},
        'BKG': {'B': (254, 256), 'G': (254, 256), 'R': (254, 256)}
    }

    for label, color_range in tissue_colors.items():
        for channel in ['B', 'G', 'R']:
            in_range = color_range[channel][0] <= pixel_color[channel] <= color_range[channel][1]
            if not in_range:
                break
        else:
            return label
    return 'Unknown'

#Main
output_folder = 'Assignment #1/Test/Output/'
img = cv.imread(r"Test/tissue/RA23-01882-A1-1-PAS.[14848x1024].jpg")
or_mask=cv.imread("Test/Mask/RA23-01882-A1-1.[14848x1024].png")
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
plt.show()
#Calling the function to create masks
masking=masks(img)
merged_masks=merge_masks(masking)
plt.imshow(cv.cvtColor(merged_masks, cv.COLOR_BGR2RGB))
plt.title('Merged Mask with Black Background')
plt.show()
if not os.path.exists(output_folder):
   os.makedirs(output_folder)
output_path = os.path.join(output_folder, "Generated_Mask.png")
cv.imwrite(output_path, merged_masks)
dice_coefficient(merged_masks,or_mask)