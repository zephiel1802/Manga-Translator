import cv2
import numpy as np
import scipy.signal
from PIL import Image, ImageFilter
from collections import Counter

def make_growth_kernel(thickness: int) -> np.ndarray:
    """
    Creates an elliptical/circular convolution kernel.
    For diameter <= 5: square kernel with rounded corners.
    For larger: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ...)
    Returns float64 kernel.
    """
    diameter = thickness * 2 + 1
    if diameter <= 5:
        kernel = np.ones((diameter, diameter), dtype=np.float64)
        kernel[0, 0] = 0
        kernel[0, -1] = 0
        kernel[-1, 0] = 0
        kernel[-1, -1] = 0
        return kernel
    else:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter)).astype(np.float64)

def grow_mask_convolution(mask_array: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """
    Takes uint8 mask array and kernel.
    Pads, applies scipy.signal.convolve2d, crops back.
    Returns binary uint8 array (0 or 255).
    """
    convolved = scipy.signal.convolve2d(mask_array, kernel, mode='same', boundary='fill', fillvalue=0)
    result = np.zeros_like(mask_array, dtype=np.uint8)
    result[convolved > 0] = 255
    return result

def get_border_pixels(mask_array: np.ndarray, image_array: np.ndarray) -> np.ndarray:
    """
    Finds edge pixels of mask using PIL FIND_EDGES.
    Returns the corresponding pixel colors from image_array at those edge positions.
    If no border pixels found, returns empty array.
    """
    mask_pil = Image.fromarray(mask_array)
    edges = mask_pil.filter(ImageFilter.FIND_EDGES)
    edge_array = np.array(edges)
    
    # Border pixels are where edge_array is > 0
    y_idx, x_idx = np.where(edge_array > 0)
    
    if len(y_idx) == 0:
        return np.array([])
    
    return image_array[y_idx, x_idx]

def color_std_deviation(colors: np.ndarray) -> float:
    """
    Computes standard deviation of Euclidean distances from each color to the mean.
    Works for both grayscale (1D) and RGB (Nx3).
    """
    if len(colors) == 0:
        return 0.0
    mean_color = np.mean(colors, axis=0)
    distances = np.linalg.norm(colors - mean_color, axis=-1)
    return float(np.std(distances))

def geometric_median(points: np.ndarray, epsilon=1e-5, max_iterations=500) -> np.ndarray:
    """
    Weiszfeld's algorithm to find geometric median.
    Initial estimate = mean
    Iteratively update using inverse-distance weights.
    """
    if len(points) == 0:
        return np.array([])
    
    # Ensure points is 2D, e.g. (N, 3) for RGB or (N, 1) for grayscale
    if points.ndim == 1:
        points = points[:, np.newaxis]
        
    y = np.mean(points, axis=0)
    
    for _ in range(max_iterations):
        distances = np.linalg.norm(points - y, axis=-1)
        
        # Avoid division by zero
        zero_dist = (distances < epsilon)
        distances[zero_dist] = epsilon
        
        weights = 1.0 / distances
        weights[zero_dist] = 0
        
        weight_sum = np.sum(weights)
        if weight_sum == 0:
            break
            
        new_y = np.sum(points * weights[..., np.newaxis], axis=0) / weight_sum
        
        if np.linalg.norm(new_y - y) < epsilon:
            y = new_y
            break
        y = new_y
        
    return np.squeeze(y)

class SmartMasker:
    """
    Cleans text from speech bubbles by finding the optimal mask size and filling with the background color.
    """
    
    def clean_bubble(self, image_cv: np.ndarray, force_dark: bool = False) -> tuple:
        """
        Takes OpenCV BGR image of a bubble region.
        Returns (cleaned_image, contour, is_dark, fill_color).
        """
        gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
        
        # a. Apply Otsu threshold to get initial text mask
        _, initial_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # b. If force_dark, invert the threshold
        if force_dark:
            initial_mask = cv2.bitwise_not(initial_mask)
            
        # Clean small noise
        kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        clean_mask = cv2.morphologyEx(initial_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Determine background type
        is_dark = force_dark
        
        # c. Generate mask dilations
        min_thickness = 2
        growth_step = 2
        total_steps = 8
        
        best_mask = initial_mask.copy()
        best_border_pixels = np.array([])
        
        std_threshold = 15.0
        
        masks = []
        stds = []
        border_pixels_list = []
        
        for step in range(total_steps):
            thickness = min_thickness + step * growth_step
            kernel = make_growth_kernel(thickness)
            dilated = grow_mask_convolution(clean_mask, kernel)
            
            # d. Compute border_std_deviation
            border_pixels = get_border_pixels(dilated, image_cv)
            if len(border_pixels) > 0:
                std_dev = color_std_deviation(border_pixels)
                masks.append(dilated)
                stds.append(std_dev)
                border_pixels_list.append(border_pixels)
            else:
                break
                
        # e. Pick the largest mask whose std_dev is still below threshold OR shows significant improvement
        if len(stds) > 0:
            best_idx = 0
            
            for i in range(1, len(stds)):
                current_std = stds[i]
                prev_std = stds[i-1]
                
                if current_std <= std_threshold:
                    best_idx = i
                elif prev_std > 0 and (prev_std - current_std) / prev_std > 0.20:
                    best_idx = i
                    
            best_mask = masks[best_idx]
            best_border_pixels = border_pixels_list[best_idx]
            
        # f. Compute median fill color from border pixels of best mask
        if len(best_border_pixels) > 0:
            fill_color = geometric_median(best_border_pixels)
            if isinstance(fill_color, np.ndarray) and fill_color.size >= 3:
                fill_color = tuple(map(int, fill_color[:3]))
            elif isinstance(fill_color, (int, float, np.number)):
                c = int(fill_color)
                fill_color = (c, c, c)
            else:
                c = int(fill_color[0])
                fill_color = (c, c, c)
        else:
            # Fallback
            fill_color = (255, 255, 255) if not is_dark else (0, 0, 0)
            
        # g. Fill the best mask area with the median color
        cleaned_image = image_cv.copy()
        cleaned_image[best_mask > 0] = fill_color
        
        # h. Create contour from the filled area
        contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            contour = max(contours, key=cv2.contourArea)
        else:
            contour = np.array([])
            
        # i. Return
        return cleaned_image, contour, is_dark, fill_color
