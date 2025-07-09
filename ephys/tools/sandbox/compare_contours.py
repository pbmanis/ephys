import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.pyplot as mpl
# ======================================
# Step 1: Generate synthetic shapes
# ======================================
def generate_square(size=1, n_points=100):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = size * np.cos(theta)
    y = size * np.sin(theta)
    # Make it a square by taking max(abs(x), abs(y))
    x_square = np.sign(x) * size
    y_square = np.sign(y) * size
    return np.column_stack((x_square, y_square))


def generate_circle(size=1, n_points=100):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    x = size * np.cos(theta)
    y = size * np.sin(theta)
    return np.column_stack((x, y))


def generate_star(size=1, n_points=100, n_points_star=6):
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    # Star with alternating radius
    r = np.array([size if i % 2 == 0 else size * 0.4 for i in range(n_points_star * 2)])
    r_interp = interp1d(
        np.linspace(0, 2 * np.pi, len(r), endpoint=False),
        r,
        kind="nearest",
        fill_value="extrapolate",
    )
    x = r_interp(theta) * np.cos(theta)
    y = r_interp(theta) * np.sin(theta)
    return np.column_stack((x, y))

def generate_pentagon(size=1, n_points=100):
    x = [0.2, 0.8, 1.0, 0.5, 0.0, 0.2]
    y = [0., 0., 0.5, 1.0, 0.5, 0.0]
    return np.column_stack((x, y))

def arc(x0, y0, radius, start_angle, end_angle, n_points=100):
    theta = np.linspace(start_angle, end_angle, n_points)
    x = x0 +  radius * np.cos(theta)
    y = y0 +  radius * np.sin(theta)
    return np.array([x, y])

def line(x0, y0, x1, y1, n_points=100):
    x = np.linspace(x0, x1, n_points)
    y = np.linspace(y0, y1, n_points)
    return np.array([x, y])

def generate_rounded_square(size=1, n_points=100, corner_radius=0.2):
   theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
   x, y = [], []
   for t in theta:
       if np.pi/4 <= t % (np.pi/2) < 3*np.pi/4:
           # Round corners
           x_val = size * np.sign(np.cos(t)) * (1 - corner_radius)
           y_val = size * np.sign(np.sin(t)) * (1 - corner_radius)
           x.append(x_val + corner_radius * np.cos(t))
           y.append(y_val + corner_radius * np.sin(t))
       else:
           # Flat edges
           x.append(size * np.cos(t))
           y.append(size * np.sin(t))
   return np.column_stack((x, y))

def generate_rounded_square2(size=1, n_points=1000, corner_radius=0.2):
    x0 = corner_radius
    x1 = size - corner_radius
    y0 = corner_radius
    y1 = size - corner_radius
    x = np.empty((2, 0), dtype=float)
    x = np.append(x, arc(x0, y1, corner_radius, np.pi, np.pi/2., n_points=n_points//4), axis=1)

    topline = line(x[0,-1], size, x1, size, n_points=n_points//4)
    # mpl.plot(x[0, -1], x[1, -1], 'ro')
    x = np.concatenate((x, topline), axis=1)
    # mpl.plot(x[0, -1], x[1, -1], color='orange', marker='s')
    x = np.concatenate((x, arc(x[0,-1], x[1,-1]-corner_radius, corner_radius, np.pi/2., 0, n_points=n_points//4)), axis=1)
    # mpl.plot(x[0, -1], x[1, -1], color='yellow', marker='o')
    x = np.concatenate((x, line(x[0, -1], x[1, -1] - corner_radius, x[0,-1], y0, n_points=n_points//4)), axis=1)
    # mpl.plot(x[0, -1], x[1, -1], color='green', marker='s')
    x = np.concatenate((x, arc(x[0,-1]-corner_radius, x[1, -1], corner_radius, 0, -np.pi/2., n_points=n_points//4)), axis=1)   
    # mpl.plot(x[0, -1], x[1, -1], color='blue', marker='o')
    x = np.concatenate((x, line(x[0, -1] - corner_radius, x[1, -1], x0, x[1, -1], n_points=n_points//4)), axis=1)
    # mpl.plot(x[0, -1], x[1, -1], color='cyan', marker='s')
    x = np.concatenate((x, arc(x[0, -1], x[1, -1]+corner_radius, corner_radius, -np.pi/2, -np.pi, n_points=n_points//4)), axis=1)
    # mpl.plot(x[0, -1], x[1, -1], color='magenta', marker='o')
    x = np.concatenate((x, line(x[0, -1], x[1, -1], x[0, -1], x[1, -1] +y1-corner_radius, n_points=n_points//4)), axis=1)
    # mpl.plot(x[0, -1], x[1, -1], color='indigo', marker='s')
    # mpl.gca().set_aspect('equal', adjustable='box')
    # mpl.plot(x[0], x[1], 'k-', linewidth=0.2)
    # mpl.show()
    # exit()
    return x


"""
### 1. **Preprocessing: Contour Extraction & Resampling**
 - **Extract contours**: For each shape (target and references), obtain the outer contour as an ordered list of points.
 - **Ensure consistent orientation**: Make all contours counter-clockwise (e.g., using the shoelace formula to check/correct).
 - **Resample contours**: Use arc-length parameterization to represent each contour with `N` equidistant points (e.g., `N=128`):
"""


def resample_contour(contour, N: int):
    # Calculate cumulative arc lengths
    distances = [0]
    for i in range(1, len(contour)):
        dx = contour[i][0] - contour[i - 1][0]
        dy = contour[i][1] - contour[i - 1][1]
        distances.append(distances[-1] + (dx**2 + dy**2) ** 0.5)
    total_length = distances[-1]

    # Interpolate N equidistant points
    new_contour = []
    step = total_length / (N - 1)
    current_distance = 0
    j = 0
    for _ in range(N):
        while j < len(distances) - 1 and distances[j + 1] < current_distance:
            j += 1
        if j == len(distances) - 1:
            new_contour.append(contour[-1])
        else:
            ratio = (current_distance - distances[j]) / (distances[j + 1] - distances[j])
            x = contour[j][0] + ratio * (contour[j + 1][0] - contour[j][0])
            y = contour[j][1] + ratio * (contour[j + 1][1] - contour[j][1])
            new_contour.append([x, y])
        current_distance += step
    return new_contour


"""### 2. **Normalization: Translation, Scale, and Rotation**
 - **Translation**: Center contours at the origin by subtracting the centroid.
 - **Scale**: Normalize contours to have unit RMS distance from the centroid.
 - **Rotation**: Align contours using the principal axis (first eigenvector of the point covariance matrix):
   ```python
"""


def normalize_contour(contour):
    # Center
    centroid = np.mean(contour, axis=0)
    centered = contour - centroid

    # Scale
    rms = np.sqrt(np.mean(np.linalg.norm(centered, axis=1) ** 2))
    scaled = centered / rms if rms > 0 else centered

    # Rotation (using PCA)
    cov = np.cov(scaled.T)
    _, eigvecs = np.linalg.eigh(cov)
    principal_axis = eigvecs[:, -1]  # Largest eigenvector
    angle = np.arctan2(principal_axis[1], principal_axis[0])
    rot_matrix = np.array([[np.cos(-angle), -np.sin(-angle)], [np.sin(-angle), np.cos(-angle)]])
    rotated = scaled @ rot_matrix
    return rotated


# """### 3. **Shape Descriptor: Fourier Transform Magnitudes**
#  - **Convert to complex numbers**: Represent each normalized contour as complex numbers \(s[k] = x[k] + iy[k]\).
#  - **Compute DFT**: Apply the Discrete Fourier Transform (DFT) to \(s[k]\).
#  - **Extract magnitudes**: Use the magnitudes of the first `K` non-DC coefficients (e.g., `K=20`).
#  - **Normalize magnitudes**: Divide by the magnitude of the first harmonic (or the first non-zero harmonic):
#    ```python
# """


def fourier_descriptor(contour, K=20):
    complex_contour = contour[:, 0] + 1j * contour[:, 1]
    fft_result = np.fft.fft(complex_contour)
    magnitudes = np.abs(fft_result[1 : K + 1])  # Skip DC (index 0)
    if magnitudes[0] > 1e-10:
        descriptor = magnitudes / magnitudes[0]
    else:
        # Fallback: find first non-zero harmonic
        non_zero_idx = next((i for i, m in enumerate(magnitudes) if m > 1e-10), 0)
        descriptor = magnitudes / magnitudes[non_zero_idx] if non_zero_idx < K else magnitudes
    return descriptor


# """
# 4. **Comparison & Interpolation**
#  - **Compute descriptors**: Generate descriptors for all reference shapes and the target.
#  - **Projection onto reference segments**: For each consecutive pair of reference descriptors \(D_i\) and \(D_{i+1}\):
#    - Find the projection \(t\) of the target descriptor \(P\) onto the line segment \(D_i \rightarrow D_{i+1}\):
#      \[
#      t = \frac{(P - D_i) \cdot (D_{i+1} - D_i)}{\|D_{i+1} - D_i\|^2}, \quad t \in [0, 1]
#      \]
#    - Compute the projected point \(Q = D_i + t(D_{i+1} - D_i)\).
#    - Calculate distance \(d = \|P - Q\|\).
#  - **Select best segment**: Choose the segment \((i, i+1)\) with minimal \(d\).
#  - **Interpolated position**: The target lies between references \(i\) and \(i+1\) at fractional position \(t\).
# """

N = 128  # Resample points
K = 20  # Fourier harmonics

# Generate shapes
references = [ generate_star(), generate_square(), generate_circle(), generate_pentagon()]
target = generate_rounded_square()
refnames = ['Star', 'Square', 'Circle', 'Pentagon']
# Resample and normalize all shapes
ref_processed = [normalize_contour(resample_contour(ref, N=N)) for ref in references]
target_processed = normalize_contour(resample_contour(target, N=N))


# Generate descriptors
ref_descriptors = []
for i_ref, ref in enumerate(references):
    # contour = extract_contour(ref)
    contour = ref_processed[i_ref]  # Use preprocessed contour
    # resampled = resample_contour(contour, N)
    # normalized = normalize_contour(resampled)
    fd = fourier_descriptor(contour, K)
    ref_descriptors.append(fd)

# Compute target descriptor
# target_contour = extract_contour(target)
target_contour = target_processed
# target_resampled = resample_contour(target_contour, N)
# target_normalized = normalize_contour(target_resampled)
target_fd = fourier_descriptor(target_contour, K)

# Find best matching segment and interpolate
min_dist = float("inf")
best_i = 0
best_t = 0
import matplotlib.pyplot as mpl
f, ax = mpl.subplots(1, 1, figsize=(10, 6))
# this assumes the descriptors are ordered and the target is between two references
for i, ref_d in enumerate(ref_descriptors):
    # Di, Di1 = ref_descriptors[i], ref_descriptors[i + 1]
    # v = Di1 - Di
    print(f"\n{refnames[i]}")
    print(target_fd)
    print(ref_d)
    ax.plot(np.arange(len(ref_d))[:], ref_d[:], 'o-', label=f'Reference {refnames[i]}')
    if i == 0:
        ax.plot(np.arange(len(target_fd))[:], target_fd[:], 'k--', label='Target (Rounded Square)')
    w = target_fd - ref_d
    # t_val = np.dot(w, v) / np.dot(v, v)
    # t_clamped = max(0, min(0, t_val))
    # Q = Di + t_clamped * v
    dist = np.linalg.norm(w[:])
    print(i, refnames[i], dist)   # print(i, t_val, dist)
    if dist < min_dist:
        min_dist = dist
        best_i = i
        # best_t = t_clamped
        # print(i, t_val, t_clamped, dist)
ax.legend()
mpl.show()
# Result: Target is between references[best_i] and references[best_i+1]
interpolated_position = best_i + best_t

# best_i, best_t = find_interpolation_position(target_descriptor, ref_descriptors)

# ======================================

# Step 5: Visualize results
# ======================================
plt.figure(figsize=(12, 8))

# Plot reference shapes
for i, ref in enumerate(references):
   plt.plot(ref[:, 0], ref[:, 1], label=f'Reference {refnames[i]}', alpha=0.7)

# Plot target shape
plt.plot(target[:, 0], target[:, 1], 'k--', linewidth=2, label='Target (Rounded Square)')

# Highlight best match
plt.title(f'Target interpolated between Reference {refnames[best_i]} and {refnames[best_i+1]}\nFractional position: t = {interpolated_position:.6f}')
plt.legend()
plt.axis('equal')
plt.grid(True)
plt.show()

print(f"Result: Target is between Reference {best_i+1} {refnames[best_i]} and {best_i+2} {refnames[best_i+1]} at t = {interpolated_position:.6f}")
