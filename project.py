import numpy as np
import cv2
import os
import glob
from matplotlib import pyplot as plt

def load_and_preprocess_video(video_path, max_frames=None):
    cap = cv2.VideoCapture(video_path)
    frames = []
    original_frames = []
    
    if max_frames is None:
        max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frame_count = 0
    while cap.isOpened() and frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
            
        original_frames.append(frame)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray.flatten())
        frame_count += 1
    
    cap.release()
    
    if len(frames) == 0:
        raise ValueError("Не вдалося завантажити жодного кадру з відео")
    
    video_matrix = np.vstack(frames)
    return video_matrix.T, original_frames, np.vstack(frames)

video_path = '/Users/veronikabagatyr-zaharcenko/Downloads/IMG_0393.MOV'
video_matrix, original_frames, gray_matrices = load_and_preprocess_video(video_path)

ATA = np.dot(video_matrix.T, video_matrix)

def power_iteration(A, max_iter=1000, tol=1e-6):
    n = A.shape[0]
    v = np.random.rand(n)
    for _ in range(max_iter):
        v_new = A @ v
        v_new = v_new / np.linalg.norm(v_new)
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    eigenvalue = (v.T @ A @ v) / (v.T @ v)
    return eigenvalue, v

def find_all_eigenvalues(A, tol=1e-6, max_eigenvalues=None):
    A_k = A.copy()
    eigenvalues = []
    n = A.shape[0]
    
    while len(eigenvalues) < (max_eigenvalues or n):
        lambda_k, v_k = power_iteration(A_k, tol=tol)
        if np.abs(lambda_k) < tol:
            break

        eigenvalues.append(lambda_k)
        A_k = A_k - lambda_k * np.outer(v_k, v_k)

        if np.linalg.norm(A_k) < tol:
            break

    eigenvalues.sort(key=lambda x: -abs(x))
    return eigenvalues

all_eigenvalues = find_all_eigenvalues(ATA, tol=1e-3)
# print(f"Знайдено {len(all_eigenvalues)} власних значень:")
# print(np.array(all_eigenvalues))
# print(len(all_eigenvalues))

def find_eigenvectors(A, eigenvalues, tol=1e-6):
    eigenvectors = []
    n = A.shape[0]
    I = np.eye(n)

    for lambda_k in eigenvalues:
        M = A - lambda_k * I

        U, s, Vh = np.linalg.svd(M)

        v_k = Vh[-1]

        v_k = v_k / np.linalg.norm(v_k)
        eigenvectors.append(v_k)

    return eigenvectors

eigenvectors = find_eigenvectors(ATA, all_eigenvalues)


singular_values = np.sqrt(np.abs(all_eigenvalues))
m, n = video_matrix.shape
k = min(m, n)

Sigma = np.zeros((k, k))
Sigma[:k, :k] = np.diag(singular_values[:k])
# print(f"Розмір Σ: {Sigma.shape}")

V = np.column_stack(eigenvectors)
# print(f"Розмір V: {V.shape}")

U = np.zeros((video_matrix.shape[0], len(singular_values)))

# print(f"Розмір U: {U.shape}")

for i in range(len(singular_values)):
    if singular_values[i] > 1e-10:
        U[:, i] = (1 / singular_values[i]) * video_matrix @ eigenvectors[i]
    else:
        U[:, i] = 0

A_reconstructed = U @ Sigma @ V.T
error = np.linalg.norm(video_matrix - A_reconstructed)
# print(f"Похибка реконструкції: {error:.2e}")

k = 5
U_k = U[:, :k]
Sigma_k = Sigma[:k, :k]
V_k = V[:, :k]

A_back = U_k @ Sigma_k @ V_k.T
A_just_move = video_matrix - A_back
# print(A_just_move)

def training_data(humans, not_humans, patch_size=(64, 64)):
    X = []
    y = []
    
    for img_path in glob.glob(os.path.join(humans, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, patch_size)
        X.append(img.flatten())
        y.append(1) # Label 1 for humans
    for img_path in glob.glob(os.path.join(not_humans, "*.jpg")):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, patch_size)
        X.append(img.flatten())
        y.append(0)  # Label 0 for not humans
    
    return np.array(X).T, np.array(y)

X_train = training_data("../training/humans/", "../training/non_humans/")

mean = np.mean(X_train, axis=1, keepdims=True)
X_centered = X_train - mean

ATA = np.dot(X_centered.T, X_centered)
eigenvalues = find_all_eigenvalues(ATA)
eigenvectors = find_eigenvectors(ATA, eigenvalues)

singular_values = np.sqrt(np.abs(eigenvalues))
U = np.zeros((X_centered.shape[0], len(eigenvalues)))
for i in range(len(eigenvalues)):
    if singular_values[i] > 1e-10: #added cuz lights change, and that creates noise, that still fills the matrix
        U[:, i] = (1 / singular_values[i]) * X_centered @ eigenvectors[i]
    else:
        U[:, i] = 0

k = 50 #change for making human clearer
U_reduced = U[:, :k] #eigrn space

def classify_blob(blob, U_reduced, mean, threshold=0.3): #0.3 is just a test number, we will play with it
    blob_resized = cv2.resize(blob, (64, 64)).flatten().reshape(-1, 1)
    blob_centered = blob_resized - mean
    y = U_reduced.T @ blob_centered
    x_recon = U_reduced @ y + mean
    error = np.linalg.norm(blob_resized - x_recon) / np.linalg.norm(blob_resized)
    return error < threshold

#getting blobs of moving objects from the video
def extract_blobs(A_just_move, original_frames, min_area=100, patch_size=(64, 64)):
    blobs = []
    locations = []
    
    for i in range(A_just_move.shape[1]):
        foreground = A_just_move[:, i].reshape(original_frames[0].shape[:2])
        foreground = np.abs(foreground).astype(np.uint8)
        
        _, thresh = cv2.threshold(foreground, 30, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            if cv2.contourArea(cnt) > min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                
                blob = original_frames[i][y:y+h, x:x+w]
                
                blob_resized = cv2.resize(blob, patch_size)
                blobs.append(blob_resized)
                locations.append((x, y, w, h, i))
    
    return blobs, locations

blobs, locations = extract_blobs(A_just_move, original_frames)

#putting humans in boxes
human_tracking = []
for blob, (x, y, w, h, i) in zip(blobs, locations):
    if classify_blob(blob, U_reduced, mean):
        human_tracking.append((x, y, x+w, y+h, i))

for x1, y1, x2, y2, i in human_tracking:
    cv2.rectangle(original_frames[i], (x1, y1), (x2, y2), (0, 255, 0), 2)