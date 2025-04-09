import cv2
import numpy as np


def power_iteration(A, max_iter=1000, tol=1e-8):
    n = A.shape[1]
    v = np.random.rand(n)
    v = v / np.linalg.norm(v)

    for _ in range(max_iter):
        Av = A @ v
        v_new = Av / np.linalg.norm(Av)
        if np.linalg.norm(v_new - v) < tol:
            break
        v = v_new
    sigma_squared = np.linalg.norm(A @ v) ** 2
    return np.sqrt(sigma_squared), v


def manual_svd_power(A, k=1):
    m, n = A.shape
    U = np.zeros((m, k))
    S = np.zeros(k)
    Vt = np.zeros((k, n))

    A_residual = A.copy()

    for i in range(k):
        sigma, v = power_iteration(A_residual.T @ A_residual)
        u = A_residual @ v / sigma

        U[:, i] = u
        S[i] = sigma
        Vt[i, :] = v
        A_residual -= sigma * np.outer(u, v)
    return U, S, Vt


cap = cv2.VideoCapture("sample_video.mp4")
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frames.append(gray)

height, width = frames[0].shape
A = np.array([frame.flatten() for frame in frames]).T.astype(np.float32)

mean_frame = np.mean(A, axis=1, keepdims=True)
A_centered = A - mean_frame

k = 1
U, S, Vt = manual_svd_power(A_centered, k=k)

A_background = (U[:, :k] * S[:k]) @ Vt[:k, :]
A_foreground = A_centered - A_background

foreground_vid = (
    (A_foreground - A_foreground.min())
    / (A_foreground.max() - A_foreground.min())
    * 255
)
foreground_vid = foreground_vid.astype(np.uint8)

for i in range(foreground_vid.shape[1]):
    frame = foreground_vid[:, i].reshape(height, width)
    cv2.imshow("Foreground", frame)
    if cv2.waitKey(30) & 0xFF == 27:  # press ESC to close the video
        break

cap.release()
cv2.destroyAllWindows()
