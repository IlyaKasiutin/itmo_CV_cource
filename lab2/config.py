#!/usr/bin/env python3
"""
Конфигурационные параметры трекера
"""
import cv2
# Параметры детектора
DETECTOR_SETTINGS = {
    'sift': {
        'nfeatures': 50000,
        'index_params': dict(algorithm=1, trees=5),
        'search_params': dict(checks=50),
        'is_binary': False
    },
    'akaze': {
        'is_binary': True
    },
    'orb': {
        'nfeatures': 3000,
        'is_binary': True
    }
}

# Параметры трекера по умолчанию
TRACKER_DEFAULTS = {
    'detector_type': 'sift',
    'lowe_ratio': 0.75,
    'min_inliers': 14,
    'max_template_variants': 12,
    'debug': False
}

# Параметры шаблона
TEMPLATE_SETTINGS = {
    'scales': [0.6, 0.8, 1.0, 1.3, 1.7],
    'angles': [0, -15, 15],
    'min_template_size': 80  # минимальный размер шаблона в пикселях
}

# Параметры LK (оптического потока)
LK_PARAMS = {
    'winSize': (21, 21),
    'maxLevel': 3,
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
}

# Параметры предобработки
PREPROCESSING = {
    'clahe_clip_limit': 2.0,
    'clahe_tile_grid_size': (8, 8),
    'laplacian_weight': 0.15,
    'base_weight': 0.85
}

# Параметры фильтров
FILTER_SETTINGS = {
    'bbox_history_len': 8,
    'kalman_state_dim': 4,
    'kalman_measure_dim': 2,
    'process_noise_cov': 1e-2,
    'measurement_noise_cov': 1e-1,
    'scale_consistency_tol': (0.4, 2.5)  # допуск для фильтрации ключевых точек
}

# Параметры RANSAC
RANSAC_SETTINGS = {
    'confidence': 0.995,
    'max_iters': 2000,
    'min_inliers_factor': 0.4,
    'reprojection_threshold_factor': 0.02  # от размера объекта
}

# Параметры видео
VIDEO_SETTINGS = {
    'default_fps': 25.0,
    'fourcc': 'mp4v',
    'output_resolution': None  # None = сохранять исходное разрешение
}