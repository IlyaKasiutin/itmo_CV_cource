#!/usr/bin/env python3
"""
Основной класс RobustObjectTracker
"""

import cv2
import numpy as np
from collections import deque
from config import (
    DETECTOR_SETTINGS, TRACKER_DEFAULTS, TEMPLATE_SETTINGS,
    LK_PARAMS, PREPROCESSING, FILTER_SETTINGS, RANSAC_SETTINGS
)


class RobustObjectTracker:
    """
    Комбинированный трекер:
      - SIFT/ORB детектор + FLANN/BF matcher
      - мульти-скейл / поворотные вариации шаблона (пирамида)
      - адаптивный RANSAC
      - переход в режим трекинга через calcOpticalFlowPyrLK (LK) после успешной детекции
      - простой Kalman для сглаживания центра
    """
    
    def __init__(self, **kwargs):
        # Устанавливаем параметры с учетом значений по умолчанию
        self.detector_type = kwargs.get('detector_type', TRACKER_DEFAULTS['detector_type']).lower()
        self.lowe_ratio = kwargs.get('lowe_ratio', TRACKER_DEFAULTS['lowe_ratio'])
        self.min_inliers = kwargs.get('min_inliers', TRACKER_DEFAULTS['min_inliers'])
        self.max_template_variants = kwargs.get('max_template_variants', 
                                              TRACKER_DEFAULTS['max_template_variants'])
        self.debug = kwargs.get('debug', TRACKER_DEFAULTS['debug'])

        self._init_detector()
        self._init_template_variants()
        self._init_tracking()
        self._init_filters()

    def _init_detector(self):
        """Инициализация детектора и матчера"""
        if self.detector_type == 'sift':
            settings = DETECTOR_SETTINGS['sift']
            self.detector = cv2.SIFT_create(nfeatures=settings['nfeatures'])
            self.matcher = cv2.FlannBasedMatcher(
                settings['index_params'], 
                settings['search_params']
            )
            self.is_binary = settings['is_binary']
        elif self.detector_type == 'akaze':
            self.detector = cv2.AKAZE_create()
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.is_binary = DETECTOR_SETTINGS['akaze']['is_binary']
        elif self.detector_type == 'orb':
            settings = DETECTOR_SETTINGS['orb']
            self.detector = cv2.ORB_create(nfeatures=settings['nfeatures'])
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            self.is_binary = settings['is_binary']
        else:
            raise ValueError(f"Unsupported detector_type: '{self.detector_type}'. "
                           f"Choose from {list(DETECTOR_SETTINGS.keys())}")

    def _init_template_variants(self):
        """Инициализация списка вариаций шаблона"""
        self.template_variants = []  # список словарей: img_gray, kp, desc, scale, angle, shape
        self.template_frame = None
        self.template_shape = None

    def _init_tracking(self):
        """Инициализация переменных трекинга"""
        self.tracking_mode = False
        self.template_pts = None     # точки шаблона (Nx2 float32)
        self.tracked_frame_pts = None  # соответствующие точки в кадре
        self.kp_frame = None
        self.desc_frame = None
        self.prev_gray = None

    def _init_filters(self):
        """Инициализация фильтров и истории"""
        self.bbox_history = deque(maxlen=FILTER_SETTINGS['bbox_history_len'])
        self._init_kalman_filter()

    def _init_kalman_filter(self):
        """Инициализация фильтра Калмана"""
        state_dim = FILTER_SETTINGS['kalman_state_dim']
        measure_dim = FILTER_SETTINGS['kalman_measure_dim']
        
        self.kalman = cv2.KalmanFilter(state_dim, measure_dim)
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        self.kalman.processNoiseCov = np.eye(state_dim, dtype=np.float32) * \
                                    FILTER_SETTINGS['process_noise_cov']
        self.kalman.measurementNoiseCov = np.eye(measure_dim, dtype=np.float32) * \
                                        FILTER_SETTINGS['measurement_noise_cov']
        self.kalman.statePost = np.zeros((state_dim, 1), dtype=np.float32)

    # ---------- Utilities ----------
    def _preprocess_gray(self, img):
        """CLAHE + небольшое усиление текстуры через Laplacian (легкий)"""
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        clahe = cv2.createCLAHE(
            clipLimit=PREPROCESSING['clahe_clip_limit'],
            tileGridSize=PREPROCESSING['clahe_tile_grid_size']
        )
        gray = clahe.apply(gray)
        
        lap = cv2.Laplacian(gray, cv2.CV_8U)
        gray = cv2.addWeighted(
            gray, PREPROCESSING['base_weight'],
            lap, PREPROCESSING['laplacian_weight'],
            0
        )
        return gray

    def _compute_kp_desc(self, img_gray):
        """Вычисление ключевых точек и дескрипторов"""
        kp, desc = self.detector.detectAndCompute(img_gray, None)
        return kp, desc

    # ---------- Template building ----------
    def set_template(self, template_frame, scales=None, angles=None):
        """
        Устанавливает шаблон и строит вариации.
        scales: list of scale multipliers (например [0.5,0.75,1.0,1.5])
        angles: list of rotation angles in degrees (e.g. [0, -15, 15])
        """
        if scales is None:
            scales = TEMPLATE_SETTINGS['scales']
        if angles is None:
            angles = TEMPLATE_SETTINGS['angles']

        self.template_frame = template_frame.copy()
        base_gray = self._preprocess_gray(template_frame)
        h0, w0 = base_gray.shape[:2]
        self.template_shape = (h0, w0)
        
        # Генерация всех комбинаций масштабов и углов
        combos = [(s, a) for s in scales for a in angles]
        # Сортируем чтобы сначала шли наиболее вероятные (scale ~1, angle ~0)
        combos.sort(key=lambda x: (abs(x[0] - 1.0) + abs(x[1] / 45.0)))
        combos = combos[:self.max_template_variants]  # ограничение по скорости
        
        variants = []
        for scale, angle in combos:
            new_w = max(20, int(w0 * scale))
            new_h = max(20, int(h0 * scale))
            
            # Масштабирование
            scaled = cv2.resize(
                base_gray, (new_w, new_h),
                interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
            )
            
            # Поворот
            if angle != 0:
                M = cv2.getRotationMatrix2D((new_w / 2, new_h / 2), angle, 1.0)
                rotated = cv2.warpAffine(
                    scaled, M, (new_w, new_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT
                )
            else:
                rotated = scaled
            
            # Вычисление ключевых точек
            kp, desc = self._compute_kp_desc(rotated)
            if desc is None or kp is None or len(kp) < 4:
                continue
            
            variants.append({
                'img': rotated,
                'kp': kp,
                'desc': desc,
                'scale': scale,
                'angle': angle,
                'shape': rotated.shape[:2]
            })
        
        # fallback: если ничего не нашли — попробовать оригинал
        if len(variants) == 0:
            kp, desc = self._compute_kp_desc(base_gray)
            if desc is None or kp is None or len(kp) < 4:
                if self.debug:
                    print("[TEMPLATE] Ошибка: недостаточно ключевых точек в шаблоне")
                return False
            variants.append({
                'img': base_gray,
                'kp': kp,
                'desc': desc,
                'scale': 1.0,
                'angle': 0,
                'shape': base_gray.shape[:2]
            })
        
        self.template_variants = variants
        if self.debug:
            print(f"[TEMPLATE] Вариантов: {len(self.template_variants)}")
            for v in self.template_variants:
                print(f"  scale={v['scale']}, angle={v['angle']}, kps={len(v['kp'])}")
        return True

    # ---------- Detection (matching) ----------
    def _match_template_variant(self, variant, frame_gray):
        """Возвращает структуру с лучшей гомографией для данного варианта или None"""
        tpl_kp = variant['kp']
        tpl_desc = variant['desc']
        if tpl_desc is None or len(tpl_kp) < 4:
            return None
        
        # Detect keypoints in frame
        kp_frame, desc_frame = self._compute_kp_desc(frame_gray)
        if desc_frame is None or kp_frame is None or len(kp_frame) < 4:
            return None

        # knnMatch + Lowe ratio test
        try:
            matches = self.matcher.knnMatch(tpl_desc, desc_frame, k=2)
        except Exception:
            # For BFMatcher with binary descriptors matches may throw - fallback
            try:
                self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                matches = self.matcher.knnMatch(tpl_desc, desc_frame, k=2)
            except Exception:
                return None

        # Lowe ratio filtering
        good = []
        for m_n in matches:
            if len(m_n) == 2:
                m, n = m_n
                if m.distance < self.lowe_ratio * n.distance:
                    good.append(m)
        
        min_good = max(4, int(self.min_inliers * 0.4))
        if len(good) < min_good:
            return None

        # Фильтрация по согласованности размеров
        good = self._filter_scale_consistency(good, tpl_kp, kp_frame)
        if len(good) < 4:
            return None

        # Подготовка точек для гомографии
        src_pts = np.float32([tpl_kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Адаптивный RANSAC порог
        h_tpl, w_tpl = variant['shape']
        base_thresh = max(3.0, min(max(w_tpl, h_tpl) * 
                                 RANSAC_SETTINGS['reprojection_threshold_factor'], 40.0))

        try:
            M, mask = cv2.findHomography(
                src_pts, dst_pts, cv2.RANSAC,
                ransacReprojThreshold=base_thresh,
                maxIters=RANSAC_SETTINGS['max_iters'],
                confidence=RANSAC_SETTINGS['confidence']
            )
        except Exception:
            return None
        
        if M is None or mask is None:
            return None
        
        inliers = int(np.sum(mask))
        min_inliers = max(6, int(self.min_inliers * RANSAC_SETTINGS['min_inliers_factor']))
        if inliers < min_inliers:
            return None

        # Трансформация углов шаблона
        h, w = variant['shape']
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(corners, M)
        except Exception:
            return None
        
        area = abs(cv2.contourArea(transformed.reshape(-1, 2)))
        if area < 1.0:
            return None

        return {
            'M': M,
            'mask': mask.ravel(),
            'inliers': inliers,
            'src_pts': src_pts,
            'dst_pts': dst_pts,
            'variant': variant,
            'corners': transformed
        }

    def _filter_scale_consistency(self, matches, tpl_kp, frame_kp):
        """Оставляет пары с относительно согласованными размерами ключевых точек"""
        if len(matches) < 4:
            return matches
        
        ratios = []
        for m in matches:
            tk = tpl_kp[m.queryIdx]
            fk = frame_kp[m.trainIdx]
            tsize = getattr(tk, 'size', 1.0) or 1.0
            fsize = getattr(fk, 'size', 1.0) or 1.0
            ratios.append(fsize / tsize)
        
        med = np.median(ratios) if len(ratios) > 0 else 1.0
        tol_min, tol_max = FILTER_SETTINGS['scale_consistency_tol']
        
        good = []
        for idx, m in enumerate(matches):
            r = ratios[idx]
            if tol_min * med <= r <= tol_max * med:
                good.append(m)
        
        if len(good) < max(4, int(len(matches) * 0.4)):
            return matches
        return good

    # ---------- Optical flow tracking ----------
    def _init_tracking_from_detection(self, detection_result, frame_gray):
        """
        Инициализация режима tracking (LK) после детекции:
          - сохраняем соответствия шаблонных точек и их координаты в кадре
        """
        src_pts = detection_result['src_pts'].reshape(-1, 2)   # template pts
        dst_pts = detection_result['dst_pts'].reshape(-1, 2)   # frame pts (matched)
        mask = detection_result['mask'].reshape(-1).astype(bool)
        
        # Берем только инлаеры
        if mask.sum() < 4:
            return False
        
        src_inliers = src_pts[mask]
        dst_inliers = dst_pts[mask]
        
        # Подготовка для LK: prevPts as Nx1x2 float32
        prev_pts = dst_inliers.reshape(-1, 1, 2).astype(np.float32)
        self.template_pts = src_inliers.astype(np.float32)  # template coordinates
        self.tracked_frame_pts = prev_pts.copy()
        self.prev_gray = frame_gray.copy()
        self.tracking_mode = True
        
        # Инициализация фильтра Калмана
        centroid = np.mean(dst_inliers, axis=0).astype(np.float32)
        self.kalman.statePost = np.array(
            [[centroid[0]], [centroid[1]], [0.], [0.]], 
            dtype=np.float32
        )
        
        if self.debug:
            print(f"[TRACK INIT] inliers={len(self.template_pts)} centroid={centroid}")
        return True

    def _track_with_lk(self, frame_gray):
        """
        Попытка трекинга через Lucas-Kanade для текущих tracked_frame_pts.
        Возвращает bbox (4 corners) если успешен, иначе None.
        """
        if self.tracked_frame_pts is None or len(self.tracked_frame_pts) < 4:
            return None
        
        # Расчет оптического потока
        next_pts, st, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, frame_gray, 
            self.tracked_frame_pts, None, **LK_PARAMS
        )
        
        if next_pts is None or st is None:
            return None
        
        st = st.reshape(-1)
        good_mask = st == 1
        good_new = next_pts.reshape(-1, 2)[good_mask]
        good_old = self.tracked_frame_pts.reshape(-1, 2)[good_mask]
        good_template = self.template_pts.reshape(-1, 2)[good_mask]
        
        min_points = max(4, int(len(self.template_pts) * 0.4))
        if len(good_new) < min_points:
            if self.debug:
                print(f"[LK] мало хороших точек: {len(good_new)}")
            return None
        
        # Оценка гомографии между template_pts и good_new
        try:
            h_tpl, w_tpl = self.template_shape
            base_thresh = max(3.0, min(max(w_tpl, h_tpl) * 
                                     RANSAC_SETTINGS['reprojection_threshold_factor'], 40.0))
            
            M, mask = cv2.findHomography(
                good_template.reshape(-1, 1, 2), 
                good_new.reshape(-1, 1, 2),
                cv2.RANSAC, 
                ransacReprojThreshold=base_thresh,
                maxIters=RANSAC_SETTINGS['max_iters'],
                confidence=RANSAC_SETTINGS['confidence']
            )
        except Exception:
            return None
        
        if M is None:
            return None
        
        mask = mask.ravel()
        inliers = int(np.sum(mask))
        min_inliers = max(6, int(self.min_inliers * RANSAC_SETTINGS['min_inliers_factor']))
        if inliers < min_inliers:
            if self.debug:
                print(f"[LK] гомография мало инлаеров: {inliers}")
            return None
        
        # Трансформация углов
        h, w = self.template_shape
        corners = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        try:
            transformed = cv2.perspectiveTransform(corners, M)
        except Exception:
            return None
        
        # Обновление точек для следующей итерации
        good_new_inliers = good_new[mask == 1]
        good_template_inliers = good_template[mask == 1]
        if len(good_new_inliers) < 4:
            return None
        
        self.tracked_frame_pts = good_new_inliers.reshape(-1, 1, 2).astype(np.float32)
        self.template_pts = good_template_inliers.reshape(-1, 2).astype(np.float32)
        self.prev_gray = frame_gray.copy()
        
        # Коррекция фильтра Калмана
        centroid = np.mean(good_new_inliers, axis=0).astype(np.float32)
        measurement = np.array([[np.float32(centroid[0])], [np.float32(centroid[1])]])
        pred = self.kalman.predict()
        self.kalman.correct(measurement)
        
        return transformed

    # ---------- Public track function ----------
    def track(self, frame):
        """
        Трекер на один кадр.
        Возвращает: frame_with_bbox, bbox_corners (4x1x2 ints), matches_img (для debug or None)
        """
        frame_gray = self._preprocess_gray(frame)

        # 1) Если в режиме tracking (LK) — пытаться трекать
        if self.tracking_mode:
            transformed = self._track_with_lk(frame_gray)
            if transformed is not None:
                bbox = transformed.astype(int)
                self.bbox_history.append(bbox.copy())
                smoothed = self._smooth_bbox()
                out_frame = self._draw_result(frame, smoothed, mode="LK")
                return out_frame, smoothed, None
            else:
                # Потеряли трекинг — переход на детекцию
                if self.debug:
                    print("[MODE] LK failed -> back to detection")
                self.tracking_mode = False
                self.tracked_frame_pts = None
                self.template_pts = None

        # 2) Detection: перебор вариаций шаблона, выбрать лучшую
        best = None
        for variant in self.template_variants:
            res = self._match_template_variant(variant, frame_gray)
            if res is None:
                continue
            if best is None or res['inliers'] > best['inliers']:
                best = res
        
        if best is None:
            # ничего не нашли
            if self.debug:
                print("[DETECT] Объект не найден на кадре")
            return frame, None, None

        # Инициализация tracking с найденными inliers
        ok = self._init_tracking_from_detection(best, frame_gray)
        if not ok:
            return frame, None, None

        # Рисование результата
        bbox = best['corners'].astype(int)
        self.bbox_history.append(bbox.copy())
        smoothed = self._smooth_bbox()
        out_frame = self._draw_result(frame, smoothed, mode="Detect", 
                                     inliers=best['inliers'])

        # Для отладки: подготовка matches_img
        matches_img = None
        if self.debug and best['variant'] is not None:
            matches_img = self._create_matches_image(best, frame_gray)

        return out_frame, smoothed, matches_img

    def _draw_result(self, frame, bbox, mode="Detect", inliers=0):
        """Рисование bounding box и текста на кадре"""
        out_frame = frame.copy()
        
        if bbox is not None:
            # Рисование полигона
            cv2.polylines(out_frame, [bbox.reshape(-1, 1, 2)], True, 
                         (0, 255, 0), 3, lineType=cv2.LINE_AA)
            
            # Добавление текста
            centroid = np.mean(bbox.reshape(-1, 2), axis=0).astype(int)
            if mode == "Detect":
                text = f"In-Detect (inliers={inliers})"
                x_offset = -80
            else:  # LK mode
                text = "In-LK"
                x_offset = -40
            
            cv2.putText(out_frame, text, 
                       (centroid[0] + x_offset, centroid[1] - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return out_frame

    def _create_matches_image(self, detection_result, frame_gray):
        """Создание изображения с матчами для отладки"""
        try:
            variant = detection_result['variant']
            tpl_img = variant['img']
            # Конвертация в цвет для отрисовки
            tpl_color = cv2.cvtColor(tpl_img, cv2.COLOR_GRAY2BGR)
            frame_color = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
            
            # Рисование матчей
            matches_img = cv2.drawMatches(
                tpl_color, variant['kp'],
                frame_color, self.kp_frame,
                [], None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            return matches_img
        except Exception:
            return None

    def _smooth_bbox(self):
        """Сглаживание: усреднение истории + Kalman центр"""
        if not self.bbox_history:
            return np.zeros((4, 1, 2), dtype=int)
        
        arr = np.array(self.bbox_history)
        avg = np.mean(arr.astype(np.float32), axis=0).astype(int)
        
        # Поправка фильтром Калмана
        pred = self.kalman.predict()
        pred_cent = np.array([pred[0, 0], pred[1, 0]], dtype=np.int32)
        
        # Смещение усреднённой рамки к pred_cent
        cur_cent = np.mean(avg.reshape(-1, 2), axis=0).astype(np.int32)
        shift = pred_cent - cur_cent
        
        avg_shifted = avg.copy()
        avg_shifted = (avg_shifted.reshape(-1, 2) + shift).reshape(-1, 1, 2)
        
        return avg_shifted.astype(int)