#!/usr/bin/env python3
"""
Утилиты для работы с видео и пользовательским интерфейсом
"""

import cv2
import time
from pathlib import Path
from config import VIDEO_SETTINGS, TEMPLATE_SETTINGS


def select_template_region(video_path):
    """
    Выбор региона для шаблона из первого кадра видео
    """
    cap = cv2.VideoCapture(str(video_path))
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Не удалось прочитать первый кадр видео")
        return None
    
    print("Выберите объект (текстурированную область)")
    roi = cv2.selectROI(
        "Выберите объект (Drag rectangle, press SPACE/ENTER)", 
        frame, False
    )
    cv2.destroyAllWindows()
    
    if roi[2] == 0 or roi[3] == 0:
        print("Регион не выбран, используется весь кадр")
        return frame
    
    x, y, w, h = map(int, roi)
    # Обеспечение минимального размера шаблона
    w = max(w, TEMPLATE_SETTINGS['min_template_size'])
    h = max(h, TEMPLATE_SETTINGS['min_template_size'])
    
    tpl = frame[y:y+h, x:x+w]
    
    # Показ выбранного шаблона
    cv2.imshow("Template", tpl)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    
    return tpl


def create_video_writer(output_path, cap):
    """
    Создание VideoWriter для сохранения результата
    """
    fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_SETTINGS['default_fps']
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_SETTINGS['fourcc'])
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    return out


def add_status_info(frame, frame_idx, found_count, bbox_found):
    """
    Добавление служебной информации на кадр
    """
    if bbox_found:
        status = f"FOUND ({found_count})"
        color = (0, 255, 0)
    else:
        status = "NOT FOUND"
        color = (0, 0, 255)
    
    success_rate = found_count / max(1, frame_idx) * 100
    
    # Добавление текста на кадр
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, status, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Success: {success_rate:.1f}%", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame


def process_video(input_video, output_video=None, detector='sift', 
                  show_matches=False, debug=False, max_variants=12):
    """
    Основная функция обработки видео
    """
    cap = cv2.VideoCapture(str(input_video))
    if not cap.isOpened():
        print("Не удалось открыть видео")
        return
    
    # Проверка первого кадра
    ret, _ = cap.read()
    if not ret:
        print("Не удалось прочитать первый кадр")
        cap.release()
        return
    
    # Выбор шаблона
    template = select_template_region(input_video)
    if template is None:
        cap.release()
        return
    
    # Инициализация трекера
    from tracker import RobustObjectTracker
    tracker = RobustObjectTracker(
        detector_type=detector,
        lowe_ratio=0.75,
        min_inliers=14,
        max_template_variants=max_variants,
        debug=debug
    )
    
    if not tracker.set_template(template):
        print("Не удалось инициализировать шаблон (мало keypoints)")
        cap.release()
        return
    
    # Подготовка VideoWriter
    if output_video:
        out = create_video_writer(output_video, cap)
    else:
        out = None
    
    # Основной цикл обработки
    frame_idx = 0
    found_count = 0
    t0 = time.time()
    
    print("Start tracking. Controls: Q - quit, P - pause, D - toggle debug")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        
        # Трекинг кадра
        out_frame, bbox, matches_img = tracker.track(frame)
        
        # Статистика
        if bbox is not None:
            found_count += 1
        
        # Добавление информации о статусе
        out_frame = add_status_info(out_frame, frame_idx, found_count, bbox is not None)
        
        # Отображение результатов
        cv2.imshow("Robust Object Tracker", out_frame)
        
        if show_matches and matches_img is not None:
            cv2.imshow("Matches", matches_img)
        
        # Сохранение в файл
        if out is not None:
            out.write(out_frame)
        
        # Обработка клавиш
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.waitKey(0)
        elif key == ord('d'):
            tracker.debug = not tracker.debug
            print("Debug:", tracker.debug)
    
    # Завершение работы
    dt = time.time() - t0
    success_rate = found_count / max(1, frame_idx) * 100
    
    print(f"Done. Frames: {frame_idx}, Found: {found_count}, "
          f"Success: {success_rate:.1f}%, Time: {dt:.1f}s")
    
    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()