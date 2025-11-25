# Лабораторная работа 1: Эрозия в компьютерном зрении

**Дисциплина:** Компьютерное зрение  
**Тема:** Эрозия  
**Выполнили:** студенты гр. P4154, P4155  
Ковалёв Артём, Мавлюкеев Артем, Касютин Илья  
**Санкт-Петербург, 2025**

## 1. Теоретическая база

### 1.1. Общая информация

Эрозия — операция в математической морфологии для анализа и обработки изображений в компьютерном зрении. Она основана на теории множеств и применяется в основном к бинарным изображениям, где объекты — белые пиксели, фон — чёрные. Эрозия уменьшает размеры объектов, удаляя внешние слои, сужая границы и устраняя мелкие детали или шум. Выполняется с помощью структурирующего элемента (ядра), задающего форму и масштаб изменений.

Эрозия используется для предварительной обработки изображений: удаления шумов (изолированные пиксели, тонкие линии), разделения соприкасающихся объектов в сегментации (например, клетки в медицинских изображениях или частицы в материалах). В сочетании с дилатацией формирует операции открытия и закрытия для баланса фильтрации и сохранения форм. Для градационных изображений адаптируется путём взятия минимума в окрестности, применимо к серым или цветным изображениям для выделения тёмных зон. Пример действия эрозии показан на рисунке 1.

Эрозия упрощает структуры изображений для дальнейшего анализа (распознавание контуров, классификация объектов) и применяется в робототехнике, биомедицине.

### 1.2. Структурный элемент

Структурный элемент (ядро, маска) — ключевой компонент морфологических операций, включая эрозию, определяющий локальную окрестность для анализа и модификации изображения. В бинарных изображениях — это подмножество пикселей в матрице фиксированного размера, с активными (1) и неактивными (0) позициями относительно центра. Центр служит опорой для наложения на каждый пиксель изображения, оценивая соответствие шаблону.

Форма и размер элемента влияют на результат: квадратная матрица 3x3 сужает границы равномерно, дисковая сохраняет плавные контуры. Асимметричные элементы возможны для анизотропных задач, но рекомендуется симметрия. Малые ядра (3x3) удаляют тонкие детали, большие (5x5+) существенно уменьшают объекты. Пример с разными размерами показан на рисунке 2.

Теоретически — множество в евклидовом пространстве или сетке для вычисления минимумов/пересечений. Для градационных изображений учитывает интенсивности. В библиотеках задаётся программно для гибкости в задачах от фильтрации шума до сегментации.

### 1.3. Алгоритм

Для бинарного изображения (объекты — 1, фон — 0) инициализируется результирующее изображение (копия исходного или пустой массив). Для каждого пикселя (x, y) накладывается структурный элемент с центром в (x, y), определяя окрестность.

Проверяется: все активные позиции элемента соответствуют объекту (1) в исходном изображении? Если да, результирующий пиксель = 1; иначе = 0. Повторяется для всех пикселей с padding по краям. При итерациях результат предыдущей становится входом. Алгоритм представлен на рисунке 3.

Для градационных изображений вычисляется минимум интенсивности в окрестности. Вычислительная сложность линейна от размера изображения и элемента; интегрируется в библиотеки с оптимизациями.

### 1.4. Базовая реализация эрозии

Пример реализации на C++ с OpenCV (сокращённая версия только для эрозии) взят с официального сайта: [https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html).

**Краткое объяснение кода:**  
Код загружает изображение, создаёт окно с трекбарами для выбора типа (прямоугольник, крест, эллипс) и размера ядра. Функция `Erosion` генерирует структурный элемент с `getStructuringElement`, применяет `erode` для эрозии и отображает результат. Это демонстрирует интерактивную обработку: изменения трекбаров обновляют изображение в реальном времени. Основная операция — `erode(src, erosion_dst, element)`, где `src` — исходное, `element` — ядро.

```cpp
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
using namespace cv;
using namespace std;

Mat src, erosion_dst;
int erosion_elem = 0;
int erosion_size = 0;
int const max_elem = 2;
int const max_kernel_size = 21;

void Erosion(int, void*);

int main(int argc, char** argv) {
    CommandLineParser parser(argc, argv, "{@input | LinuxLogo.jpg | input image}");
    src = imread(samples::findFile(parser.get<String>("@input")), IMREAD_COLOR);
    if (src.empty()) {
        cout << "Could not open or find the image!\n" << endl;
        cout << "Usage: " << argv[0] << " <Input image>" << endl;
        return -1;
    }
    namedWindow("Erosion Demo", WINDOW_AUTOSIZE);
    createTrackbar("Element:\n 0: Rect \n 1: Cross \n 2: Ellipse", "Erosion Demo", &erosion_elem, max_elem, Erosion);
    createTrackbar("Kernel size:\n 2n +1", "Erosion Demo", &erosion_size, max_kernel_size, Erosion);
    Erosion(0, 0);
    waitKey(0);
    return 0;
}

void Erosion(int, void*) {
    int erosion_type = 0;
    if (erosion_elem == 0) { erosion_type = MORPH_RECT; }
    else if (erosion_elem == 1) { erosion_type = MORPH_CROSS; }
    else if (erosion_elem == 2) { erosion_type = MORPH_ELLIPSE; }
    Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));
    erode(src, erosion_dst, element);
    imshow("Erosion Demo", erosion_dst);
}
```

# 2. Описание разработанной системы

### 2.1. Описание датасета

В качестве набора данных был выбран датасет [Flower Images Dataset](https://www.kaggle.com/datasets/alsaniipe/flowerdatasets).
Количество изображений: 1190
Формат изображений: jpeg
Средний размер изображений: 500x500px

### 2.2. Своя реализация эрозии
Был реализован простой класс с единственной главной функцией erode():

```python
class Erosion:
    def __init__(self, kernel=None):
        if kernel is None:
            kernel = np.ones((3, 3), dtype=np.uint8)
        self.kernel = kernel

        self.kernel_h, self.kernel_w = self.kernel.shape
        self.pad_h = self.kernel_h // 2
        self.pad_w = self.kernel_w // 2
    
    def erode(self, image: np.ndarray):
        img_h, img_w = image.shape
        
        padded_img = np.pad(image, ((self.pad_h, self.pad_h), (self.pad_w, self.pad_w)), mode='constant', constant_values=0)
        
        output = np.zeros_like(image)
        
        for i in range(img_h):
            for j in range(img_w):
                region = padded_img[i:i+self.kernel_h, j:j+self.kernel_w]
                
                match = np.where(self.kernel == 1, region, 1)
                output[i, j] = np.all(match == 1)
        
        return output
```
Для тестирования корректности работы алгоритма было взято несколько примеров, в том числе из изображения из [документации](https://docs.opencv.org/3.4/db/df6/tutorial_erosion_dilatation.html) OpenCV

Базовое изображение

<img src="images/base_image.png" width="200" height="200">
<img src="images/base_image_1.png" width="200" height="200">

Обработанное изображение

<img src="images/eroded_image.png" width="200" height="200">
<img src="images/eroded_image_1.png" width="200" height="200">

### 2.3. Использование готовой функции

В качестве уже реализованной функции была выбрана функция erode() из библиотеки OpenCV (Python)
Листинг скрипта использования приведён в приложении 1.

# 3. Тестирование быстродействия алгоритмов

Предварительно изображения были бинаризованы с помощью встроенных методов OpenCV:
```python
def otsu_binarize(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    return binary_image
```

Время обработки бинаризованных изображений самописным методом занимает ~24 минуты:
```

 15%|█▍        | 177/1190 [03:33<20:21,  1.21s/it]
CPU times: user 3min 31s, sys: 1.03 s, total: 3min 32s
Wall time: 3min 33s
```

В то же самое время обработка методами из OpenCV заняла менее 1 секунды:
```
CPU times: user 183 ms, sys: 597 ms, total: 780 ms
Wall time: 776 ms
```
Полный скрипта использования приведён в приложении 2.

# 4. Выводы
1. Наивная реализация эрозии, несмотря на корректность работы, является вычислительно затратной из-за использования вложенных циклов и отсутствия оптимизаций, что приводит к значительному времени обработки (24 минуты на 1190 изображений).
2. Реализация эрозии в библиотеке OpenCV демонстрирует высокую производительность (менее 1 секунды на тот же набор данных) благодаря оптимизированным алгоритмам и использованию аппаратных возможностей.
3. Для практических задач обработки изображений рекомендуется использовать проверенные и оптимизированные библиотеки, такие как OpenCV, чтобы избежать проблем с производительностью.
4. Выбор структурного элемента (ядра) важен для достижения желаемого результата эрозии, и его форма и размер должны подбираться в зависимости от конкретной задачи.

# 5. Использованные источники
1. https://docs.opencv.org/3.4/index.html - официальная документация OpenCV
2. Лекции по курсу "Компьютерное зрение" ПиРСИИ, 2025

# Приложение 1. Листинг скрипта замера скорости работы алгоритма эрозии из OpenCV

test_time_default.py

    ```
    import cv2 as cv
    import glob
    
    from tqdm import tqdm
    
    images = glob.glob("data/*/*.jpg")
    binary_images = []
    
    for image in tqdm(images):
        img = cv.imread(image, cv.IMREAD_GRAYSCALE)
        binary_images.append(
            cv.adaptiveThreshold(
                img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        )


    %%time
    
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
    final_images = []
    
    for image in binary_images:
        final_images.append(cv.erode(image, kernel))
    ```

# Приложение 2. Листинг скрипта замера скорости работы самописного алгоритма эрозии


test_time_custom.py

    ```
    from utils import Erosion, image_to_bin_array
    import os
    import cv2


    erosion = Erosion()

    images = []
    input_folder = "flowers_binarized"
    for dir in os.listdir(input_folder):
        for filename in os.listdir(os.path.join(input_folder, dir)):
            if filename.lower().endswith('.jpg'):
                img_path = os.path.join(input_folder, dir, filename)
                img = cv2.imread(img_path)
                if len(img.shape) == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                images.append(image_to_bin_array(os.path.join(input_folder, dir, filename)))


    %%time

    for image in tqdm.tqdm(images, total=len(images)):
        erosion.erode(image)
    ```
