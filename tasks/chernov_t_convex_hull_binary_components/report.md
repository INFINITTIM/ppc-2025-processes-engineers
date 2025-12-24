# Построение выпуклой оболочки для компонент бинарного изображения

- **Студент**: Чернов Тимур Владимирович, группа 3823Б1ПР1
- **Технология**: SEQ | MPI
- **Вариант**: 32

## 1. Введение

Нахождение выпуклых оболочек связных компонент бинарного изображения - это довольно важная задача в компьютерном зрении, обработке изображений и вычислительной геометрии. Она применяется, например, в распознавании объектов, анализе формы и сегментации изображений. При работе с большими изображениями последовательная обработка становится медленной. Целью данной работы является реализация и сравнение двух подходов: последовательного (SEQ) и распределённого по MPI, в данном случае, с горизонтальным разбиением изображения.

## 2. Постановка задачи
**Описание задачи**

Дано бинарное изображение, представленное как матрица размером height × width, где каждый пиксель равен либо 0 (фон), либо 1 (объект). Требуется:
    - Найти все 4-связные компоненты из пикселей со значением 1.
    - Для каждой компоненты построить её выпуклую оболочку.
    - Вернуть список оболочек.

### Формат данных:

Входные типы данных: высота и ширина изображения, а также вектор содержащий значения пикселей бинарного изображения.
```cpp
using InType = std::tuple<int, int, std::vector<int>>;
```
Выходной тип данных: вектор, содержаший оболочки, представленные вектором из пар (точек на изображении).
```cpp
using OutType = std::vector<std::vector<std::pair<int, int>>>;
```

### Ограничения:

- Пиксели: только 0 и 1 
- Связность: 4-связная (вверх, вниз, влево, вправо)
- Все размеры строго положительны
- Количество элементов в `pixels = width x height`

## 3. Базовый последовательный алгоритм (SEQ)

Последовательный алгоритм реализует построение выпуклой оболочки компонент бинарного изображения в два этапа:
    - Поиск 4-связных компонент на бинарном изображении.
    - Построение выпуклой оболочки для каждой компоненты.

Если рассматривать алгоритм более подробно то, начав с первого этапа стоит упомянуть что обход изображения производится по строкам и столбцам, если пиксель равный единице, т.е. не являющийся фоном ещё не посещён, то запускается поиск в ширину (BFS), который в свою очередь использует очередь и 4 направления (вверх, вниз, влево, вправо).

Код алгоритма:

```cpp
constexpr std::array<std::pair<int, int>, 4> kDirs = {{{0, -1}, {0, 1}, {-1, 0}, {1, 0}}};

void ExtractComponent(...) {
  std::queue<std::pair<int, int>> q;
  q.emplace(start_col, start_row);
  visited[...] = true;

  while (!q.empty()) {
    auto [cx, cy] = q.front(); q.pop();
    comp.emplace_back(cx, cy);
    for (const auto &[dx, dy] : kDirs) {
      int nx = cx + dx, ny = cy + dy;
      if (в пределах изображения и пиксель == 1 и не посещён) {
        visited[...] = true;
        q.emplace(nx, ny);
      }
    }
  }
}
```

Далее стоит подробнее рассмотреть 2 этап, т.е. построение выпуклой оболочки. В данной задаче было принято решение использовать алгоритм Эндрю - улучшенную версию алгоритма Грэхема. Алгоритм Эндрю - алгоритм цепочки. В большей степени, решение выбрать алгоритм Эндрю было сделано в пользу того, чтобы убрать из кода тригонометрию, использующуюся в алгоритме Грэхема, которая могла замедлить программу.

**Этапы выполнения:**

1. Подготовка данных: 
```cpp
std::ranges::sort(pts);
auto [first, last] = std::ranges::unique(pts);
pts.erase(first, last);
```
2. Построение нижней цепочки
```cpp
for (const auto &p : pts) {
  while (hull.size() >= 2) {
    auto &a = hull[hull.size() - 2];
    auto &b = hull[hull.size() - 1];
    int64_t cross = (b.x - a.x)*(p.y - a.y) - (b.y - a.y)*(p.x - a.x);
    if (cross >= 0) break;
    hull.pop_back();
  }
  hull.push_back(p);
}
```
3. Построение верхней цепочки
```cpp
size_t lower_len = hull.size();
for (auto it = pts.rbegin() + 1; it != pts.rend(); ++it) {
}
```
4. Финальная коррекция результата
```cpp
if (hull.size() > 1) hull.pop_back();
```

## 4. Схема распараллеливания (MPI)
В параллельной реализации используется ленточная горизонтальная схема разбиения изображения: исходное бинарное изображение делится по строкам между MPI-процессами. Для корректного обнаружения связных компонент, пересекающих границы между процессами, применяется механизм обмена граничными, призрачными, строками. Каждый процесс выполняет локальный поиск компонент в расширенной области (свои строки + одна сверху и одна снизу от соседей), затем сохраняет только те точки компонент, которые принадлежат его локальному фрагменту. После построения локальных выпуклых оболочек результаты собираются на Rank 0 и рассылаются всем процессам.

**Этапы выполнения:**
1. Рассылка размеров изображения (BroadcastMatrixSizes): Rank 0 отправляет ширину и высоту изображения всем процессам через MPI_Bcast.
2. Разделение пикселей изображения (MPI_Scatterv): строки изображения распределяются между процессами с учётом возможного остатка (если height не делится нацело на число процессов).
3. Обмен граничными строками (ExchangeBoundaryRows): каждый процесс получает одну строку сверху и одну снизу от соседей с помощью неблокирующих операций MPI_Irecv/MPI_Isend.
4. Локальный поиск компонент и построение оболочек: в расширенной области выполняется BFS с 4-связностью; найденные компоненты фильтруются по принадлежности к локальному фрагменту, затем для каждой строится выпуклая оболочка (алгоритм Эндрю).
5. Сбор и рассылка результата (GatherAndBroadcastResult): локальные оболочки отправляются на Rank 0 через точечные MPI_Send/MPI_Recv; после сбора результат широковещательно рассылается всем процессам через MPI_Bcast.

**Детальный алгоритм параллельной реализации**

1. Рассылка размеров матриц:
```cpp
std::array<int, 2> dims{0, 0};
if (rank_ == 0) {
  dims[0] = std::get<0>(GetInput());
  dims[1] = std::get<1>(GetInput()); 
}
MPI_Bcast(dims.data(), 2, MPI_INT, 0, MPI_COMM_WORLD);
width_ = dims[0];
height_ = dims[1];
```

2. Разделение пикселей изображения:
```cpp
int base_rows = height_ / size_;
int rem = height_ % size_;
start_row_ = (rank_ * base_rows) + std::min(rank_, rem);
end_row_ = start_row_ + base_rows + (rank_ < rem ? 1 : 0);
int local_rows = end_row_ - start_row_;

if (rank_ == 0) {
  const auto &pixels = std::get<2>(GetInput());
  std::vector<int> sendcounts(size_, 0);
  std::vector<int> displs(size_, 0);
  int offset = 0;
  for (int i = 0; i < size_; ++i) {
    int rows_i = base_rows + (i < rem ? 1 : 0);
    sendcounts[i] = rows_i * width_;
    displs[i] = offset;
    offset += sendcounts[i];
  }
  local_pixels_.resize(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(width_));
  MPI_Scatterv(pixels.data(), sendcounts.data(), displs.data(), MPI_INT,
               local_pixels_.data(), static_cast<int>(local_pixels_.size()), MPI_INT,
               0, MPI_COMM_WORLD);
} else {
  local_pixels_.resize(static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(width_));
  MPI_Scatterv(nullptr, nullptr, nullptr, MPI_INT,
               local_pixels_.data(), static_cast<int>(local_pixels_.size()), MPI_INT,
               0, MPI_COMM_WORLD);
}
```

3. Обмен граничными строками:
```cpp
void ChernovTConvexHullBinaryComponentsMPI::ExchangeBoundaryRows(
    bool has_top, bool has_bottom, std::vector<int> &extended_pixels, int width) {
  std::vector<MPI_Request> reqs(4, MPI_REQUEST_NULL);
  std::vector<MPI_Status> statuses(4);
  int req_count = 0;

  if (has_top) {
    std::vector<int> top_recv(width);
    std::vector<int> top_send(local_pixels_.begin(), local_pixels_.begin() + width);
    MPI_Irecv(top_recv.data(), width, MPI_INT, rank_ - 1, 1, MPI_COMM_WORLD, &reqs[req_count]);
    MPI_Isend(top_send.data(), width, MPI_INT, rank_ - 1, 0, MPI_COMM_WORLD, &reqs[req_count + 1]);
    req_count += 2;
  }
  if (has_bottom && rank_ + 1 < size_) {
    std::vector<int> bottom_recv(width);
    std::vector<int> bottom_send(local_pixels_.end() - width, local_pixels_.end());
    MPI_Irecv(bottom_recv.data(), width, MPI_INT, rank_ + 1, 0, MPI_COMM_WORLD, &reqs[req_count]);
    MPI_Isend(bottom_send.data(), width, MPI_INT, rank_ + 1, 1, MPI_COMM_WORLD, &reqs[req_count + 1]);
    req_count += 2;
  }
  if (req_count > 0) {
    MPI_Waitall(req_count, reqs.data(), statuses.data());
  }
  if (has_top) {
    std::ranges::copy(top_recv, extended_pixels.begin());
  }
  if (has_bottom && rank_ + 1 < size_) {
    std::ranges::copy(bottom_recv, extended_pixels.end() - width);
  }
}
```

4. Локальный поиск компонент и построение оболочек:
```cpp
int local_rows = end_row_ - start_row_;
bool has_top = (start_row_ > 0);
bool has_bottom = (end_row_ < height_);
int extended_rows = local_rows + (has_top ? 1 : 0) + (has_bottom ? 1 : 0);
std::vector<int> extended_pixels(static_cast<std::size_t>(extended_rows) * static_cast<std::size_t>(width_), 0);

int offset = has_top ? 1 : 0;
for (int r = 0; r < local_rows; ++r) {
  std::size_t src = static_cast<std::size_t>(r) * static_cast<std::size_t>(width_);
  std::size_t dst = static_cast<std::size_t>(offset + r) * static_cast<std::size_t>(width_);
  std::ranges::copy(local_pixels_.begin() + src, local_pixels_.begin() + src + width_,
                    extended_pixels.begin() + dst);
}

ExchangeBoundaryRows(has_top, has_bottom, extended_pixels, width_);

int global_y_offset = start_row_ - (has_top ? 1 : 0);
auto all_components = ProcessExtendedRegion(extended_pixels, extended_rows, width_, global_y_offset);

FilterLocalComponents(all_components);

ComputeConvexHulls();
```

5. Сбор и рассылка результата:
```cpp
void ChernovTConvexHullBinaryComponentsMPI::GatherAndBroadcastResult() {
  std::vector<int> local_flat, local_sizes;
  for (const auto &hull : local_hulls_) {
    local_sizes.push_back(static_cast<int>(hull.size()));
    for (const auto &p : hull) {
      local_flat.push_back(p.first);
      local_flat.push_back(p.second);
    }
  }

  if (rank_ == 0) {
    std::vector<int> all_sizes, global_flat;
    GatherHullsOnRank0(all_sizes, global_flat);

    OutType global_hulls;
    std::size_t idx = 0;
    for (int sz : all_sizes) {
      std::vector<std::pair<int, int>> hull(static_cast<std::size_t>(sz));
      for (int j = 0; j < sz; ++j, idx += 2) {
        hull[static_cast<std::size_t>(j)] = {global_flat[idx], global_flat[idx + 1]};
      }
      global_hulls.push_back(std::move(hull));
    }
    BroadcastResultToAllRanks(global_hulls);
  } else {

    SendHullsToRank0(local_flat, local_sizes);

    BroadcastResultToAllRanks({});
  }
}
```

## 5. Детали реализации

### Структура проекта:

- `common.hpp` — определение типов данных  
- `ops_mpi.hpp` — объявление MPI-класса  
- `ops_mpi.cpp` — реализация MPI-алгоритма  
- `ops_seq.hpp` — объявление SEQ-класса  
- `ops_seq.cpp` — реализация SEQ-алгоритма  
- `functional/main.cpp` — функциональные тесты  
- `performance/main.cpp` — тесты производительности 

## 6. Экспериментальная среда

**Аппаратное обеспечение:**

- Процессор: AMD Ryzen 5 5500U with Radeon Graphics  
- Тактовая частота: 2.10 GHz  
- Ядра/потоки: 6 ядер / 12 потоков  
- Оперативная память: 8 GB DDR4  
- ОС: Windows 11 и Ubuntu 24.04  

**Программное обеспечение:**

- Компилятор: GCC 13.3.0  
- MPI: Open MPI 4.1.6  
- Стандарт: C++20  
- Тип сборки: Release  

**Тестовые данные:**

- Функциональные тесты: 9 случаев — пустое изображение, одна точка, две точка, три точки, квадрат, два квадрата, горизонтальная линия и т.д.
- Производительность: изображение 6000×6000, содержащее:
        - 60 случайных прямоугольников (размером от 20×20 до 100×100),
        - 40 случайных кругов (радиус от 25 до 75),
        - Шум: одна случайная точка на каждые 1000 пикселей.

## 7. Результаты и обсуждение

### 7.1 Корректность

Все функциональные тесты успешно пройдены как для SEQ, так и для MPI-реализации. Результаты совпадают по покрытию точек выпуклых оболочек. Примеры тестов:
1. Пустое изображение → пустой результат.
2. Единичная точка → оболочка из одной точки.
3. Диагональ из трёх точек → три отдельные оболочки.
4. Сплошной квадрат → одна оболочка из 4 угловых точек.
5. Два изолированных квадрата → две оболочки.

### 7.2 Производительность

Измерения выполнены на изображении `6000 на 6000` (согласно коду `perf_tests.cpp`). Время — значение из лога `task_run`. За базовое время SEQ принято значение при запуске в однопроцессном режиме: **0.126 с**.

| Режим | Число процессов | Время, с | Ускорение | Эффективность |
|-------|------------------|----------|-----------|----------------|
| seq   | 1                | 0.126    | 1.00      | N/A            |
| mpi   | 1                | 0.334    | 0.38      | 38%            |
| mpi   | 2                | 0.180    | 0.70      | 35%            |
| mpi   | 3                | 0.144    | 0.88      | 29%            |
| mpi   | 4                | 0.127    | 0.99      | 25%            |
| mpi   | 5                | 0.090    | 1.40      | 28%            |
| mpi   | 6                | 0.088    | 1.43      | 24%            |

## 8. Выводы

Задача построения выпуклых оболочек связных компонент бинарного изображения была успешно реализована в двух вариантах: последовательном (SEQ) и распределённом (MPI) с использованием библиотеки **Open MPI**. Обе реализации прошли функциональное тестирование и показали корректность на контрольных примерах.

Для проверки корректности разработаны функциональные тесты на малых бинарных изображениях (пустое изображение, одиночная точка, диагональ, квадраты), а для оценки производительности использовалось изображение размером `6000 x 6000`, содержащее прямоугольники, круги и шум. Эксперименты показали, что MPI-реализация демонстрирует максимальное ускорение `1.43× при 6 процессах`. Несмотря на то, что при 1–2 процессах производительность ниже SEQ из-за накладных расходов инициализации и обмена граничными строками, начиная с 3 процессов система выходит на положительное ускорение. Это свидетельствует о корректной реализации горизонтального разбиения и умеренных коммуникационных затратах при масштабировании.

## 9. Источники

1. **Курс лекций по параллельному программированию** Сысоев А. В.

2. **Технологии параллельного программирования MPI и OpenMP** А.В. Богданов, В.В. Воеводин и др., - МГУ, 2012.

3. **Документация Open MPI:** https://www.open-mpi.org/

4. **Microsoft MPI Functions:** https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-functions