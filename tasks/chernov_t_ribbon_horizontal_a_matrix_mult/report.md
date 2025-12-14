# Ленточная горизонтальная схема - разбиение только матрицы А - умножение матрицы на матрицу

- **Студент**: Чернов Тимур Владимирович, группа 3823Б1ПР1
- **Технология**: SEQ | MPI
- **Вариант**: 13

## 1. Введение

Умножение матриц - одна из самых фундаментальных операций в линейной алгебре и численных методах, активно используемая в машинном обучении, компьютерной графике,научных вычислениях и многих других аспектах компьютерных наук. При работе с большими матрицами последовательные реализации становятся узким местом по времени выполнения. В данной работе реализованы и сравнены две версии умножения матриц: последовательная (SEQ) и распределённая по MPI с горизонтальным ленточным разбиением первой матрицы.

## 2. Постановка задачи
**Описание задачи**

Даны две целочисленные матрицы - матрица **A** размера `m на n` и матрица **B** размера `n на p`. Требуется вычислить их произведение `C = A * B`, где результирующая матрица **C** будет иметь размер `m на p`.

### Формат данных:

Входные типы данных: размеры матрицы m на n типа int, вектор элементов матрицы A типа int и аналогично данные матрицы B - размеры матрицы n на p типа int и вектор элементов матрицы B типа int.
```cpp
using InType = std::tuple<int, int, std::vector<int>, int, int, std::vector<int>>;
```
Выходной тип данных: вектор значений типа int, представляющий собой элементы матрицы C размером m * p.
```cpp
using OutType = std::vector<int>;
```

### Ограничения:

- Матрицы представлены в виде одномерных векторов в row-major порядке  
- Умножение возможно только если `colsA == rowsB`
- Все размеры строго положительны
- Размеры матриц должны соответствовать количеству элементов в векторах данных  
- Алгоритм должен корректно обрабатывать матрицы различных размерностей 

## 3. Базовый последовательный алгоритм (SEQ)

Последовательный алгоритм реализует стандартное умножение матриц - тройное вложение циклов.

Код алгоритма:

```cpp
for (int i = 0; i < rows_a; i++) {
    for (int j = 0; j < cols_b; j++) {
      int sum = 0;
      for (int k = 0; k < cols_a; k++) {
        sum += matrix_a[(i * cols_a) + k] * matrix_b[(k * cols_b) + j];
      }
      output[(i * cols_b) + j] = sum;
    }
  }
```

## 4. Схема распараллеливания (MPI)

В параллельной реализации используется ленточное горизонтальное разбиение. Исходная матрица A разбивается по строкам на блоки, которые распределяются между процессами. Матрица B полностью дублируется на каждом процессе, по средством широковещательной рассылки — MPI_Bcast. Каждый процесс вычисляет свой блок строк результирующей матрицы C, выполняя локальное умножение своей части A на всю матрицу B.

**Этапы выполнения:**

1. Рассылка размеров (BroadcastMatrixSizes): процесс 0 отправляет всем размеры обеих матриц через MPI_Bcast.
2. Рассылка матрицы B (BroadcastMatrixB): вся матрица B раздаётся всем процессам через MPI_Bcast.
3. Разделение матрицы A (ScatterMatrixA): строки матрицы A распределяются между процессами с использованием MPI_Scatterv с учётом возможного остатка (если m не делится нацело на число процессов).
4. Локальное умножение (ComputeLocalC): каждый процесс умножает свои строки A на всю матрицу B.
5. Сбор результата (GatherResult): результаты собираются в процесс 0 с помощью MPI_Gatherv и затем рассылаются всем процессам через MPI_Bcast для соблюдения семантики задачи.

**Детальный алгоритм параллельной реализации**

Рассылка размеров матриц:
```cpp
void ChernovTRibbonHorizontalAMmatrixMultMPI::BroadcastMatrixSizes(int rank) {
  std::array<int, 4> sizes{};
  if (rank == 0) {
    sizes[0] = rowsA_;
    sizes[1] = colsA_;
    sizes[2] = rowsB_;
    sizes[3] = colsB_;
  }

  MPI_Bcast(sizes.data(), sizes.size(), MPI_INT, 0, MPI_COMM_WORLD);
  global_rowsA_ = sizes[0];
  global_colsA_ = sizes[1];
  global_rowsB_ = sizes[2];
  global_colsB_ = sizes[3];
}
```

Рассылка матрицы B:
```cpp
void ChernovTRibbonHorizontalAMmatrixMultMPI::BroadcastMatrixB(int rank) {
  if (rank != 0) {
    matrixB_.resize(static_cast<size_t>(global_rowsB_) * static_cast<size_t>(global_colsB_));
  }
  MPI_Bcast(matrixB_.data(), global_rowsB_ * global_colsB_, MPI_INT, 0, MPI_COMM_WORLD);
}
```

Разделение матрицы A:
```cpp
std::vector<int> ChernovTRibbonHorizontalAMmatrixMultMPI::ScatterMatrixA(int rank, int size) {
  int base_rows = global_rowsA_ / size;
  int remainder = global_rowsA_ % size;

  int local_rows = base_rows + (rank < remainder ? 1 : 0);
  int local_elements = static_cast<int>(static_cast<size_t>(local_rows) * static_cast<size_t>(global_colsA_));

  std::vector<int> local_a(local_elements);
  std::vector<int> sendcounts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
      int rows_for_i = base_rows + (i < remainder ? 1 : 0);
      sendcounts[i] = rows_for_i * global_colsA_;
      displacements[i] = offset;
      offset += sendcounts[i];
    }
  }

  std::vector<int> recvcounts(size);
  if (rank == 0) {
    recvcounts = sendcounts;
  }
  MPI_Bcast(recvcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Scatterv(rank == 0 ? matrixA_.data() : nullptr, rank == 0 ? sendcounts.data() : nullptr,
               rank == 0 ? displacements.data() : nullptr, MPI_INT, local_a.data(), local_elements, MPI_INT, 0,
               MPI_COMM_WORLD);

  return local_a;
}
```

Локальное умножение:
```cpp
std::vector<int> ChernovTRibbonHorizontalAMmatrixMultMPI::ComputeLocalC(int local_rows,
                                                                        const std::vector<int> &local_a) {
  std::vector<int> local_c(static_cast<size_t>(local_rows) * static_cast<size_t>(global_colsB_), 0);
  for (int i = 0; i < local_rows; i++) {
    for (int j = 0; j < global_colsB_; j++) {
      int sum = 0;
      for (int k = 0; k < global_colsA_; k++) {
        sum += local_a[(i * global_colsA_) + k] * matrixB_[(k * global_colsB_) + j];
      }
      local_c[(i * global_colsB_) + j] = sum;
    }
  }
  return local_c;
}
```

Сбор и рассылка результата:
```cpp
void ChernovTRibbonHorizontalAMmatrixMultMPI::GatherResult(int rank, int size, const std::vector<int> &local_c) {
  int base_rows = global_rowsA_ / size;
  int remainder = global_rowsA_ % size;

  std::vector<int> recvcounts(size);
  std::vector<int> displacements(size);

  if (rank == 0) {
    int offset = 0;
    for (int i = 0; i < size; i++) {
      int rows_for_i = base_rows + (i < remainder ? 1 : 0);
      recvcounts[i] = rows_for_i * global_colsB_;
      displacements[i] = offset;
      offset += recvcounts[i];
    }
  }
  MPI_Bcast(recvcounts.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(displacements.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

  std::vector<int> result(static_cast<size_t>(global_rowsA_) * static_cast<size_t>(global_colsB_));
  MPI_Gatherv(local_c.data(), static_cast<int>(local_c.size()), MPI_INT, result.data(), recvcounts.data(),
              displacements.data(), MPI_INT, 0, MPI_COMM_WORLD);

  MPI_Bcast(result.data(), global_rowsA_ * global_colsB_, MPI_INT, 0, MPI_COMM_WORLD);

  GetOutput() = result;
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

- Функциональные тесты: умножение матриц 2x3 и 3x3, а так же 3x2 и 2x4 с известными результатами.
- Производительность: умножение матриц 1100x1100.

## 7. Результаты и обсуждение

### 7.1 Корректность

Все функциональные тесты успешно пройдены. Обе реализации выдают идентичные результаты для тестовых матриц. Тестовыми данными являлись матрицы 2 на 3 и 3 на 3, а так же матрицы 3 на 2 и 2 на 4 в двух файлах matrix_1.txt и matrix_2.txt, содержащих размер матриц и элементы из которых состояли матриц.

### 7.2 Производительность

Измерения выполнены на матрицах `1100 на 1100` (согласно коду `perf_tests.cpp`). Время — значение из лога `task_run`. За базовое время SEQ принято значение при запуске в однопроцессном режиме: **3.7263 с**.

| Режим | Число процессов | Время, с | Ускорение | Эффективность |
|-------|------------------|----------|-----------|----------------|
| seq   | 1                | 3.7263   | 1.00      | N/A            |
| mpi   | 1                | 0.6653   | 5.60      | 560%           |
| mpi   | 2                | 0.9737   | 3.83      | 191%           |
| mpi   | 3                | 0.8979   | 4.15      | 138%           |
| mpi   | 4                | 0.7978   | 4.67      | 117%           |
| mpi   | 5                | 0.7452   | 5.00      | 100%           |
| mpi   | 6                | 0.7026   | 5.30      | 88%            |

## 8. Выводы

Задача умножения матриц с горизонтальным ленточным разбиением одной матрицы была успешно реализована в двух вариантах: **последовательном (SEQ)** и **распределённом (MPI)** с использованием библиотеки **Open MPI**. Обе реализации прошли функциональное тестирование и показали корректность на контрольных примерах.


Для проверки корректности разработаны функциональные тесты на малых матрицах, а для оценки производительности использовались две квадратные матрицы размером `1100 на 1100`. Эксперименты показали, что MPI-реализация демонстрирует **максимальное ускорение 5.30×** при 6 процессах. При этом наилучшая параллельная эффективность **191%** наблюдается при 2 процессах. Начиная с 3 процессов эффективность снижается, достигая 88% при 6 процессах, что свидетельствует о хорошей масштабируемости и умеренных коммуникационных накладных расходах.

## 9. Источники

1. **Курс лекций по параллельному программированию** Сысоев А. В.

2. **Технологии параллельного программирования MPI и OpenMP** А.В. Богданов, В.В. Воеводин и др., - МГУ, 2012.

3. **Документация Open MPI:** https://www.open-mpi.org/

4. **Microsoft MPI Functions:** https://learn.microsoft.com/ru-ru/message-passing-interface/mpi-functions