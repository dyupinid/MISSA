#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>


// Параметры алгоритма
const double ST = 0.8;      // Пороговое значение для выбора стратегии
const double PD = 0.2;      // Доля производителей
const double SD = 0.2;      // Доля предотвратителей опасности
const int POP_SIZE = 50;    // Размер популяции
const int MAX_ITER = 1000;  // Максимальное число итераций
const int DIM = 2;          // Размерность задачи
const double LB = -10;      // Нижняя граница поиска
const double UB = 10;       // Верхняя граница поиска

// Глобальная лучшая особь
std::vector<double> best_individual;

// Целевая функция (Sphere)
double sphere(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

double rastrigin(const std::vector<double>& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += (xi * xi - 10 * std::cos(2 * M_PI * xi) + 10);
    }
    return sum;
}

double rosenbrock(const std::vector<double>& x) {
    return std::pow(1 - x[0], 2) + 100 * std::pow(x[1] - std::pow(x[0], 2), 2);
}

// Вычисление приспособленности
double fitness(const std::vector<double>& individual) {
    return sphere(individual);
}

// Инициализация популяции
std::vector<std::vector<double>> initialize_population(std::mt19937& gen) {
    std::uniform_real_distribution<double> dis(LB, UB);
    std::vector<std::vector<double>> population(POP_SIZE, std::vector<double>(DIM));
    for (auto& ind : population) {
        for (auto& val : ind) {
            val = dis(gen);
        }
    }
    return population;
}

// ISS: Улучшенная стратегия поиска (уравнения 8 и 9)
std::vector<double> iss_strategy(const std::vector<double>& X, int t, int T, std::mt19937& gen) {
    // Динамический коэффициент n
    double n = std::exp(-static_cast<double>(t)) * std::exp(-20 * std::pow(static_cast<double>(t) / T, 2));
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    double R2 = uni(gen);
    if (R2 < ST) {
        // Стратегия исследования
        std::vector<double> res(DIM);
        for (int i = 0; i < DIM; ++i) {
            res[i] = X[i] + n * (2 * uni(gen) - 1) * X[i];
        }
        return res;
    }
    else {
        // Стратегия эксплуатации
        std::normal_distribution<double> norm(0.0, 1.0);
        double Q = norm(gen);
        std::vector<double> res(DIM);
        for (int i = 0; i < DIM; ++i) {
            res[i] = X[i] + Q * (best_individual[i] - X[i]);
        }
        return res;
    }
}

// GFS: Групповое следование (аналогично алгоритму COOT)
std::vector<double> gfs_strategy(const std::vector<double>& X, const std::vector<std::vector<double>>& leaders, std::mt19937& gen) {
    std::uniform_int_distribution<int> int_dis(0, static_cast<int>(leaders.size()) - 1);
    int idx = int_dis(gen);
    const auto& leader = leaders[idx];
    std::uniform_real_distribution<double> uni(0.0, 1.0);
    std::vector<double> res(DIM);
    for (int i = 0; i < DIM; ++i) {
        res[i] = X[i] + uni(gen) * (leader[i] - X[i]);
    }
    return res;
}

// ROBLS: Случайная противоположная стратегия
std::vector<double> rob_ls_strategy(const std::vector<double>& X) {
    std::vector<double> opposite(DIM);
    for (int i = 0; i < DIM; ++i) {
        opposite[i] = LB + UB - X[i];
    }
    if (fitness(opposite) < fitness(X)) {
        return opposite;
    }
    return X;
}

// Обрезка значений по границам
double clamp(double x) {
    return std::max(LB, std::min(UB, x));
}

// Основной алгоритм MISSA
std::vector<double> missa() {
    std::random_device rd;
    std::mt19937 gen(rd());

    auto population = initialize_population(gen);
    std::vector<double> fitness_values(POP_SIZE);
    for (int i = 0; i < POP_SIZE; ++i) {
        fitness_values[i] = fitness(population[i]);
    }
    auto min_it = std::min_element(fitness_values.begin(), fitness_values.end());
    int best_index = std::distance(fitness_values.begin(), min_it);
    best_individual = population[best_index];

    for (int t = 0; t < MAX_ITER; ++t) {
        // Сортировка популяции по приспособленности
        std::vector<std::pair<std::vector<double>, double>> sorted_pop(POP_SIZE);
        for (int i = 0; i < POP_SIZE; ++i) {
            sorted_pop[i] = { population[i], fitness_values[i] };
        }
        std::sort(sorted_pop.begin(), sorted_pop.end(), [](const auto& a, const auto& b) {
            return a.second < b.second;
            });

        for (int i = 0; i < POP_SIZE; ++i) {
            population[i] = sorted_pop[i].first;
            fitness_values[i] = sorted_pop[i].second;
        }

        // Обновление позиций производителей (ISS)
        int num_producers = static_cast<int>(PD * POP_SIZE);
        for (int i = 0; i < num_producers; ++i) {
            population[i] = iss_strategy(population[i], t, MAX_ITER, gen);
            for (auto& xi : population[i]) {
                xi = clamp(xi);
            }
        }

        // Обновление позиций искателей (GFS)
        std::vector<std::vector<double>> leaders(population.begin(), population.begin() + static_cast<int>(SD * POP_SIZE));
        for (int i = num_producers; i < POP_SIZE; ++i) {
            population[i] = gfs_strategy(population[i], leaders, gen);
            for (auto& xi : population[i]) {
                xi = clamp(xi);
            }
        }

        // Противоположная стратегия (ROBLS)
        for (int i = 0; i < POP_SIZE; ++i) {
            population[i] = rob_ls_strategy(population[i]);
            for (auto& xi : population[i]) {
                xi = clamp(xi);
            }
        }

        // Обновление приспособленностей
        for (int i = 0; i < POP_SIZE; ++i) {
            fitness_values[i] = fitness(population[i]);
        }

        // Обновление лучшей особи
        min_it = std::min_element(fitness_values.begin(), fitness_values.end());
        double current_best = *min_it;
        if (current_best < fitness(best_individual)) {
            int idx = std::distance(fitness_values.begin(), min_it);
            best_individual = population[idx];
        }
    }

    return best_individual;
}

int main() {
    auto best_solution = missa();
    double best_fitness_value = fitness(best_solution);

    std::cout << "Лучшее решение:";
    for (double val : best_solution) {
        std::cout << " " << std::setprecision(6) << val;
    }
    std::cout << std::endl;
    std::cout << "Значение функции: " << std::setprecision(6) << best_fitness_value << std::endl;

    return 0;
}