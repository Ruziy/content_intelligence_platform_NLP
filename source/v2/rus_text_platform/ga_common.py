"""Ядро генетического алгоритма на DEAP для модулей платформы.

Каждый модуль декларирует:
    - search_space: список Gene
    - fitness_fn(params: dict) -> float (maximize)

И вызывает run_ga(search_space, fitness_fn, config, on_progress=...).

Поддерживаются три типа генов: Categorical, IntRange, FloatRange. Внутренне
индивид хранится как list[float] длины len(search_space); декодирование в
читаемый dict params делается перед каждым вызовом fitness.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from deap import base, creator, tools


# -----------------------------
# Декларация search space
# -----------------------------


@dataclass
class Categorical:
    name: str
    choices: List[Any]

    def sample(self) -> float:
        return float(random.randint(0, len(self.choices) - 1))

    def decode(self, gene: float) -> Any:
        idx = int(round(gene)) % len(self.choices)
        if idx < 0:
            idx += len(self.choices)
        return self.choices[idx]

    def mutate(self, gene: float) -> float:
        return self.sample()


@dataclass
class IntRange:
    name: str
    low: int
    high: int  # включительно

    def sample(self) -> float:
        return float(random.randint(self.low, self.high))

    def decode(self, gene: float) -> int:
        value = int(round(gene))
        return max(self.low, min(self.high, value))

    def mutate(self, gene: float) -> float:
        # ±1 шаг с шансом, иначе ресэмпл
        if random.random() < 0.5:
            step = random.choice([-1, 1])
            value = self.decode(gene) + step
            return float(max(self.low, min(self.high, value)))
        return self.sample()


@dataclass
class FloatRange:
    name: str
    low: float
    high: float

    def sample(self) -> float:
        return random.uniform(self.low, self.high)

    def decode(self, gene: float) -> float:
        return max(self.low, min(self.high, float(gene)))

    def mutate(self, gene: float) -> float:
        # гауссовский шаг ширины (high-low)*0.1
        sigma = (self.high - self.low) * 0.1
        return self.decode(gene + random.gauss(0.0, sigma))


Gene = Categorical | IntRange | FloatRange


# -----------------------------
# Конфиг ГА
# -----------------------------


@dataclass
class GAConfig:
    population_size: int = 10
    generations: int = 5
    cxpb: float = 0.6
    mutpb: float = 0.3
    tournament_size: int = 3
    elitism: int = 1
    seed: Optional[int] = 42


# -----------------------------
# Результат
# -----------------------------


@dataclass
class GAResult:
    best_params: Dict[str, Any]
    best_fitness: float
    history: List[Dict[str, float]] = field(default_factory=list)
    runtime_s: float = 0.0
    evaluations: int = 0
    ga_config: Dict[str, Any] = field(default_factory=dict)


# -----------------------------
# Хелперы
# -----------------------------


def decode_individual(individual: Sequence[float], space: Sequence[Gene]) -> Dict[str, Any]:
    return {gene.name: gene.decode(individual[i]) for i, gene in enumerate(space)}


def _ensure_creator():
    # creator.create — глобальный side-effect, нельзя дублировать.
    if not hasattr(creator, "GAFitnessMax"):
        creator.create("GAFitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "GAIndividual"):
        creator.create("GAIndividual", list, fitness=creator.GAFitnessMax)


def _make_toolbox(space: Sequence[Gene], fitness_fn: Callable[[Dict[str, Any]], float]) -> base.Toolbox:
    _ensure_creator()
    toolbox = base.Toolbox()

    def init_individual():
        return creator.GAIndividual([gene.sample() for gene in space])

    def mutate(individual):
        for i, gene in enumerate(space):
            if random.random() < 1.0 / len(space):
                individual[i] = gene.mutate(individual[i])
        return (individual,)

    def evaluate(individual):
        params = decode_individual(individual, space)
        score = fitness_fn(params)
        return (float(score),)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutate)
    toolbox.register("evaluate", evaluate)
    return toolbox


# -----------------------------
# Основной запуск
# -----------------------------


ProgressCb = Callable[[Dict[str, Any]], None]


def run_ga(
    search_space: Sequence[Gene],
    fitness_fn: Callable[[Dict[str, Any]], float],
    config: Optional[GAConfig] = None,
    on_progress: Optional[ProgressCb] = None,
) -> GAResult:
    """Запускает (μ+λ)-подобный цикл с турнирной селекцией и элитизмом.

    on_progress вызывается после каждого поколения со словарём:
        {generation, best_fitness, avg_fitness, best_params, evaluations}
    """
    config = config or GAConfig()
    if config.seed is not None:
        random.seed(config.seed)

    toolbox = _make_toolbox(search_space, fitness_fn)
    toolbox.register("select", tools.selTournament, tournsize=config.tournament_size)

    start = time.perf_counter()
    population = toolbox.population(n=config.population_size)

    # Начальная оценка
    evaluations = 0
    for ind in population:
        ind.fitness.values = toolbox.evaluate(ind)
        evaluations += 1

    history: List[Dict[str, float]] = []

    def _report(generation: int):
        fits = [ind.fitness.values[0] for ind in population]
        best_idx = max(range(len(population)), key=lambda i: fits[i])
        best_ind = population[best_idx]
        entry = {
            "generation": generation,
            "best_fitness": float(fits[best_idx]),
            "avg_fitness": float(sum(fits) / len(fits)),
            "best_params": decode_individual(best_ind, search_space),
            "evaluations": evaluations,
        }
        history.append(entry)
        if on_progress:
            on_progress(entry)

    _report(0)

    for gen in range(1, config.generations + 1):
        # Элитизм
        elites = tools.selBest(population, config.elitism) if config.elitism > 0 else []
        elites = [toolbox.clone(e) for e in elites]

        # Селекция и потомство
        offspring = toolbox.select(population, len(population) - len(elites))
        offspring = [toolbox.clone(ind) for ind in offspring]

        # Кроссовер
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < config.cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        # Мутация
        for mutant in offspring:
            if random.random() < config.mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Оценка только новых
        for ind in offspring:
            if not ind.fitness.valid:
                ind.fitness.values = toolbox.evaluate(ind)
                evaluations += 1

        population = elites + offspring
        _report(gen)

    runtime = time.perf_counter() - start
    best = tools.selBest(population, 1)[0]
    return GAResult(
        best_params=decode_individual(best, search_space),
        best_fitness=float(best.fitness.values[0]),
        history=history,
        runtime_s=runtime,
        evaluations=evaluations,
        ga_config={
            "population_size": config.population_size,
            "generations": config.generations,
            "cxpb": config.cxpb,
            "mutpb": config.mutpb,
            "tournament_size": config.tournament_size,
            "elitism": config.elitism,
            "seed": config.seed,
        },
    )
