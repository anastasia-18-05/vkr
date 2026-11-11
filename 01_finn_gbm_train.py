"""FINN GBM training and isolated seed workers; importing never starts training."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import hashlib
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import platform
import random
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
from scipy.special import ndtr
import torch
from torch import nn

@dataclass
class Config:
    seed: int = 42  # Начальный seed; серия и усреднение задаются через run_seeds.
    data_seed: int | None = None
    sigma: float = 0.15
    spot: float = 100.0  # S0; риск-нейтральный GBM, mu=r=0, без дивидендов.
    dt: float = 1 / 250  # Один торговый день.
    max_tau: float = 2.0  # Верхняя граница по Figure 2
    # Согласовано: m=S/K ~ U(0.5,2), K=S/m.
    m_min: float = 0.5
    m_max: float = 2.0
    train_paths: int = 800  # 800 * 100 = 80 000 фиксированных примеров.
    validation_paths: int = 200  # Отдельные траектории: 200 * 100 = 20 000.
    transitions_per_path: int = 100  # Из 500 шагов выбираем без повторений.
    test_paths: int = 256  # Пакет независимых траекторий для набора 10 000 точек.
    test_samples: int = 10000  # Те же пары (S,K) при каждом из семи сроков.
    test_spot_min: float = 75.0  # Отбор состояний S, без обрезания GBM-цен.
    test_spot_max: float = 125.0
    test_strike_min: float = 75.0
    test_strike_max: float = 125.0
    batch_size: int = 512
    epochs: int = 250
    steps_per_epoch: int = 100  # Батчи повторно выбираются из фиксированного train.
    learning_rate: float = 5e-4
    patience: int = 40
    lr_final_ratio: float = 0.1  # lr в последней эпохе / начальный lr
    min_tau: float = 0.0  # Равномерный срок от 0 до max_tau, без добавленной массы у нуля.
    path_horizon: float = 2.0  # t от 0 до 2 лет: ровно 500 переходов.
    sampling: str = "spot_over_strike"  # Последнее согласование: равномерно S/K, не K/S.
    atm_fraction: float = 0.0  # Базовый опыт: без дополнительной массы около ATM
    transition_protocol: str = "uniform_tau_short_step_v1"  # h=min(dt,tau); новая серия данных.
    price_mask: float = 1e-4  # Авторы не раскрывают конкретный порог delta
    delta_mask: float = 1e-6
    surface_sigma: float = 0.15  # Только этот случай выводим на графиках.
    surface_tau: float = 0.36
    device: str = "cpu"
    threads: int = 4

    @property
    def effective_data_seed(self) -> int:
        return self.seed if self.data_seed is None else self.data_seed

    @property
    def path_steps(self) -> int:
        return int(round(self.path_horizon / self.dt))

    @property
    def train_samples(self) -> int:
        return self.train_paths * self.transitions_per_path

    @property
    def validation_samples(self) -> int:
        return self.validation_paths * self.transitions_per_path


class FINN(nn.Module):
    """Точно 2 -> 50 -> 50 -> 1; tanh, tanh, softplus; выход C/K."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 50), nn.Tanh(),
            nn.Linear(50, 50), nn.Tanh(),
            nn.Linear(50, 1), nn.Softplus(),
        )

    def forward(self, m: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        return self.layers(torch.cat((m, tau), dim=1))

    def price(self, s: torch.Tensor, k: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        raw_price = k * self(s / k, tau)
        # В экспирацию цена известна по контракту; отдельного terminal loss нет.
        return torch.where(tau <= 0, (s - k).clamp_min(0), raw_price)

    def bounded_price(self, s: torch.Tensor, k: torch.Tensor,
                      tau: torch.Tensor) -> torch.Tensor:
        # Только предсказание: нижняя граница для call при r=0.
        return torch.maximum(self.price(s, k, tau), (s - k).clamp_min(0))


DATA_COLUMNS = ("s", "s_next", "k", "tau", "tau_next")
SAMPLE_COLUMNS = (*DATA_COLUMNS, "h", "path_id", "step")


def simulate_gbm(cfg: Config, count: int, seed) -> np.ndarray:
    """Точный переход GBM (35), mu=r=0; seed также принимает Generator."""
    rng = np.random.default_rng(seed)
    increments = (-0.5 * cfg.sigma**2 * cfg.dt
                  + cfg.sigma * math.sqrt(cfg.dt) * rng.standard_normal((count, cfg.path_steps)))
    logs = np.concatenate((np.zeros((count, 1)), np.cumsum(increments, axis=1)), axis=1)
    return cfg.spot * np.exp(logs)


def gbm_step_to_maturity(s: np.ndarray, daily_next: np.ndarray, tau: np.ndarray,
                         cfg: Config, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GBM в момент h=min(dt,tau), согласованный с уже созданной дневной траекторией."""
    h = np.minimum(cfg.dt, tau)
    sn = daily_next.copy()
    short = h < cfg.dt
    # Условный лог-GBM внутри дня: среднее a*log(S_day/S), дисперсия sigma²*h*(1-a).
    a = h[short] / cfg.dt
    log_move = (a * np.log(daily_next[short] / s[short])
                + cfg.sigma * np.sqrt(h[short] * (1-a)) * rng.standard_normal(int(short.sum())))
    sn[short] = s[short] * np.exp(log_move)
    # Сохраняем до float32, чтобы округление tau около dt не создавало ложную экспирацию.
    tau_next = np.maximum(tau - h, 0.0)
    return sn, h, tau_next


def make_dataset(cfg: Config, split: str) -> dict[str, np.ndarray]:
    """Равномерные сроки; 100 разных дневных интервалов на путь, короткие шаги до выплаты."""
    split_id, count = {"train": (0, cfg.train_paths),
                       "validation": (1, cfg.validation_paths)}[split]
    # Пути и контракты используют прежние потоки; третий поток достраивает короткие шаги.
    path_seed, sample_seed, bridge_seed = np.random.SeedSequence(
        [cfg.effective_data_seed, split_id]).spawn(3)
    paths = simulate_gbm(cfg, count, path_seed)
    rng = np.random.default_rng(sample_seed)
    bridge_rng = np.random.default_rng(bridge_seed)
    rows = np.repeat(np.arange(count), cfg.transitions_per_path)
    steps = np.concatenate([rng.choice(cfg.path_steps, cfg.transitions_per_path, replace=False)
                            for _ in range(count)])
    s, daily_next = paths[rows, steps], paths[rows, steps + 1]
    ratio = rng.uniform(cfg.m_min, cfg.m_max, s.size)
    if cfg.atm_fraction:
        ratio = np.where(rng.random(s.size) < cfg.atm_fraction,
                         rng.uniform(0.8, 1.2, s.size), ratio)
    k = s / ratio if cfg.sampling == "spot_over_strike" else s * ratio
    tau = rng.uniform(cfg.min_tau, cfg.max_tau, s.size)  # Никаких замен сроков или квот.
    sn, h, tau_next = gbm_step_to_maturity(s, daily_next, tau, cfg, bridge_rng)
    return {"s": s, "s_next": sn, "k": k, "tau": tau, "tau_next": tau_next,
            "h": h, "path_id": rows, "step": steps}


def fixed_dataset(cfg: Config, split: str, directory: Path) -> dict[str, np.ndarray]:
    """Создать выборку один раз или прочитать её из папки текущей конфигурации."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{split}_samples.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if stored["transition_protocol"].item() != cfg.transition_protocol:
                raise ValueError(f"Dataset transition protocol mismatch: {path}")
            data = {name: stored[name] for name in SAMPLE_COLUMNS}
        expected = cfg.train_samples if split == "train" else cfg.validation_samples
        if any(values.shape != (expected,) for values in data.values()):
            raise ValueError(f"Incorrect saved {split} dataset size")
        return data
    data = make_dataset(cfg, split)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, transition_protocol=np.array(cfg.transition_protocol), **data)
    temporary.replace(path)
    return data


def device_dataset(data: dict[str, np.ndarray], cfg: Config) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(data[name].reshape(-1, 1), dtype=torch.float32, device=cfg.device)
                 for name in DATA_COLUMNS)


def sample_batch(dataset: tuple[torch.Tensor, ...], size: int,
                 rng: torch.Generator) -> tuple[torch.Tensor, ...]:
    """Повторные обращения к фиксированному train не создают новых контрактов."""
    indices = torch.randint(dataset[0].shape[0], (size,), generator=rng, device=dataset[0].device)
    return tuple(values[indices] for values in dataset)


def losses(model: FINN, batch: tuple[torch.Tensor, ...], cfg: Config,
           training: bool) -> dict[str, torch.Tensor]:
    s, sn, k, tau, next_tau = batch
    s = s.detach().requires_grad_(True)
    price = model.price(s, k, tau)
    delta = torch.autograd.grad(price.sum(), s, create_graph=training,
                                retain_graph=training)[0]
    # next_tau=tau-min(dt,tau) сохранён вместе с соответствующей ценой S_next.
    # При tau_next=0 используется выплата; иначе C_next предсказывает сеть.
    with torch.set_grad_enabled(training):
        next_price = model.price(sn, k, next_tau)
    # Algorithm 1 при r=0: денежные C и одна Delta_t на обоих концах шага.
    residual = delta * (sn - s) - (next_price - price)
    return {"loss": residual.square().mean()}  # Только hedging loss, без множителей и штрафов.



def validation(model: FINN, batch: tuple[torch.Tensor, ...], cfg: Config) -> dict:
    """Delta требует autograd; итог переносится с устройства один раз за проход."""
    model.eval()
    total = torch.zeros((), dtype=torch.float64, device=cfg.device)
    count = batch[0].shape[0]
    with torch.enable_grad():
        for start in range(0, count, cfg.batch_size):
            part = tuple(t[start:start + cfg.batch_size] for t in batch)
            value = losses(model, part, cfg, training=False)["loss"]
            total += value.detach().double() * len(part[0])
    return {"loss": total.item() / count}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def read_history(path: Path) -> np.ndarray:
    return np.atleast_1d(np.genfromtxt(path, delimiter=",", names=True))

def load_checkpoint(path: Path, device: str, expected_config: Config | None = None) -> dict:
    saved = torch.load(path, map_location=device, weights_only=True)
    if expected_config is not None and saved["config"] != asdict(expected_config):
        raise ValueError(f"Checkpoint configuration mismatch: {path}")
    return saved

def restore_model(saved: dict, device: str) -> FINN:
    model = FINN().to(device)
    model.load_state_dict(saved["state_dict"])
    return model


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def train(cfg: Config, out: Path) -> FINN:
    started = time.perf_counter()
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        torch.cuda.manual_seed_all(cfg.seed)
    torch.use_deterministic_algorithms(True)
    model = FINN().to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    gamma = cfg.lr_final_ratio ** (1/max(cfg.epochs-1, 1))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    train_fixed = fixed_dataset(cfg, "train", out)
    val_fixed = fixed_dataset(cfg, "validation", out)
    train_data = device_dataset(train_fixed, cfg)
    val_batch = device_dataset(val_fixed, cfg)
    rng = torch.Generator(device=cfg.device).manual_seed(cfg.seed)
    write_json(out / "data_summary.json", {
        "seed": cfg.effective_data_seed, "path_steps": cfg.path_steps,
        "train_paths": cfg.train_paths, "validation_paths": cfg.validation_paths,
        "transitions_per_path": cfg.transitions_per_path,
        "train_samples": cfg.train_samples, "validation_samples": cfg.validation_samples,
        "sampling": cfg.sampling, "tau_range": [cfg.min_tau, cfg.max_tau],
        "fixed_contracts": True, "transition_protocol": cfg.transition_protocol,
        "maturity_sampling": "uniform; no forced expiry quota",
        "effective_step": "h=min(dt,tau); log-GBM Brownian bridge inside the existing daily interval",
        "train_expiry_transitions": int(np.count_nonzero(train_fixed["tau_next"] == 0)),
        "validation_expiry_transitions": int(np.count_nonzero(val_fixed["tau_next"] == 0)),
        "loss": "mean squared monetary hedging residual; exact payoff at tau_next=0; no added penalties",
        "prediction": "max(raw price, intrinsic); delta differentiates the bounded price",
    })
    best, stale, best_epoch = float("inf"), 0, 0
    fields = ["epoch", "learning_rate", "train_loss", "val_loss"]
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for epoch in range(1, cfg.epochs+1):
            model.train()
            lr = optimizer.param_groups[0]["lr"]
            average = torch.zeros((), device=cfg.device)
            for _ in range(cfg.steps_per_epoch):
                batch = sample_batch(train_data, cfg.batch_size, rng)
                optimizer.zero_grad(set_to_none=True)
                vals = losses(model, batch, cfg, training=True)
                vals["loss"].backward()
                optimizer.step()
                average += vals["loss"].detach()
            train_loss = (average/cfg.steps_per_epoch).item()
            val = validation(model, val_batch, cfg)
            if not math.isfinite(train_loss) or not math.isfinite(val["loss"]):
                raise FloatingPointError("Non-finite train/validation loss")
            writer.writerow(dict(zip(fields, [epoch, lr, train_loss, val["loss"]])))
            stream.flush()
            if val["loss"] < best:
                best, stale, best_epoch = val["loss"], 0, epoch
                torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                            "epoch": epoch, "validation": val}, out / "best_model.pt")
            else:
                stale += 1
            if epoch == 1 or epoch % 50 == 0 or epoch == cfg.epochs:
                print(f"sigma={cfg.sigma:.3f} seed={cfg.seed} epoch={epoch:4d} lr={lr:.2e} train={train_loss:.6g} "
                      f"val={val['loss']:.6g}", flush=True)
            scheduler.step()
            if stale >= cfg.patience:
                print(f"sigma={cfg.sigma:.3f} seed={cfg.seed}: early stopping at epoch {epoch}", flush=True)
                break
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg),
                "epoch": epoch, "validation": val}, out / "last_model.pt")
    write_json(out / "training_summary.json", {
        "best_epoch": best_epoch, "last_epoch": epoch, "best_validation_loss": best,
        "last_validation_loss": val["loss"], "optimizer_steps": epoch*cfg.steps_per_epoch,
        "elapsed_seconds": time.perf_counter()-started,
        "stop_reason": "patience" if stale >= cfg.patience else "epoch_limit",
    })
    saved = load_checkpoint(out / "best_model.pt", cfg.device, cfg)
    model.load_state_dict(saved["state_dict"])
    return model


def run_directory(directory: Path, cfg: Config) -> Path:
    return Path(directory) / f"sigma_{cfg.sigma:.3f}" / f"seed_{cfg.seed}"


def execution_settings(workers: int = 15, worker_threads: int = 1) -> dict:
    """Execution identity is separate from the unchanged scientific configuration."""
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    if type(worker_threads) is not int or worker_threads < 1:
        raise ValueError("worker_threads must be a positive integer")
    return {
        "backend": "spawn_seed_workers_v1",
        "workers": workers,
        "worker_threads": worker_threads,
        "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }


def checked_training_summary(folder: Path, cfg: Config) -> dict | None:
    """Only a fully written training result can bypass training after a restart."""
    folder = Path(folder)
    config_path = folder / "config.json"
    if config_path.exists() and read_json(config_path) != asdict(cfg):
        raise ValueError(f"Configuration mismatch: {folder}")
    summary_path = folder / "training_summary.json"
    if not summary_path.is_file():
        return None
    required = ("config.json", "best_model.pt", "last_model.pt", "history.csv",
                "train_samples.npz", "validation_samples.npz")
    missing = [name for name in required if not (folder / name).is_file()]
    if missing:
        raise ValueError(f"Incomplete training result in {folder}: {missing}")
    summary = read_json(summary_path)
    if not (1 <= summary["best_epoch"] <= summary["last_epoch"] <= cfg.epochs):
        raise ValueError(f"Invalid training epochs: {folder}")
    if summary["optimizer_steps"] != summary["last_epoch"] * cfg.steps_per_epoch:
        raise ValueError(f"Invalid optimizer step count: {folder}")
    for filename, epoch_key, loss_key in (
        ("best_model.pt", "best_epoch", "best_validation_loss"),
        ("last_model.pt", "last_epoch", "last_validation_loss"),
    ):
        saved = load_checkpoint(folder / filename, "cpu", cfg)
        if (saved["epoch"] != summary[epoch_key]
                or saved["validation"]["loss"] != summary[loss_key]
                or not math.isfinite(summary[loss_key])):
            raise ValueError(f"Checkpoint/summary mismatch: {folder / filename}")
        # Validate tensor names and shapes without initializing another model/RNG.
        shapes = {"layers.0.weight": (50, 2), "layers.0.bias": (50,),
                  "layers.2.weight": (50, 50), "layers.2.bias": (50,),
                  "layers.4.weight": (1, 50), "layers.4.bias": (1,)}
        state = saved["state_dict"]
        if set(state) != set(shapes) or any(tuple(state[k].shape) != v for k, v in shapes.items()):
            raise ValueError(f"Invalid FINN state: {folder / filename}")
    return summary


def train_job(config: dict, directory: str, worker_threads: int) -> dict:
    """Top-level picklable worker; tensors and RNG state never cross processes."""
    cfg = Config(**config)
    folder = run_directory(Path(directory), cfg)
    folder.mkdir(parents=True, exist_ok=True)
    summary = checked_training_summary(folder, cfg)
    if summary is not None:
        return {"sigma": cfg.sigma, "seed": cfg.seed, "reused": True}
    torch.set_num_threads(worker_threads)
    write_json(folder / "config.json", asdict(cfg))
    write_json(folder / "environment.json", {
        "python": platform.python_version(), "torch": str(torch.__version__),
        "numpy": np.__version__, "device": cfg.device,
        "worker_threads": worker_threads,
    })
    print(f"sigma={cfg.sigma:.3f} seed={cfg.seed}: обучение", flush=True)
    model = train(cfg, folder)
    del model
    checked_training_summary(folder, cfg)
    return {"sigma": cfg.sigma, "seed": cfg.seed, "reused": False}


def read_training_plan(directory: Path, workers: int, worker_threads: int) -> list[Config]:
    plan = read_json(Path(directory) / "study.json")
    if plan.get("execution") != execution_settings(workers, worker_threads):
        raise ValueError("Training script or worker settings differ from study.json; prepare a new study")
    environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                   "numpy": np.__version__}
    if plan.get("environment") != environment:
        raise ValueError("Training environment differs from study.json; prepare a new study")
    configurations = [Config(**value) for value in plan["configurations"]]
    destinations = [run_directory(directory, cfg) for cfg in configurations]
    if not configurations or len(set(destinations)) != len(destinations):
        raise ValueError("Study must contain distinct nonempty sigma/seed jobs")
    return configurations


def train_study(directory: Path, workers: int = 15, worker_threads: int = 1) -> dict:
    """Bounded queue of independent seed jobs. Evaluation is left to the notebook."""
    directory = Path(directory).resolve()
    configurations = read_training_plan(directory, workers, worker_threads)
    # Validate all existing results before launching any workers.
    pending = [cfg for cfg in configurations
               if checked_training_summary(run_directory(directory, cfg), cfg) is None]
    started = time.perf_counter()
    status = {"expected_runs": len(configurations), "skipped_runs": len(configurations)-len(pending),
              "trained_runs": 0, "workers": workers, "worker_threads": worker_threads,
              "status": "running", "elapsed_seconds": 0.0}
    status_path = directory / "training_status.json"
    write_json(status_path, status)
    executor = None
    futures = []
    try:
        if workers == 1:
            for cfg in pending:
                result = train_job(asdict(cfg), str(directory), worker_threads)
                status["trained_runs"] += 1
                print(f"Обучено {status['trained_runs']}/{len(pending)}: "
                      f"sigma={result['sigma']:.3f}, seed={result['seed']}", flush=True)
        elif pending:
            executor = ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                           mp_context=mp.get_context("spawn"))
            futures = [executor.submit(train_job, asdict(cfg), str(directory), worker_threads)
                       for cfg in pending]
            for future in as_completed(futures):
                result = future.result()
                status["trained_runs"] += 1
                print(f"Обучено {status['trained_runs']}/{len(pending)}: "
                      f"sigma={result['sigma']:.3f}, seed={result['seed']}", flush=True)
        status["status"] = "complete"
    except BaseException as error:
        for future in futures:
            future.cancel()
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        status["elapsed_seconds"] = time.perf_counter() - started
        write_json(status_path, status)
    return status


TABLE_MATURITIES = (0.24, 0.28, 0.32, 0.36, 0.40, 0.44, 0.48)
METRICS = ("RMAD", "RMSE_relative", "NMAD", "MSE")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def completed_report(directory: Path, cfg: Config) -> dict | None:
    folder = run_directory(directory, cfg)
    required = ("complete.json", "metrics.json", "best_model.pt", "training_summary.json")
    if not all((folder / name).is_file() for name in required):
        return None
    marker = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if marker["config"] != asdict(cfg):
        raise ValueError(f"Configuration mismatch: {folder}")
    report = json.loads((folder / "metrics.json").read_text(encoding="utf-8"))
    rows = report["by_maturity"]
    expected = {(t, q) for t in TABLE_MATURITIES for q in ("price", "delta")}
    if len(rows) != len(expected) or {(r["tau"], r["quantity"]) for r in rows} != expected:
        raise ValueError(f"Incomplete maturity metrics: {folder}")
    if report["evaluation"]["seed"] != cfg.seed or report["evaluation"]["sigma"] != cfg.sigma:
        raise ValueError(f"Evaluation metadata mismatch: {folder}")
    for row in rows:
        if row["count"] != cfg.test_samples:
            raise ValueError(f"Incorrect number of test contracts: {folder}")
        for key in ("RMAD", "RMSE_relative", "NMAD", "MSE"):
            if row[key] is None or not math.isfinite(row[key]):
                raise ValueError(f"Missing or non-finite {key}: {folder}")
    return report


def black_scholes(s: np.ndarray, k: np.ndarray, tau: np.ndarray,
                  sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """Только проверка: европейский call, r=0, tau>0."""
    d1 = (np.log(s / k) + 0.5 * sigma**2 * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return s * ndtr(d1) - k * ndtr(d2), ndtr(d1)


def predict(model: FINN, s: np.ndarray, k: np.ndarray, tau: np.ndarray,
            cfg: Config) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    prices, deltas = [], []
    with torch.enable_grad():
        for start in range(0, len(s), cfg.batch_size):
            values = (s[start:start + cfg.batch_size], k[start:start + cfg.batch_size],
                      tau[start:start + cfg.batch_size])
            st, kt, tt = [torch.as_tensor(v.reshape(-1, 1), dtype=torch.float32,
                                         device=cfg.device) for v in values]
            st = st.detach().requires_grad_(True)
            p = model.bounded_price(st, kt, tt)  # Цена и Delta после одной корректировки.
            delta = torch.autograd.grad(p.sum(), st)[0]
            prices.append(p.detach().cpu().numpy().ravel())
            deltas.append(delta.detach().cpu().numpy().ravel())
    return np.concatenate(prices), np.concatenate(deltas)


def test_contracts(cfg: Config) -> tuple[np.ndarray, np.ndarray, int]:
    """10 000 разных состояний из независимого GBM; S и K в [75,125]."""
    path_seed, sample_seed = np.random.SeedSequence([cfg.effective_data_seed, 2]).spawn(2)
    path_rng = np.random.default_rng(path_seed)
    rng = np.random.default_rng(sample_seed)
    pools, count, path_count = [], 0, 0
    # При нехватке подходящих состояний добавляем новые траектории, не дублируем точки.
    while count < cfg.test_samples:
        paths = simulate_gbm(cfg, cfg.test_paths, path_rng)
        states = paths[:, 1:].ravel()  # Начальное одинаковое S0 не включаем.
        eligible = states[(states >= cfg.test_spot_min) & (states <= cfg.test_spot_max)]
        pools.append(eligible)
        count += eligible.size
        path_count += cfg.test_paths
    states = np.concatenate(pools)
    s = states[rng.choice(states.size, cfg.test_samples, replace=False)]
    k = rng.uniform(cfg.test_strike_min, cfg.test_strike_max, cfg.test_samples)
    return s, k, path_count


def paper_metrics(pred: np.ndarray, target: np.ndarray, tolerance: float) -> dict:
    """Метрики F.1: RMAD (62), relative RMSE (63), NMAD (67), MSE (68)."""
    pred, target = np.asarray(pred, dtype=float), np.asarray(target, dtype=float)
    if pred.shape != target.shape or pred.size == 0:
        raise ValueError("Metrics require nonempty arrays of identical shape")
    if not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError("Non-finite prediction or reference")
    mask = np.abs(target) >= tolerance
    e = target - pred  # F.1: BSM - FINN; абсолютные/квадратичные метрики не меняются.
    result = {"count": int(target.size), "masked_count": int(mask.sum()),
              "mask_tolerance": tolerance, "MSE": float(np.mean(e**2))}
    if not mask.any():
        result.update(RMAD=None, RMSE_relative=None, NMAD=None)
    else:
        y, em = target[mask], e[mask]
        result.update(RMAD=float(np.abs(em).sum() / (np.abs(y).sum() + 1e-12)),
                      RMSE_relative=float(np.sqrt(np.sum(em**2) / (np.sum(y**2) + 1e-12))),
                      NMAD=float(np.abs(em).mean() / (np.ptp(y) + 1e-12)))
    return result


def prediction_grid(cfg: Config) -> tuple:
    ss, kk = np.meshgrid(np.linspace(80, 120, 41), np.linspace(90, 110, 21), indexing="ij")
    m = np.linspace(0.5, 2.0, 301)
    spots = np.concatenate((ss.ravel(), 100 * m))
    strikes = np.concatenate((kk.ravel(), np.full_like(m, 100)))
    times = np.full_like(spots, cfg.surface_tau)
    return ss, kk, spots, strikes, times


def draw_prediction_plots(directory: Path, cfg: Config, ss: np.ndarray, kk: np.ndarray,
                          spots: np.ndarray, predictions: np.ndarray, truth: np.ndarray,
                          std: np.ndarray | None = None, count: int = 1) -> None:
    """Одинаковые сетки и оформление для одной модели и среднего по seed."""
    import matplotlib.pyplot as plt
    label = "FINN" if count == 1 else f"FINN mean ({count} seeds)"
    errors = truth - predictions
    fig = plt.figure(figsize=(13, 10))
    for i, quantity in enumerate(("Price", "Delta")):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        for values, color, legend in ((predictions, "red", label), (truth, "green", "Black-Scholes")):
            ax.plot_wireframe(ss, kk, values[i, :ss.size].reshape(ss.shape), color=color,
                              rstride=1, cstride=1, linewidth=0.5, label=legend)
        ax.set(xlabel="S", ylabel="K", zlabel="C" if i == 0 else "Δ", title=quantity)
        ax.view_init(elev=30, azim=-135)
        ax.legend(loc="upper right", fontsize=8)
        ax = fig.add_subplot(2, 2, i + 3, projection="3d")
        ax.plot_surface(ss, kk, errors[i, :ss.size].reshape(ss.shape), cmap="coolwarm")
        ax.set(xlabel="S", ylabel="K", zlabel="BSM - FINN",
               title="Price error (BSM - FINN)" if i == 0 else "Delta error (BSM - FINN)")
        ax.view_init(elev=30, azim=-135)
    fig.suptitle(f"{label}; sigma={cfg.sigma}, tau={cfg.surface_tau}")
    fig.tight_layout()
    fig.savefig(directory / "surfaces.png", dpi=160)
    plt.close(fig)

    m = spots[ss.size:] / 100
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for i, quantity in enumerate(("Call price, K=100", "Delta")):
        estimate, reference = predictions[i, ss.size:], truth[i, ss.size:]
        axes[i, 0].plot(m, estimate, color="red", label=label)
        if std is not None:
            spread = std[i, ss.size:]
            axes[i, 0].fill_between(m, estimate - spread, estimate + spread, color="red", alpha=0.12,
                                    label="±1 std across seeds")
        axes[i, 0].plot(m, reference, "--", color="green", label="Black-Scholes")
        axes[i, 0].set_title(quantity)
        axes[i, 1].plot(m, reference - estimate, label="BSM - FINN")
        axes[i, 1].set(title="Price error" if i == 0 else "Delta error", ylabel="BSM - FINN")
    for ax in axes.flat:
        ax.set_xlabel("S/K")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle(f"sigma={cfg.sigma}, tau={cfg.surface_tau}")
    fig.tight_layout()
    fig.savefig(directory / "comparison.png", dpi=160)
    plt.close(fig)


def draw_loss_plot(directory: Path, epochs: np.ndarray, curves: dict, title: str) -> None:
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, values in curves.items():
        ax.semilogy(epochs, np.maximum(values, 1e-16), label=label)
    ax.set(title=title, xlabel="Epoch", ylabel="Mean squared monetary residual")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "training.png", dpi=160)
    plt.close(fig)


def plot_training_history(out: Path, cfg: Config) -> None:
    h = read_history(out / "history.csv")
    draw_loss_plot(out, h["epoch"], {split: h[f"{split}_loss"] for split in ("train", "val")},
                   "Hedging loss")


def plot_model_predictions(model: FINN, cfg: Config, directory: Path) -> None:
    ss, kk, spots, strikes, times = prediction_grid(cfg)
    predictions = np.stack(predict(model, spots, strikes, times, cfg))
    truth = np.stack(black_scholes(spots, strikes, times, cfg.sigma))
    draw_prediction_plots(directory, cfg, ss, kk, spots, predictions, truth)


def evaluate(model: FINN, cfg: Config, out: Path) -> dict:
    s, k, test_path_count = test_contracts(cfg)
    np.savetxt(out / "test_contracts.csv", np.column_stack((s, k)), delimiter=",",
               header="S,K", comments="", fmt="%.17g")
    metrics = {"evaluation": {"sigma": cfg.sigma, "seed": cfg.seed,
               "data_seed": cfg.effective_data_seed, "test_points": cfg.test_samples,
               "test_paths": test_path_count, "strike_range": [cfg.test_strike_min, cfg.test_strike_max],
               "spot_range": [cfg.test_spot_min, cfg.test_spot_max],
               "realized_spot_range": [float(s.min()), float(s.max())],
               "sampling": "S from independent GBM after t=0, filtered to [75,125]; K uniform [75,125]; same pairs at every maturity",
               "metric_source": "arXiv:2412.12213v2, Appendix F.1",
               "prediction_protocol": "Intrinsic floor at evaluation; delta of bounded price",
               "protocol_note": "GBM spot sampling and uniform K are explicit choices; the authors do not fully specify the joint test distribution."}}
    rows = []
    maturities = [t for t in TABLE_MATURITIES if t <= cfg.max_tau + 1e-9]
    for t in maturities:
        sv, kv = s, k
        times = np.full_like(sv, t)
        p, d = predict(model, sv, kv, times, cfg)
        ref, rd = black_scholes(sv, kv, times, cfg.sigma)
        for label, estimate, truth, tol in (("price", p, ref, cfg.price_mask),
                                            ("delta", d, rd, cfg.delta_mask)):
            rows.append({"sigma": cfg.sigma, "tau": t, "quantity": label,
                         **paper_metrics(estimate, truth, tol)})
    write_csv(out / "metrics_by_maturity.csv", rows)
    metrics["by_maturity"] = rows
    write_json(out / "metrics.json", metrics)
    if math.isclose(cfg.sigma, cfg.surface_sigma, rel_tol=0, abs_tol=1e-12):
        plot_model_predictions(model, cfg, out)
        plot_training_history(out, cfg)
    return metrics


def evaluate_job(config: dict, directory: str, worker_threads: int) -> dict:
    """Top-level picklable worker for parallel evaluation of trained checkpoints."""
    import matplotlib
    matplotlib.use("Agg")
    cfg = Config(**config)
    directory_path = Path(directory)
    folder = run_directory(directory_path, cfg)
    if completed_report(directory_path, cfg) is not None:
        return {"sigma": cfg.sigma, "seed": cfg.seed, "reused": True}
    summary = checked_training_summary(folder, cfg)
    if summary is None:
        raise FileNotFoundError(f"Обучение не завершено: {folder}")
    torch.set_num_threads(worker_threads)
    model = restore_model(load_checkpoint(folder / "best_model.pt", cfg.device, cfg), cfg.device)
    report = evaluate(model, cfg, folder)
    report["evaluated_checkpoint"] = {"epoch": summary["best_epoch"], "selection": "minimum_validation_loss"}
    write_json(folder / "metrics.json", report)
    write_json(folder / "complete.json", {"config": asdict(cfg)})
    del model
    return {"sigma": cfg.sigma, "seed": cfg.seed, "reused": False}


def evaluate_study(directory: Path, workers: int = 15, worker_threads: int = 1) -> dict:
    """Bounded queue of independent evaluation jobs across isolated spawn processes."""
    directory = Path(directory).resolve()
    configurations = read_training_plan(directory, workers, worker_threads)
    pending = []
    for index, cfg in enumerate(configurations, 1):
        folder = run_directory(directory, cfg)
        if completed_report(directory, cfg) is not None:
            print(f"[{index}/{len(configurations)}] sigma={cfg.sigma}, seed={cfg.seed}: сохранено", flush=True)
            continue
        if checked_training_summary(folder, cfg) is None:
            raise FileNotFoundError(f"Обучение не завершено: {folder}")
        pending.append(cfg)
    started = time.perf_counter()
    status = {"expected_runs": len(configurations), "skipped_runs": len(configurations) - len(pending),
              "evaluated_runs": 0, "workers": workers, "worker_threads": worker_threads,
              "status": "running", "elapsed_seconds": 0.0}
    status_path = directory / "evaluation_status.json"
    write_json(status_path, status)
    executor = None
    futures = []
    try:
        if workers == 1:
            for cfg in pending:
                result = evaluate_job(asdict(cfg), str(directory), worker_threads)
                status["evaluated_runs"] += 1
                print(f"Оценено {status['evaluated_runs']}/{len(pending)}: "
                      f"sigma={result['sigma']:.3f}, seed={result['seed']}", flush=True)
        elif pending:
            executor = ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                                           mp_context=mp.get_context("spawn"))
            futures = [executor.submit(evaluate_job, asdict(cfg), str(directory), worker_threads)
                       for cfg in pending]
            for future in as_completed(futures):
                result = future.result()
                status["evaluated_runs"] += 1
                print(f"Оценено {status['evaluated_runs']}/{len(pending)}: "
                      f"sigma={result['sigma']:.3f}, seed={result['seed']}", flush=True)
        status["status"] = "complete"
    except BaseException as error:
        for future in futures:
            future.cancel()
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        status["elapsed_seconds"] = time.perf_counter() - started
        write_json(status_path, status)
    return status


def _stop_process_group(process: subprocess.Popen) -> None:
    """Notebook interruption also stops CUDA workers of the launched script."""
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def launch_training(directory: Path, workers: int = 15, worker_threads: int = 1) -> dict:
    """Launch on the kernel's machine, using the kernel's Python environment."""
    directory = Path(directory).resolve()
    read_training_plan(directory, workers, worker_threads)
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--study", str(directory), "--workers", str(workers),
               "--worker-threads", str(worker_threads)]
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               env=environment, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        _stop_process_group(process)
        raise
    finally:
        process.stdout.close()
    status = read_json(directory / "training_status.json")
    if status.get("status") != "complete":
        raise RuntimeError("Training script did not finish successfully")
    return status


def launch_evaluation(directory: Path, workers: int = 15, worker_threads: int = 1) -> dict:
    """Launch parallel checkpoint evaluation on the kernel's machine."""
    directory = Path(directory).resolve()
    read_training_plan(directory, workers, worker_threads)
    command = [sys.executable, "-u", str(Path(__file__).resolve()),
               "--study", str(directory), "--workers", str(workers),
               "--worker-threads", str(worker_threads), "--evaluate"]
    environment = os.environ.copy()
    environment.update(PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", bufsize=1,
                               env=environment, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end="", flush=True)
        code = process.wait()
        if code:
            raise subprocess.CalledProcessError(code, command)
    except BaseException:
        _stop_process_group(process)
        raise
    finally:
        process.stdout.close()
    status = read_json(directory / "evaluation_status.json")
    if status.get("status") != "complete":
        raise RuntimeError("Evaluation script did not finish successfully")
    return status


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate FINN GBM seeds in isolated spawn processes")
    parser.add_argument("--study", type=Path, required=True, help="Existing directory with study.json")
    parser.add_argument("--workers", type=int, default=15)
    parser.add_argument("--worker-threads", type=int, default=1)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate trained checkpoints in parallel")
    args = parser.parse_args()
    if args.evaluate:
        status = evaluate_study(args.study, args.workers, args.worker_threads)
        print(f"Оценка завершена: {status['evaluated_runs']} новых, {status['skipped_runs']} готовых; "
              f"время оценки {status['elapsed_seconds']:.1f} с", flush=True)
    else:
        status = train_study(args.study, args.workers, args.worker_threads)
        print(f"Обучение завершено: {status['trained_runs']} новых, {status['skipped_runs']} готовых; "
              f"время серии {status['elapsed_seconds']:.1f} с", flush=True)


if __name__ == "__main__":
    main()
