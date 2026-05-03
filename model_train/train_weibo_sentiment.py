import argparse
import inspect
import io
import json
import os
import random
import shutil
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def guess_columns(column_names: List[str]) -> Tuple[str, str]:
    text_candidates = ["text", "sentence", "content", "review", "comment", "weibo"]
    label_candidates = ["label", "sentiment", "target", "y"]

    text_col = next((c for c in text_candidates if c in column_names), None)
    label_col = next((c for c in label_candidates if c in column_names), None)

    if not text_col:
        raise ValueError(f"Cannot find text column in dataset columns: {column_names}")
    if not label_col:
        raise ValueError(f"Cannot find label column in dataset columns: {column_names}")
    return text_col, label_col


def normalize_labels(labels: List) -> List[int]:
    out: List[int] = []
    for v in labels:
        if v is None:
            out.append(0)
            continue
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
            continue
        s = str(v).strip().lower()
        if s in {"1", "pos", "positive", "积极"}:
            out.append(1)
        elif s in {"0", "neg", "negative", "消极"}:
            out.append(0)
        else:
            try:
                out.append(int(float(s)))
            except Exception:
                out.append(0)
    uniq = sorted(set(out))
    if uniq == [1, 2]:
        out = [x - 1 for x in out]
    return out


def compute_metrics_builder():
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    except Exception as e:
        raise RuntimeError(
            f"Missing scikit-learn: {e}. Install: python3 -m pip install --user scikit-learn"
        )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        labels = labels.astype(int)
        preds = preds.astype(int)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
        }

    return compute_metrics


def _extract_series_by_epoch(log_history: List[Dict], key: str) -> Dict[float, float]:
    out: Dict[float, float] = {}
    for it in log_history or []:
        if not isinstance(it, dict):
            continue
        if "epoch" not in it:
            continue
        if key not in it:
            continue
        try:
            ep = float(it["epoch"])
            val = float(it[key])
        except Exception:
            continue
        out[ep] = val
    return out


def _aggregate_epoch_stats(per_fold: List[Dict[float, float]]) -> Dict[float, Dict[str, float]]:
    epochs = sorted({ep for d in per_fold for ep in (d or {}).keys()})
    out: Dict[float, Dict[str, float]] = {}
    for ep in epochs:
        vals = []
        for d in per_fold:
            if d is None:
                continue
            v = d.get(ep)
            if v is None:
                continue
            vals.append(float(v))
        if not vals:
            continue
        arr = np.array(vals, dtype=float)
        out[ep] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out


def _plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: List[str], title: str, save_path: Optional[str], show: bool):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Missing matplotlib: {e}. Install: python3 -m pip install --user matplotlib") from e

    try:
        from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    except Exception as e:
        raise RuntimeError(
            f"Missing scikit-learn: {e}. Install: python3 -m pip install --user scikit-learn"
        ) from e

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


def _plot_training_curves(
    train_loss_stats: Dict[float, Dict[str, float]],
    eval_loss_stats: Dict[float, Dict[str, float]],
    eval_f1_stats: Dict[float, Dict[str, float]],
    title: str,
    save_path: Optional[str],
    show: bool,
):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError(f"Missing matplotlib: {e}. Install: python3 -m pip install --user matplotlib") from e

    epochs = sorted(set(train_loss_stats.keys()) | set(eval_loss_stats.keys()) | set(eval_f1_stats.keys()))
    if not epochs:
        return

    def series(stats: Dict[float, Dict[str, float]]):
        xs, ys, yerr = [], [], []
        for ep in epochs:
            if ep not in stats:
                continue
            xs.append(ep)
            ys.append(stats[ep]["mean"])
            yerr.append(stats[ep]["std"])
        return xs, ys, yerr

    x_t, y_t, e_t = series(train_loss_stats)
    x_e, y_e, e_e = series(eval_loss_stats)
    x_f, y_f, e_f = series(eval_f1_stats)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.2))
    if x_t:
        ax1.plot(x_t, y_t, label="train_loss", linewidth=2)
        ax1.fill_between(x_t, np.array(y_t) - np.array(e_t), np.array(y_t) + np.array(e_t), alpha=0.15)
    if x_e:
        ax1.plot(x_e, y_e, label="eval_loss", linewidth=2)
        ax1.fill_between(x_e, np.array(y_e) - np.array(e_e), np.array(y_e) + np.array(e_e), alpha=0.15)
    ax1.set_title("Loss Curve (Mean±Std)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.25)
    ax1.legend()

    if x_f:
        ax2.plot(x_f, y_f, label="eval_macro_f1", linewidth=2, color="#ef4444")
        ax2.fill_between(x_f, np.array(y_f) - np.array(e_f), np.array(y_f) + np.array(e_f), alpha=0.15, color="#ef4444")
    ax2.set_title("Macro-F1 Curve (Mean±Std)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Macro-F1")
    ax2.grid(True, alpha=0.25)
    ax2.legend()

    fig.suptitle(title)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    if show:
        plt.show()
    plt.close(fig)


@dataclass(frozen=True)
class TrainConfig:
    dataset_name: str
    model_name: str
    output_dir: str
    num_folds: int
    max_length: int
    learning_rate: float
    num_train_epochs: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    seed: int
    fp16: bool
    bf16: bool
    save_total_limit: int
    eval_strategy: str
    logging_steps: int


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="asap")
    parser.add_argument("--model", default="IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment")
    parser.add_argument("--out_dir", default=os.path.join(os.getcwd(), "model_train_outputs", "restaurant_sentiment_kfold"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--train_bs", type=int, default=32)
    parser.add_argument("--eval_bs", type=int, default=64)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=200)
    parser.add_argument("--hf_endpoint", default="https://hf-mirror.com")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--show_plots", action="store_true")
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--use_official_test", action="store_true")
    parser.add_argument("--eval_strategy", default="no", choices=["no", "epoch"])
    parser.add_argument("--dataloader_num_workers", type=int, default=2)
    parser.add_argument("--save_all_folds", action="store_true")
    parser.add_argument("--train_final_model", action="store_true")
    parser.add_argument("--tune_mode", default="head", choices=["head", "head+last_n", "full"])
    parser.add_argument("--unfreeze_last_n_layers", type=int, default=0)
    parser.add_argument("--asap_sample", action="store_true")
    args = parser.parse_args()

    if args.hf_endpoint and not os.environ.get("HF_ENDPOINT"):
        os.environ["HF_ENDPOINT"] = str(args.hf_endpoint)

    cfg = TrainConfig(
        dataset_name=str(args.dataset),
        model_name=str(args.model),
        output_dir=str(args.out_dir),
        num_folds=int(args.folds),
        max_length=int(args.max_length),
        learning_rate=float(args.lr),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.train_bs),
        per_device_eval_batch_size=int(args.eval_bs),
        gradient_accumulation_steps=int(args.grad_accum),
        warmup_ratio=float(args.warmup_ratio),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
        fp16=bool(args.fp16),
        bf16=bool(args.bf16),
        save_total_limit=int(args.save_total_limit),
        eval_strategy=str(args.eval_strategy),
        logging_steps=int(args.logging_steps),
    )

    try:
        from datasets import Dataset, concatenate_datasets, load_dataset
    except Exception as e:
        raise RuntimeError(f"Missing datasets: {e}. Install: python3 -m pip install --user datasets") from e

    try:
        from sklearn.model_selection import StratifiedKFold
    except Exception as e:
        raise RuntimeError(
            f"Missing scikit-learn: {e}. Install: python3 -m pip install --user scikit-learn"
        ) from e

    try:
        import torch
    except Exception as e:
        raise RuntimeError(f"Missing torch: {e}. Install: python3 -m pip install --user torch") from e

    try:
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:
        raise RuntimeError(
            f"Missing transformers: {e}. Install: python3 -m pip install --user transformers"
        ) from e

    set_seed(cfg.seed)

    def _asap_cache_dir() -> str:
        if args.cache_dir:
            base = os.path.join(str(args.cache_dir), "asap")
        else:
            base = os.path.join(cfg.output_dir, "_dataset_cache", "asap")
        os.makedirs(base, exist_ok=True)
        return base

    def _download_asap_if_needed(dst_dir: str) -> str:
        url = "https://github.com/Meituan-Dianping/asap/archive/refs/heads/master.zip"
        zip_path = os.path.join(dst_dir, "asap_master.zip")
        extract_dir = os.path.join(dst_dir, "asap-master")
        if os.path.isdir(extract_dir) and any(
            os.path.exists(os.path.join(extract_dir, "data", fn)) for fn in ["train.csv", "dev.csv", "test.csv"]
        ):
            return extract_dir
        if not os.path.exists(zip_path):
            last_err = None
            for _ in range(3):
                try:
                    with urllib.request.urlopen(url, timeout=90) as r:
                        data = r.read()
                    with open(zip_path, "wb") as f:
                        f.write(data)
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    try:
                        if os.path.exists(zip_path):
                            os.remove(zip_path)
                    except Exception:
                        pass
            if last_err is not None:
                raise last_err
        tmp_dir = os.path.join(dst_dir, "_extract_tmp")
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        os.makedirs(tmp_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(tmp_dir)
        src = os.path.join(tmp_dir, "asap-master")
        if not os.path.isdir(src):
            raise RuntimeError("ASAP zip structure unexpected (missing asap-master/)")
        if os.path.exists(extract_dir):
            shutil.rmtree(extract_dir)
        shutil.move(src, extract_dir)
        shutil.rmtree(tmp_dir)
        return extract_dir

    def _load_asap_splits(sample: bool) -> Dict[str, "Dataset"]:
        cache_dir = _asap_cache_dir()
        root = _download_asap_if_needed(cache_dir)
        data_dir = os.path.join(root, "data")
        train_name = "train_sample.csv" if sample else "train.csv"
        dev_name = "dev_sample.csv" if sample else "dev.csv"
        test_name = "test_sample.csv" if sample else "test.csv"

        import pandas as pd

        def read_csv(name: str) -> pd.DataFrame:
            path = os.path.join(data_dir, name)
            if not os.path.exists(path):
                raise RuntimeError(f"Missing ASAP file: {path}")
            return pd.read_csv(path)

        def build_ds(df: pd.DataFrame) -> "Dataset":
            text_col = "review" if "review" in df.columns else ("reviewbody" if "reviewbody" in df.columns else None)
            if text_col is None or "star" not in df.columns:
                raise RuntimeError(f"Unexpected ASAP columns: {list(df.columns)}")
            df = df[[text_col, "star"]].copy()
            df[text_col] = df[text_col].fillna("").astype(str)
            df = df[df[text_col].str.strip() != ""]
            star = pd.to_numeric(df["star"], errors="coerce")
            pos_mask = star >= 4
            neg_mask = star <= 2
            keep = pos_mask | neg_mask
            df = df[keep].copy()
            df["labels"] = (star[keep] >= 4).astype(int).tolist()
            df = df.drop(columns=["star"]).rename(columns={text_col: "review"})
            return Dataset.from_pandas(df, preserve_index=False)

        return {
            "train": build_ds(read_csv(train_name)),
            "validation": build_ds(read_csv(dev_name)),
            "test": build_ds(read_csv(test_name)),
        }

    def _standardize_dataset_and_split() -> Tuple["Dataset", "Dataset", str]:
        ds_dict: Dict[str, "Dataset"] = {}
        text_col = "text"

        if str(cfg.dataset_name).strip().lower() == "asap":
            ds_dict = _load_asap_splits(sample=bool(args.asap_sample))
            text_col = "review"
        else:
            raw = load_dataset(cfg.dataset_name, cache_dir=args.cache_dir)
            if isinstance(raw, dict):
                for k in ["train", "validation", "test"]:
                    if k in raw:
                        ds_dict[k] = raw[k]
                if not ds_dict:
                    if len(raw) != 1:
                        raise RuntimeError(f"Unexpected dataset splits: {list(raw.keys())}")
                    ds_dict["train"] = next(iter(raw.values()))
            else:
                ds_dict["train"] = raw

            inferred_text_col, label_col = guess_columns(list(ds_dict["train"].features.keys()))
            text_col = inferred_text_col
            for split_name, ds in list(ds_dict.items()):
                ds = ds.filter(lambda x: x.get(text_col) is not None and str(x.get(text_col)).strip() != "")
                labels = normalize_labels(ds[label_col])
                ds = ds.remove_columns([label_col]).add_column("labels", labels)
                ds_dict[split_name] = ds

        use_official_test = bool(args.use_official_test) or str(cfg.dataset_name).strip().lower() == "asap"
        if use_official_test and "test" in ds_dict and "train" in ds_dict:
            train_splits = [ds_dict["train"]]
            if "validation" in ds_dict:
                train_splits.append(ds_dict["validation"])
            train_val = concatenate_datasets(train_splits)
            test = ds_dict["test"]
            return train_val, test, text_col

        ds_all = concatenate_datasets(list(ds_dict.values()))
        try:
            from sklearn.model_selection import train_test_split
        except Exception as e:
            raise RuntimeError(
                f"Missing scikit-learn: {e}. Install: python3 -m pip install --user scikit-learn"
            ) from e
        y_all = np.array(ds_all["labels"], dtype=int)
        idx = np.arange(len(ds_all))
        train_idx, test_idx = train_test_split(
            idx,
            test_size=float(args.test_size),
            random_state=cfg.seed,
            stratify=y_all,
        )
        train_val = Dataset.from_dict(ds_all.select(train_idx).to_dict())
        test = Dataset.from_dict(ds_all.select(test_idx).to_dict())
        return train_val, test, text_col

    train_val_ds, test_ds, text_col = _standardize_dataset_and_split()

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, cache_dir=args.cache_dir, local_files_only=bool(args.local_files_only))
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    def tokenize_fn(batch):
        return tokenizer(
            batch[text_col],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_val_ds = train_val_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])
    test_ds = test_ds.map(tokenize_fn, batched=True, remove_columns=[text_col])

    y = np.array(train_val_ds["labels"], dtype=int)
    skf = StratifiedKFold(n_splits=cfg.num_folds, shuffle=True, random_state=cfg.seed)

    results: List[Dict] = []
    os.makedirs(cfg.output_dir, exist_ok=True)

    compute_metrics = compute_metrics_builder()

    use_mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

    def _apply_partial_finetune_policy(m):
        mode = str(args.tune_mode).strip().lower()
        if mode == "full":
            for p in m.parameters():
                p.requires_grad = True
            return

        for p in m.parameters():
            p.requires_grad = False

        head_keys = ("classifier", "score", "pre_classifier", "classification_head")
        for name, p in m.named_parameters():
            if any(k in name for k in head_keys):
                p.requires_grad = True

        if mode == "head+last_n":
            n = int(args.unfreeze_last_n_layers or 0)
            if n <= 0:
                return
            base = None
            for attr in ["bert", "roberta", "deberta", "electra", "albert", "xlnet"]:
                if hasattr(m, attr):
                    base = getattr(m, attr)
                    break
            if base is None or not hasattr(base, "encoder") or not hasattr(base.encoder, "layer"):
                return
            layers = base.encoder.layer
            try:
                last_layers = list(layers)[-n:]
            except Exception:
                return
            for layer in last_layers:
                for p in layer.parameters():
                    p.requires_grad = True

    all_val_true: List[int] = []
    all_val_pred: List[int] = []
    per_fold_train_loss: List[Dict[float, float]] = []
    per_fold_eval_loss: List[Dict[float, float]] = []
    per_fold_eval_f1: List[Dict[float, float]] = []
    best_fold_score: Optional[float] = None
    best_fold_dir = os.path.join(cfg.output_dir, "best_fold_model")
    if os.path.exists(best_fold_dir):
        shutil.rmtree(best_fold_dir)

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        fold_dir = os.path.join(cfg.output_dir, f"fold_{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        train_ds = Dataset.from_dict(train_val_ds.select(train_idx).to_dict())
        val_ds = Dataset.from_dict(train_val_ds.select(val_idx).to_dict())

        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=2,
            cache_dir=args.cache_dir,
            local_files_only=bool(args.local_files_only),
        )
        _apply_partial_finetune_policy(model)

        ta_kwargs = {
            "output_dir": fold_dir,
            "learning_rate": cfg.learning_rate,
            "num_train_epochs": cfg.num_train_epochs,
            "per_device_train_batch_size": cfg.per_device_train_batch_size,
            "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
            "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "save_strategy": "no",
            "save_total_limit": cfg.save_total_limit,
            "logging_steps": cfg.logging_steps,
            "seed": cfg.seed,
            "fp16": cfg.fp16 and torch.cuda.is_available(),
            "bf16": cfg.bf16,
            "report_to": [],
        }
        ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
        if "dataloader_num_workers" in ta_params:
            ta_kwargs["dataloader_num_workers"] = int(args.dataloader_num_workers)
        if "dataloader_pin_memory" in ta_params:
            ta_kwargs["dataloader_pin_memory"] = False if use_mps else True
        if "evaluation_strategy" in ta_params:
            ta_kwargs["evaluation_strategy"] = cfg.eval_strategy
        elif "eval_strategy" in ta_params:
            ta_kwargs["eval_strategy"] = cfg.eval_strategy
        else:
            raise RuntimeError("TrainingArguments does not support evaluation strategy settings in this transformers version.")

        training_args = TrainingArguments(**ta_kwargs)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        metrics = trainer.evaluate()
        metrics = {k: float(v) for k, v in (metrics or {}).items() if isinstance(v, (int, float, np.number))}
        metrics["fold"] = int(fold)
        results.append(metrics)

        pred_out = trainer.predict(val_ds)
        logits = pred_out.predictions
        y_true = pred_out.label_ids
        y_pred = np.argmax(logits, axis=-1)
        all_val_true.extend([int(x) for x in y_true])
        all_val_pred.extend([int(x) for x in y_pred])

        log_history = list(getattr(trainer.state, "log_history", []) or [])
        train_loss_by_epoch = _extract_series_by_epoch(log_history, "loss")
        eval_loss_by_epoch = _extract_series_by_epoch(log_history, "eval_loss")
        eval_f1_by_epoch = _extract_series_by_epoch(log_history, "eval_macro_f1")
        per_fold_train_loss.append(train_loss_by_epoch)
        per_fold_eval_loss.append(eval_loss_by_epoch)
        per_fold_eval_f1.append(eval_f1_by_epoch)

        try:
            cur = metrics.get("eval_macro_f1")
            cur_f = float(cur) if cur is not None else None
        except Exception:
            cur_f = None
        if cur_f is not None and (best_fold_score is None or cur_f > best_fold_score):
            best_fold_score = cur_f
            if os.path.exists(best_fold_dir):
                shutil.rmtree(best_fold_dir)
            trainer.save_model(best_fold_dir)
            tokenizer.save_pretrained(best_fold_dir)
        if bool(args.save_all_folds):
            trainer.save_model(fold_dir)
            tokenizer.save_pretrained(fold_dir)

    def summarize(key: str) -> Dict[str, float]:
        vals = [r.get(key) for r in results if r.get(key) is not None]
        if not vals:
            return {"mean": 0.0, "std": 0.0}
        arr = np.array(vals, dtype=float)
        return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0}

    summary = {
        "dataset": cfg.dataset_name,
        "model": cfg.model_name,
        "tune_mode": str(args.tune_mode),
        "unfreeze_last_n_layers": int(args.unfreeze_last_n_layers or 0),
        "use_official_test": bool(args.use_official_test) or str(cfg.dataset_name).strip().lower() == "asap",
        "test_size": float(args.test_size),
        "best_fold_macro_f1": float(best_fold_score) if best_fold_score is not None else None,
        "num_folds": cfg.num_folds,
        "max_length": cfg.max_length,
        "learning_rate": cfg.learning_rate,
        "epochs": cfg.num_train_epochs,
        "train_batch_size": cfg.per_device_train_batch_size,
        "eval_batch_size": cfg.per_device_eval_batch_size,
        "grad_accum": cfg.gradient_accumulation_steps,
        "warmup_ratio": cfg.warmup_ratio,
        "weight_decay": cfg.weight_decay,
        "seed": cfg.seed,
        "accuracy": summarize("eval_accuracy"),
        "macro_f1": summarize("eval_macro_f1"),
        "macro_precision": summarize("eval_macro_precision"),
        "macro_recall": summarize("eval_macro_recall"),
        "fold_metrics": results,
    }

    if os.path.exists(best_fold_dir):
        try:
            best_model = AutoModelForSequenceClassification.from_pretrained(best_fold_dir)
            ta_kwargs_eval = {
                "output_dir": os.path.join(cfg.output_dir, "_best_fold_eval"),
                "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
                "seed": cfg.seed,
                "fp16": cfg.fp16 and torch.cuda.is_available(),
                "bf16": cfg.bf16,
                "report_to": [],
            }
            ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
            if "dataloader_num_workers" in ta_params:
                ta_kwargs_eval["dataloader_num_workers"] = int(args.dataloader_num_workers)
            if "dataloader_pin_memory" in ta_params:
                ta_kwargs_eval["dataloader_pin_memory"] = False if use_mps else True
            trainer_best = Trainer(
                model=best_model,
                args=TrainingArguments(**ta_kwargs_eval),
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            holdout = trainer_best.evaluate()
            summary["holdout_test"] = {
                k: float(v) for k, v in (holdout or {}).items() if isinstance(v, (int, float, np.number))
            }
        except Exception as e:
            summary["holdout_test_error"] = str(e)

    if bool(args.train_final_model):
        try:
            final_dir = os.path.join(cfg.output_dir, "final_model")
            os.makedirs(final_dir, exist_ok=True)
            final_model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model_name,
                num_labels=2,
                cache_dir=args.cache_dir,
                local_files_only=bool(args.local_files_only),
            )
            _apply_partial_finetune_policy(final_model)

            ta_kwargs_final = {
                "output_dir": final_dir,
                "learning_rate": cfg.learning_rate,
                "num_train_epochs": cfg.num_train_epochs,
                "per_device_train_batch_size": cfg.per_device_train_batch_size,
                "per_device_eval_batch_size": cfg.per_device_eval_batch_size,
                "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
                "warmup_ratio": cfg.warmup_ratio,
                "weight_decay": cfg.weight_decay,
                "save_strategy": "no",
                "logging_steps": cfg.logging_steps,
                "seed": cfg.seed,
                "fp16": cfg.fp16 and torch.cuda.is_available(),
                "bf16": cfg.bf16,
                "report_to": [],
            }
            ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
            if "dataloader_num_workers" in ta_params:
                ta_kwargs_final["dataloader_num_workers"] = int(args.dataloader_num_workers)
            if "dataloader_pin_memory" in ta_params:
                ta_kwargs_final["dataloader_pin_memory"] = False if use_mps else True
            if "evaluation_strategy" in ta_params:
                ta_kwargs_final["evaluation_strategy"] = cfg.eval_strategy
            elif "eval_strategy" in ta_params:
                ta_kwargs_final["eval_strategy"] = cfg.eval_strategy
            else:
                raise RuntimeError(
                    "TrainingArguments does not support evaluation strategy settings in this transformers version."
                )

            trainer_final = Trainer(
                model=final_model,
                args=TrainingArguments(**ta_kwargs_final),
                train_dataset=train_val_ds,
                eval_dataset=test_ds,
                tokenizer=tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
            trainer_final.train()
            test_metrics = trainer_final.evaluate()
            test_metrics = {
                k: float(v) for k, v in (test_metrics or {}).items() if isinstance(v, (int, float, np.number))
            }
            summary["holdout_test"] = test_metrics
            trainer_final.save_model(final_dir)
            tokenizer.save_pretrained(final_dir)
        except Exception as e:
            summary["holdout_test_error"] = str(e)

    out_path = os.path.join(cfg.output_dir, "cv_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(out_path)
    print(json.dumps(summary["macro_f1"], ensure_ascii=False))

    y_true_all = np.array(all_val_true, dtype=int)
    y_pred_all = np.array(all_val_pred, dtype=int)
    if len(y_true_all) == len(y_pred_all) and len(y_true_all) > 0:
        labels_en = ["Negative", "Positive"]
        cm_path = os.path.join(cfg.output_dir, "confusion_matrix.png") if bool(args.save_plots) else None
        _plot_confusion_matrix(
            y_true=y_true_all,
            y_pred=y_pred_all,
            labels=labels_en,
            title="Confusion Matrix (All Folds Combined)",
            save_path=cm_path,
            show=bool(args.show_plots),
        )

        train_stats = _aggregate_epoch_stats(per_fold_train_loss)
        eval_loss_stats = _aggregate_epoch_stats(per_fold_eval_loss)
        eval_f1_stats = _aggregate_epoch_stats(per_fold_eval_f1)
        curve_path = os.path.join(cfg.output_dir, "training_curves.png") if bool(args.save_plots) else None
        _plot_training_curves(
            train_loss_stats=train_stats,
            eval_loss_stats=eval_loss_stats,
            eval_f1_stats=eval_f1_stats,
            title="Training Curves (5-Fold CV)",
            save_path=curve_path,
            show=bool(args.show_plots),
        )


if __name__ == "__main__":
    main()
