import torch
from dataset import get_train_dataset, get_test_dataset
from torch.utils.data import DataLoader
from model import RotationClassfier
import torch.nn as nn
import gc
import wandb
from pathlib import Path

torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True

config={
    "lr_classifier": 1e-3,
    "lr_backbone": 1e-4,
    "dataset_iterations": 4, # on average, each picture is shown in each orientation per epoch
    "scheduler": {
        "type": "cosine_warm_restarts",
        "t0": 2, # 2 epochs till first restart
        "t_mult": 2, 
        "eta_min": 1e-5,
    },
    "train_class_weights": { # train on equal probability classes
        0: 0.25,
        90: 0.25,
        180: 0.25,
        270: 0.25,
    },
    "test_class_weights": { # Unbalanced on purpose to see how it biases 
        0: 0.97,
        90: 0.01,
        180: 0.01,
        270: 0.01,
    },
    "batch_size": 650,
    "datasets": ["flickr30k", "places365"],
    "epochs": 20,
    "amp": True
    # "checkpoint_file": "/Users/tim/projects/rotation_classifier/runs/dry-dust-39/46/ckpt.pt"
}

train, test = get_train_dataset(oversample=config['dataset_iterations'], rotation_sample_weights=config['train_class_weights']), get_test_dataset(oversample=config['dataset_iterations'], rotation_sample_weights=config['test_class_weights'])
train_loader = DataLoader(train, batch_size=config["batch_size"], num_workers=16, shuffle=True)
test_loader = DataLoader(test, batch_size=config["batch_size"], num_workers=16, shuffle=True)
                         
# Allows continuously sampling the test set during continuous evals.
def cyclic_generator(loader: DataLoader):
    iterator = iter(loader)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            yield next(iterator)

periodic_eval_loader = cyclic_generator(test_loader)
if config.get("checkpoint_file", None) is not None:
    model = torch.load(Path(config["checkpoint_file"]), weights_only=False)
else:
    model = RotationClassfier()
model.to("cuda")
print("Compiling...")
model.compile()
print("Compiled!")
optimizer = torch.optim.AdamW([
    {"params": model.mobilenet_v2.parameters(), "lr": config['lr_backbone']}, 
    {"params": model.classifier.parameters(), "lr": config['lr_classifier']},
])
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['scheduler']['t0'], T_mult=config['scheduler']['t_mult'], eta_min=config['scheduler']['eta_min'])
loss_fn = torch.nn.CrossEntropyLoss()
scaler = torch.amp.GradScaler("cuda", enabled=config['amp'])

def log_grad_metrics(model: RotationClassfier, run):
    # We log the following metrics for the gradients
    # L2 norm
    # L1 norm
    # Mean
    # Variance
    # % zeroes
    # for the following splits:
    # - Overall
    # - Backbone
    # - Classifier
    training = model.training
    model.train(False)
    def log_stats_for_module(module: nn.Module, prefix: str):
        params = module.parameters()
        grads = [p.grad for p in params if p.grad is not None and p.requires_grad]
        if not grads:
            return
        grads = torch.cat([g.flatten() for g in grads])
        run.log({
            f"grad/{prefix}/l2_norm": grads.norm(p=2),
            f"grad/{prefix}/l1_norm": grads.norm(p=1),
            f"grad/{prefix}/mean": grads.mean(),
            f"grad/{prefix}/variance": grads.var(),
            f"grad/{prefix}/pct_zeroes": (grads == 0).float().mean(),
        })
    log_stats_for_module(model, "overall"),
    log_stats_for_module(model.mobilenet_v2, "backbone"),
    log_stats_for_module(model.classifier, "classifier")
    model.train(training)

def log_weight_metrics(model: RotationClassfier, run):
    # Log the following metrics for the weights:
    # L2 norm
    # L1 norm
    # Mean
    # Variance
    # for the following splits:
    # - Overall
    # - Backbone
    # - Classifier
    training = model.training
    model.train(False)
    def log_stats_for_module(module: nn.Module, prefix: str):
        params = [p for p in model.parameters() if p.requires_grad]
        params = torch.cat([p.flatten() for p in params])
        run.log({
            f"weights/{prefix}/l2_norm": params.norm(p=2),
            f"weights/{prefix}/l1_norm": params.norm(p=1),
            f"weights/{prefix}/mean": params.mean(),
            f"weights/{prefix}/variance": params.var(),
        })
    log_stats_for_module(model, "overall"),
    log_stats_for_module(model.mobilenet_v2, "backbone"),
    log_stats_for_module(model.classifier, "classifier"),
    model.train(training)

def train_epoch(epoch_idx: int, run):
    model.train(True)
    print(f"Starting train of epoch {epoch_idx}")
    for idx, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, y = data['image'], data['rotation_label']
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=config['amp']):
            x = x.to("cuda", non_blocking=True)
            y = y.to("cuda", non_blocking=True)
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        preds = y_hat.argmax(axis=1)
        correct = (y == preds).float()
        accuracy = correct.mean()
        print(f"Minibatch {idx}: loss {float(loss)}, accuracy: {accuracy}")
        run.log({
            "train/loss": float(loss), 
            "train/accuracy": accuracy, 
            "lr/backbone": optimizer.param_groups[0]['lr'],
            "lr/classifier": optimizer.param_groups[1]['lr'],
            "epoch": epoch_idx + idx / len(train_loader)
            })
        optimizer.step()
        scheduler.step(epoch_idx + idx / len(train_loader)) # fractional epoch
        if idx % 10 == 0:
            log_grad_metrics(model, run)
            log_weight_metrics(model, run)
            test(run)
        if idx % 2500 == 0:
            torch.save(model, Path("runs") / run.name / str(epoch_idx) / f"{idx}.pt")
    validate(run)

def test(run):
    training = model.training
    model.train(False)
    with torch.no_grad():
        data = next(periodic_eval_loader)
        x, y = data['image'], data['rotation_label']
        x = x.to("cuda")
        y = y.to("cuda")
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        preds = y_hat.argmax(axis=1)
        correct = (y == preds).float()
        accuracy = correct.mean()
        print(f"Eval minibatch loss: {float(loss)}, accuracy: {accuracy}")
        run.log({"test/loss": loss, "test/accuracy": accuracy})
    model.train(training)


def validate(run):
    training = model.training
    model.train(False)
    total_loss = 0
    total_accuracy = 0
    minibatches = 0
    with torch.no_grad():
        for idx, data in enumerate(test_loader):
            x, y = data['image'], data['rotation_label']
            x = x.to("cuda")
            y = y.to("cuda")
            y_hat = model(x)
            loss = loss_fn(y_hat, y)
            total_loss += float(loss)
            preds = y_hat.argmax(axis=1)
            correct = (y == preds).float()
            accuracy = correct.mean()
            total_accuracy += accuracy
            minibatches += 1
            print(f"Eval minibatch {idx}: loss {float(loss)}, accuracy: {accuracy}")
    print(f"Overall eval loss: {total_loss/minibatches}, accuracy: {total_accuracy/minibatches}")
    run.log({"val/loss": total_loss/minibatches, "val/accuracy": total_accuracy/minibatches})
    model.train(training)

if __name__ == "__main__":
    run = wandb.init(
       entity="kanarsky-projects",
        project="rotation-classifier",
       config=config
    )
    for i in range(config["epochs"]):
        save_dir = Path("/workspace/rotation_classifier/runs") / run.name / str(i)
        save_dir.mkdir(exist_ok=True, parents=True)
        train_epoch(i, run)
        torch.save(model, Path("runs") / run.name / str(i) / "final.pt")
