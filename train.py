import torch
from dataset import get_train_dataset, get_test_dataset
from torch.utils.data import DataLoader
from model import RotationClassfier
import torch.nn as nn
import wandb
from pathlib import Path
config={
    "lr_classifier": 1e-2,
    "lr_backbone": 1e-4,
    "scheduler": {
        "type": "cosine_warm_restarts",
        "t0": 2,
        "t_mult": 2,
        "eta_min": 1e-5,
    },
    "batch_size": 256,
    "datasets": ["flickr30k"],
    "epochs": 20,
}

train, test = get_train_dataset(oversample=10), get_test_dataset(oversample=10)
train_loader = DataLoader(train, batch_size=config["batch_size"], num_workers=4, shuffle=True)
test_loader = DataLoader(test, batch_size=config["batch_size"], num_workers=4, shuffle=True)
                         
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


model = RotationClassfier()
model.to("mps")
optimizer = torch.optim.AdamW(model.parameters())
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['scheduler']['t0'], T_mult=config['scheduler']['t_mult'], eta_min=config['scheduler']['eta_min'])
loss_fn = torch.nn.CrossEntropyLoss()


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
        pass
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
    pass

def train_epoch(epoch_idx: int, run):
    model.train(True)
    print(f"Starting train of epoch {epoch_idx}")
    for idx, data in enumerate(train_loader):
        x, y = data['image'], data['rotation_label']
        x = x.to("mps")
        y = y.to("mps")
        optimizer.zero_grad()
        y_hat = model(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        preds = y_hat.argmax(axis=1)
        correct = (y == preds).float()
        accuracy = correct.mean()
        print(f"Minibatch {idx}: loss {float(loss)}, accuracy: {accuracy}")
        run.log({"train/loss": loss, "train/accuracy": accuracy, "lr": optimizer.param_groups[0]['lr']})
        optimizer.step()
        scheduler.step(epoch_idx + idx / len(train_loader)) # fractional epoch
        if idx % 10 == 0:
            test(run)
    validate(run)

def test(run):
    training = model.training
    model.train(False)
    with torch.no_grad():
        data = next(periodic_eval_loader)
        x, y = data['image'], data['rotation_label']
        x = x.to("mps")
        y = y.to("mps")
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
            x = x.to("mps")
            y = y.to("mps")
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
        train_epoch(i, run)
        save_dir = Path("/Users/tim/projects/rotation_classifier/runs") / run.name / str(i)
        save_dir.mkdir(exist_ok=True, parents=True)
        torch.save(model, Path("runs") / run.name / str(i) / "ckpt.pt")
