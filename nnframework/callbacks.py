import numpy as np
import time
import csv
import os


class Callbacks:
    def on_train_begin(self, model):
        pass

    def on_epoch_begin(self, model ,epoch):
        pass

    def on_batch_begin(self, model, batch):
        pass

    def on_batch_end(self, model, batch, logs):
        pass

    def on_epoch_end(self, model, epoch, logs):
        pass

    def on_train_end(self, model):
        pass


class EarlyStopping(Callbacks):
    def __init__(self, monitor="val_loss", patience=5, min_delta=0, mode="min"):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode

    def on_train_begin(self, model):
        self.model = model
        self.wait = 0

        if self.mode == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")

    def on_epoch_end(self, model, epoch, logs):
        val_loss = logs[self.monitor]

        if self.mode == "min":
            improved = val_loss < self.best - self.min_delta
        else:
            improved = val_loss > self.best + self.min_delta
        
        if improved:
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
        
        if self.wait >= self.patience:
            self.model.stop_training = True


class ModelCheckpoint(Callbacks):
    def __init__(self, filename, monitor="val_loss", mode="min"):
        self.filename = filename
        self.monitor = monitor
        self.mode = mode

    def on_train_begin(self, model):
        self.model = model

        if self.mode == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")

    def on_epoch_end(self, model, epoch, logs):
        val_loss = logs[self.monitor]

        if self.mode == "min":
            improved = val_loss < self.best
        else:
            improved = val_loss > self.best
        
        if improved:
            print("Saved Current Model!")
            self.model.save(self.filename)
            self.best = val_loss


class BackUpCheckpoint(Callbacks):
    def __init__(self, filename, every=10):
        self.filename = filename
        self.every = every

    def on_epoch_end(self, model, epoch, logs):
        epoch = epoch + 1

        if epoch % self.every == 0:
            name = self.filename[:len(self.filename) - 4] + f"_{epoch}_checkpoint.npz"
            model.save(name)
            print("Backup Saved!")


class CSVLogger(Callbacks):
    def __init__(self, filename, append=False):
        self.filename = filename
        self.append = append
        self.file = None
        self.writer = None
    
    def on_train_begin(self, model):
        if self.append:
            mode = "a"
        else:
            mode = "w"
        
        self.file = open(self.filename, mode, newline="")
        self.writer = None
    
    def on_epoch_end(self, model, epoch, logs):
        row = {"epoch": epoch + 1}
        row.update(logs)

        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=row.keys())

            if not self.append or os.path.getsize(self.filename) == 0:
                self.writer.writeheader()

        self.writer.writerow(row)
        self.file.flush()

    def on_train_end(self, model):
        if self.file is not None:
            self.file.close()


class LearningRateSchedule(Callbacks):
    def __init__(self, epoch_decay=10, epoch_amount=20):
        self.epoch_decay = epoch_decay
        self.epoch_amount = epoch_amount
    
    def on_epoch_begin(self, model, epoch):
        self.model = model

        if (epoch + 1) % self.epoch_amount == 0:
            self.model.Optimiser.learning_rate /= self.epoch_decay

class ReduceLROnPlateau(Callbacks):
    def __init__(self, monitor="val_loss", patience=5, min_delta=0, mode="min", factor=10):
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.factor = factor

    def on_train_begin(self, model):
        self.wait = 0

        if self.mode == "min":
            self.best = float("inf")
        else:
            self.best = -float("inf")

    def on_epoch_end(self, model, epoch, logs):
        val_loss = logs[self.monitor]

        if self.mode == "min":
            improved = val_loss < self.best - self.min_delta
        else:
            improved = val_loss > self.best + self.min_delta
        
        if improved:
            self.wait = 0
            self.best = val_loss
        else:
            self.wait += 1

        if self.wait >= self.patience:
            model.Optimiser.learning_rate /= self.factor
            print(f"New Learning Rate: {model.Optimiser.learning_rate}")
            self.wait = 0


class ProgressBar(Callbacks):
    def __init__(self, mode="batch", bar_length=40):
        self.mode = mode
        self.bar_length = bar_length
    
    def on_train_begin(self, model):
        self.current_epoch = 0
        self.total_epochs = model.epochs
    
    def on_epoch_begin(self, model, epoch):
        self.current_batch = 0
        self.total_batchs = model.total_batches

        if self.mode == "epoch":
            self.current_epoch += 1

            progress = self.current_epoch / self.total_epochs
            filled = int(self.bar_length * progress)

            bar = "█" * filled + "-" * (self.bar_length - filled)

            print(f"\nEpoch {epoch + 1}/{self.total_epochs}")
            print(f"[{bar}] {progress:.0%}", end="")
            print("\n")

            if self.current_epoch == self.total_epochs:
                print()


    def on_batch_end(self, model, epoch, logs):
        if self.mode == "batch":
            self.current_batch += 1

            progress = self.current_batch / self.total_batchs
            filled = int(self.bar_length * progress)

            bar = "█" * filled + "-" * (self.bar_length - filled)

            print(f"\n\r[{bar}] {progress:.0%}", end="")

            if self.current_batch == self.total_batchs:
                print()
    
    def on_epoch_end(self, model, epoch, logs):
        pass

class TerminateOnNan(Callbacks):
    def on_batch_end(self, model, epoch, logs):
        val_loss = logs["loss"]

        if np.isnan(val_loss) or np.isinf(val_loss):
            model.stop_training = True


class Timer(Callbacks):
    def on_train_begin(self, model):
        self.train_start = time.perf_counter()

    def on_epoch_begin(self, model, epoch):
        self.epoch_start = time.perf_counter()

    def on_epoch_end(self, model, epoch, logs):
        self.epoch_end = time.perf_counter()
        difference = self.epoch_end - self.epoch_start

        print(f"Epoch {epoch + 1} took {difference:.1f}s seconds.")

    def on_train_end(self, model, logs):
        self.train_end = time.perf_counter()
        difference = self.train_end - self.train_start

        print(f"Training Completed - {difference}s.")