# common/early_stopper.py

class EarlyStopper:
    def __init__(self, patience=10, min_delta=1e-3, mode="max", target_value=None, monitor="loss"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.target_value = target_value
        self.monitor = monitor
        self.reset()

    def reset(self):
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.target_reached = False

    def step(self, score):
        if self.target_value is not None:
            if self.mode == "max" and score >= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True
            if self.mode == "min" and score <= self.target_value:
                self.target_reached = True
                self.early_stop = True
                return True

        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        elif self.mode == "min":
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.early_stop = True
            return True

        return False

    def get_info(self):
        target_str = f", target={self.target_value}" if self.target_value is not None else ""
        if self.best_score is None:
            return f"Best {self.monitor}: N/A (patience: 0/{self.patience}{target_str})"
        return f"Best {self.monitor}: {self.best_score:.4f} (patience: {self.counter}/{self.patience}{target_str})"