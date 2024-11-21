class EarlyStopping:
    def __init__(self, patience: int = 20, epsilon: float = 1e-4):
        self.patience = patience
        self.epsilon = epsilon

        self.counter = 0
        self.min_loss = float("inf")

    def __call__(self, validation_loss: float) -> bool:
        if validation_loss < self.min_loss - self.epsilon:
            self.counter = 0
            self.min_loss = validation_loss
        else:
            self.counter += 1

        return self.counter >= self.patience
