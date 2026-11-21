
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from dataset import prepare_dataset
from model import MiniTransformer

def main():
    torch.manual_seed(42)

    inputs, targets, word_to_idx, idx_to_word = prepare_dataset()

    X = torch.tensor(inputs, dtype=torch.long)
    y = torch.tensor(targets, dtype=torch.long)

    model = MiniTransformer(
        vocab_size=len(word_to_idx),
        embedding_dim=32
    )
    
    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
    )

    dataset = TensorDataset(X, y)

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True
    )

    for epoch in range(30):
        model.train()

        total_loss = 0.00
        total_samples = 0
        for batch_x, batch_y in loader:

            optimizer.zero_grad()
            
            logits = model(batch_x)

            loss = criterion(logits, batch_y)

            loss.backward()

            optimizer.step()
                    
            batch_size = batch_x.size(0)

            total_loss += loss.item() * batch_size
            total_samples += batch_size

        # Средняя ошибка за эпоху
        avg_loss = total_loss / total_samples

        print(
            f"Epoch {epoch + 1:02d}/15 | "
            f"Avg loss: {avg_loss:.4f}"
        )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "word_to_idx": word_to_idx,
            "idx_to_word": idx_to_word,
        },
        "out/model.pt",
    )

    print("Модель сохранена!")


if __name__ == "__main__":
    main()