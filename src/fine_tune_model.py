import os
from datasets import load_dataset, concatenate_datasets
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


def load_corpora():
    """Load Sejong and Modu corpora from the Hugging Face Hub."""
    sejong = load_dataset("Nambeom/Sejong_llama3_Kor_sample", split="train")
    modu = load_dataset("modu_team/modu_corpus", split="train")
    return concatenate_datasets([sejong, modu])["text"]


def fine_tune(base_model: str = "klue/bert-base", output_dir: str = "fine_tuned_model"):
    texts = load_corpora()
    examples = [InputExample(texts=[t, t]) for t in texts]
    model = SentenceTransformer(base_model)
    loader = DataLoader(examples, batch_size=32, shuffle=True)
    loss_fn = losses.MultipleNegativesRankingLoss(model)
    model.fit(train_dataloader=loader, epochs=1, loss=loss_fn)
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)


if __name__ == "__main__":
    fine_tune()
