import numpy as np
import torch


class NLVR2Evaluator:
    """NLVR2 Evaluator."""

    def run_evaluation(
        self, predictions: list[str], ground_truth: list[str], sentences: list[str]
    ) -> dict[str, torch.Tensor]:
        """Evaluate all metrics (accuracy, consistency) for NLVR2."""
        accuracy = self.accuracy(predictions, ground_truth)
        consistency = self.consistency(predictions, ground_truth, sentences)
        return {"accuracy": accuracy, "consistency": consistency}

    def accuracy(self, predictions: list[str], ground_truth: list[str]) -> torch.Tensor:
        """Calculate accuracy of predictions based on each instance."""
        corrects = []
        for i, _ in enumerate(predictions):
            pred = predictions[i].lower()
            label = ground_truth[i]
            corrects.append(int(pred == label))
        return torch.tensor(100 * np.mean(np.array(corrects)), dtype=torch.float32)

    def consistency(
        self, predictions: list[str], ground_truth: list[str], sentences: list[str]
    ) -> torch.Tensor:
        """Calculate consistency of predictions based on instances with the same sentence."""
        sentence_dict: dict[str, list[int]] = {}
        for i, _ in enumerate(sentences):
            # Remove the task prefix if exist from the format 'Task_Prefix: NLVR2_Statement'
            sentence = sentences[i].split(":")[-1].strip()
            pred = predictions[i].lower()
            label = ground_truth[i].lower()
            if sentence not in sentence_dict.keys():
                sentence_dict[sentence] = []
            sentence_dict[sentence].append(int(pred == label))

        consistent = []
        for v in sentence_dict.values():
            is_consistent = int(np.all(np.array(v) == 1))
            consistent.append(is_consistent)

        return torch.tensor(100 * np.mean(np.array(consistent)), dtype=torch.float32)
