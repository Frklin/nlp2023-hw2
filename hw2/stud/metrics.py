import wandb
import config



def compute_metrics(final_predictions: dict, mode: str):
    """
    Compute the metrics for the fine-grained task (accuracy)

    Args:
    - final_predictions: a dictionary containing the final predictions for each instance
    - mode: a string representing the mode of the predictions (train, val, test)
    """

    correct = 0
    total = 0

    for instance_id, pred in final_predictions.items():
        if pred == config.single_label_pairs_fine[instance_id]:
            correct += 1
        total += 1

    accuracy = round(correct/total, 3) if total > 0 else 0

    wandb.log({f'{mode}_accuracy': accuracy})
