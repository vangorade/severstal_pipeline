from train.train import Trainer
from model.model import model
from matplotlib import pyplot as plt


def plot(scores, name):
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(scores["train"])),
             scores["train"], label=f'train {name}')
    plt.plot(range(len(scores["train"])), scores["val"], label=f'val {name}')
    plt.title(f'{name} plot')
    plt.xlabel('Epoch')
    plt.ylabel(f'{name}')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data_folder = './data'
    df_path = './data/train.csv'
    model_trainer = Trainer(model, data_folder, df_path)
    model_trainer.start()

    # PLOT TRAINING
    losses = model_trainer.losses
    dice_scores = model_trainer.dice_scores  # overall dice
    iou_scores = model_trainer.iou_scores
    F_scores = model_trainer.F_scores

    plot(losses, "Focal loss")
    plt.savefig('./A1E2_loss.jpg', bbox_inches='tight')
    plot(dice_scores, "Dice score")
    plt.savefig('./A1E2_dice.jpg', bbox_inches='tight')
    plot(iou_scores, "IoU score")
    plt.savefig('./A1E2_iou.jpg', bbox_inches='tight')
    plot(F_scores, "F Score")
