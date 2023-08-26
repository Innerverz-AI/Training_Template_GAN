import sys

sys.path.append("./")
from lib import utils
from core.model import MyModel
from accelerate import Accelerator
import warnings

warnings.filterwarnings("ignore")


def train():
    accelerator = Accelerator(log_with="wandb")
    CONFIG = utils.load_jsonnet("./configs.jsonnet")

    if accelerator.is_main_process:
        CONFIG = utils.prepare_training()

    model = MyModel(CONFIG, accelerator)
    model.accelerator.init_trackers(CONFIG["BASE"]["MODEL_ID"], config=CONFIG)

    while CONFIG["BASE"]["GLOBAL_STEP"] < CONFIG["BASE"]["MAX_STEP"]:
        model.go_step()
        model.accelerator.wait_for_everyone()

        if model.accelerator.is_main_process:
            if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["LOSS"] == 0:
                model.loss_collector.print_loss()
                model.accelerator.log(model.loss_collector.loss_dict)

            if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["TRAIN_IMAGE"] == 0:
                model.save_grid_image(phase="train")

            if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["VALID_IMAGE"] == 0:
                model.do_validation()
                model.save_grid_image(phase="valid")

            if CONFIG["BASE"]["GLOBAL_STEP"] % CONFIG["CYCLE"]["CKPT"] == 0:
                model.save_checkpoint()

        CONFIG["BASE"]["GLOBAL_STEP"] += 1

    model.accelerator.end_training()


if __name__ == "__main__":
    train()
