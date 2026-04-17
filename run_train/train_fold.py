import os
from pathlib import Path

import torch
import torch.nn as nn

from models.predictor import DDGPredictor
from models.train import run_model, save_model, train_init
from utils.datasets import (
    get_batch,
    load_default_settings,
    load_model_data,
    load_weight,
    save_procedure_parameter,
    update_requires_grad,
)


RUN_TRAIN_DIR = Path(__file__).resolve().parent


def run_training(config, fold_script_name, validation_flag, label, debug=False, continue_train=False):
    os.chdir(RUN_TRAIN_DIR)
    ori_weight, config, weight_name, py_name, logger = load_default_settings(
        config,
        fold_script_name,
        label,
        validation_flag=validation_flag,
        debug=debug,
        continue_train=continue_train,
    )
    if str(config.model.device).startswith("cuda") and not torch.cuda.is_available():
        logger.warning("CUDA is not available; falling back to CPU for training.")
        config.model.device = "cpu"
    train_dataset, validation_dataset, test_dataset, contact_area_dict = load_model_data(
        config,
        validation_flag=validation_flag,
        label=label,
    )
    train_dataset, validation_dataset, test_dataset = get_batch(
        train_dataset,
        validation_dataset,
        test_dataset,
        config,
    )

    model = DDGPredictor(config).to(config.model.device)
    if continue_train:
        model.load_state_dict(load_weight(ori_weight), strict=True)

    if hasattr(torch, "compile") and str(config.model.device).startswith("cuda"):
        model = torch.compile(model)
    torch.set_float32_matmul_precision("high")

    runner = run_model(config)
    loss_fn = nn.MSELoss(reduction="mean")
    optimizer = update_requires_grad(config, model, bool_=config.feature.freeze_parameter)

    (
        train_losses_epoch,
        train_Rp,
        validation_losses_epoch,
        validation_Rp,
        test_losses_epoch,
        test_Rp,
        continue_train_epoch,
    ) = train_init(
        label,
        py_name,
        weight_name=weight_name,
        validation_flag=validation_flag,
        continue_train=continue_train,
    )

    lr = config.train.optimizer.lr
    val_flag = 0
    for epoch in range(continue_train_epoch, config.train.max_iters):
        if weight_name is not None and epoch < int(weight_name.split("_")[-1].split(".")[0]):
            continue

        logger.info("\n---------------------------------train----------------------------------------------------")
        train_result_dict = runner.make_train_step(
            model,
            train_dataset,
            contact_area_dict,
            loss_fn,
            optimizer,
            lr,
            epoch,
        )
        runner.processing_result(
            train_result_dict,
            epoch,
            train_Rp,
            train_losses_epoch,
            logger,
            py_name,
            "train",
            index="Rp",
        )

        if validation_dataset:
            logger.info("\n----------------------------validation-------------------------------------------------")
            validation_result_dict = runner.make_validation_step(model, validation_dataset, contact_area_dict, loss_fn)
            runner.processing_result(
                validation_result_dict,
                epoch,
                validation_Rp,
                validation_losses_epoch,
                logger,
                py_name,
                "validation",
                index="Rp",
            )
            val_flag, lr = runner.leaning_rate_config(
                validation_result_dict,
                validation_losses_epoch,
                epoch,
                val_flag,
                lr,
            )

        logger.info("\n--------------------------------test------------------------------------------------------")
        test_result_dict = runner.make_validation_step(model, test_dataset, contact_area_dict, loss_fn)
        runner.processing_result(
            test_result_dict,
            epoch,
            test_Rp,
            test_losses_epoch,
            logger,
            py_name,
            "test",
            index="Rp",
        )

        logger.info("\n***********************************************************************************\n")
        if config.model.save_model:
            save_model(model.state_dict(), config, epoch, test_Rp, py_name, test_result_dict, continue_train=continue_train)

        save_procedure_parameter(
            config,
            validation_flag,
            label,
            epoch,
            py_name,
            train_Rp,
            train_losses_epoch,
            validation_Rp,
            validation_losses_epoch,
            test_Rp,
            test_losses_epoch,
        )
    return max(test_Rp) if test_Rp else None
