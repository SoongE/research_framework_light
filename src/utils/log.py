import logging


class Logger:
    def __init__(self, cfg, wandb):
        self.cfg = cfg
        self.wandb = wandb

        # self.base_path = hydra.utils.get_original_cwd()
        self._init_logger()

    def log(self, data):
        if self.logging:
            self.logging.info(data)
        if self.wandb_logger:
            self.wandb_logger.log(data)

    def _init_logger(self):
        if self.wandb:
            try:
                import wandb

                wandb.init(project=self.cfg.info.project, entity=self.cfg.info.entity, config=self.cfg,
                           # name=f"{datetime.now().strftime('%Y-%m-%d/%H:%M:%S')}/{self.cfg.name}",
                           name=f"{self.cfg.name}",
                           settings=wandb.Settings(_disable_stats=True), save_code=True, reinit=True)

                self.wandb_logger = wandb

            except (ImportError, AssertionError):
                prefix = 'Weights & Biases: '
                print(f"{prefix}run 'pip install wandb' to track and visualize.")

        self.logging = logging
