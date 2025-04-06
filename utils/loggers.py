import wandb


class MetricTracker:
    def __init__(self, name, nan=False):
        self.metric_name = name
        self.values = []
        self.sum = 0
        self.steps_count = 0
        self.nan = nan

    def update(self, value, step_length=1):
        self.values.append(value)
        if not self.nan:
            self.sum += value
            self.steps_count += step_length

    def reset(self):
        self.values = []
        self.sum = 0
        self.steps_count = 0

    @property
    def mean(self):
        if self.nan:
            raise ValueError("Cannot calculate average for metric with nan=True")
        return self.sum / self.steps_count


# wrapper is based in the reference:
# https://github.com/wandb/examples/blob/master/colabs/wandb-log/Log_(Almost)_Anything_with_W%26B_Media.ipynb
class WandbLogger:
    def __init__(self, project_name, config):
        wandb.login()

        wandb.init(project=project_name, config=config)

        self.project_name = project_name

    def log(self, metrics, step):
        wandb.log(metrics, step=step)

    def finish(self):
        wandb.finish()

    def log_image(self, image, caption, step):
        wandb.log({"image": [self.wandb.Image(image, caption)]}, step=step)
