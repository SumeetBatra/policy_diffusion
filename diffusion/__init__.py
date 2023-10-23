from diffusion.schedules import *

BETA_SCHEDULE_REGISTRY = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
    "quadratic": quadratic_beta_schedule,
    "sigmoid": sigmoid_beta_schedule
}