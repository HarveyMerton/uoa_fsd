from .ppo import PPO
from .sac import SAC, SACExpert
from .gail import GAIL
from .airl import AIRL, AIRLExpert

ALGOS = {
    'gail': GAIL,
    'airl': AIRL
}
