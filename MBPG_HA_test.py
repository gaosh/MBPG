import torch
import garage
from garage.experiment import run_experiment
from garage.experiment import LocalRunner
from garage.np.baselines import LinearFeatureBaseline
from garage.tf.envs import TfEnv
from Policy import GaussianMLPPolicy, CategoricalMLPPolicy
from Algorithms.MBPG_HA import MBPG_HA
from gym.envs.mujoco import Walker2dEnv, HopperEnv, HalfCheetahEnv
from gym.envs.classic_control import CartPoleEnv
#from garage.envs.box2d import CartpoleEnv
from garage.envs import normalize
import argparse
parser = argparse.ArgumentParser(description='HA-MBPG')
parser.add_argument('--env', default='CartPole', type=str, help='choose environment from [CartPole, Walker, Hopper, HalfCheetah]')

args = parser.parse_args()


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task.
    Args:
        snapshot_config (garage.experiment.SnapshotConfig): The snapshot
            configuration used by LocalRunner to create the snapshotter.
            If None, it will create one with default settings.
        _ : Unused parameters
    """

    th = 1.8
    g_max = 0.1
    #delta = 1e-7
    if args.env == 'CartPole':
        #CartPole

        env = TfEnv(normalize(CartPoleEnv()))
        runner = LocalRunner(snapshot_config)
        batch_size = 5000
        max_length = 100
        n_timestep = 5e5
        n_counts = 5
        name = 'CartPole'
        grad_factor = 5
        th = 1.2
        #batchsize: 1
        # lr = 0.1
        # w = 2
        # c = 50

        #batchsize: 50
        lr = 0.75
        c = 3
        w = 2

        discount = 0.995
        path = './init/CartPole_policy.pth'

    if args.env == 'Walker':
        #Walker_2d
        env = TfEnv(normalize(Walker2dEnv()))
        runner = LocalRunner(snapshot_config)
        batch_size = 50000
        max_length = 500

        n_timestep = 1e7
        n_counts = 5
        lr = 0.75
        w = 2
        c = 12
        grad_factor = 6

        discount = 0.999

        name = 'Walk'
        path = './init/Walk_policy.pth'

    if args.env == 'HalfCheetah':
        env = TfEnv(normalize(HalfCheetahEnv()))
        runner = LocalRunner(snapshot_config)

        batch_size = 50000
        max_length = 500

        n_timestep = 1e7
        n_counts = 5
        lr = 0.6
        w = 1
        c = 4
        grad_factor = 5
        th = 1.2
        g_max = 0.06

        discount = 0.999

        name = 'HalfCheetah'
        path = './init/HalfCheetah_policy.pth'

    if args.env == 'Hopper':
        #Hopper
        env = TfEnv(normalize(HopperEnv()))
        runner = LocalRunner(snapshot_config)

        batch_size = 50000
        max_length = 1000
        th = 1.5
        n_timestep = 1e7
        n_counts = 5
        lr = 0.75
        w = 1
        c = 3
        grad_factor = 6
        g_max = 0.15
        discount = 0.999

        name = 'Hopper'
        path = './init/Hopper_policy.pth'

    for i in range(n_counts):
        # print(env.spec)
        if args.env == 'CartPole':
            policy = CategoricalMLPPolicy(env.spec,
                                       hidden_sizes=[8, 8],
                                       hidden_nonlinearity=torch.tanh,
                                       output_nonlinearity=None)
        else:
            policy = GaussianMLPPolicy(env.spec,
                                       hidden_sizes=[64, 64],
                                       hidden_nonlinearity=torch.tanh,
                                       output_nonlinearity=None)


        policy.load_state_dict(torch.load(path))
        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = MBPG_HA(env_spec=env.spec,
                   env = env,
                    env_name= name,
                   policy=policy,
                   baseline=baseline,
                   max_path_length=max_length,
                   discount=discount,
                   grad_factor=grad_factor,
                   policy_lr= lr,
                   c = c,
                   w = w,
                   th=th,
                   g_max=g_max,
                   n_timestep=n_timestep,

                   batch_size=batch_size,
                   center_adv=True,
                   # delta=delta
                   #decay_learning_rate=d_lr,

                   )

        runner.setup(algo, env)
        runner.train(n_epochs=100, batch_size=batch_size)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)