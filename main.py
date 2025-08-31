from KResRL.ppo import EnvOptions, RLOptions, TrainOptions, train
from KResRL.policy import GraphPolicyOptions, GCNBlock

def main():
    env_options = EnvOptions(
        n_envs=1,
        n_drones=20,
        k = 3,
        size=10,
    )


    policy_options = GraphPolicyOptions(
        NNBlock=GCNBlock,
        hidden_dims=[128, 256, 128, 64],
        used_res=[False, True, True, False],
        norm="layer",
        act="relu",
        act_kwargs={"inplace": True},
        dropout=0.1,
        net_arch={"pi": [256, 128], "vf": [256, 128]},
        gnn_layer_kwargs={}  # Add GAT-specific params if using GATBlock
    )

    rl_options = RLOptions(
        gamma=0.99,
    )

    train_options = TrainOptions(
        total_timesteps=1_000_000,
        log_interval=100
    )

    train(env_options, policy_options, rl_options, train_options)

if __name__ == "__main__":
    main()
