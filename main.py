from KResRL.ppo import EnvOptions, RLOptions, TrainOptions, train
from KResRL.policy.graph_policy import GraphPolicyOptions, GCNBlock, GraphFeatureOptions
from KResRL.policy.att_policy import SAB, ISAB, AttFeatureOptions, AttPolicyOptions

from KResRL.environment.env import KRes

def main():
    env_options = EnvOptions(
        n_envs=5,
        n_drones=5,
        k = 2,
        size=4,
        alpha=0.001
    )

    # features_extractor_options = GraphFeatureOptions(
    #     NNBlock=GCNBlock,
    #     hidden_dims=[128, 256, 128, 64],
    #     used_res=[False, True, True, False],
    #     norm="layer",
    #     act="relu",
    #     act_kwargs={"inplace": True},
    #     dropout=0.1,
    #     gnn_layer_kwargs={}  # Add GAT-specific params if using GATBlock
    # )

    features_extractor_options = AttFeatureOptions(
        NNBlock=SAB,
        n_layers=6
    )

    policy_options = AttPolicyOptions(
        features_extractor_kwargs=features_extractor_options,
        net_arch={"pi": [256, 128], "vf": [256, 128]},
    )

    rl_options = RLOptions(
        gamma=0.99,
        n_steps=256
    )

    train_options = TrainOptions(
        total_timesteps=100_000,
        log_interval=100
    )

    model = train(env_options, policy_options, rl_options, train_options)

    model.save("trained_model")


    env = KRes(
        n_drones=env_options.n_drones,
        k=env_options.k,
        size=env_options.size,
        return_state="features",
        normalize_features=True,
        render_mode="human",
        render_fps=1,
    )

    obs, _ = env.reset()

    for i in range(256):
        env.render()
        actions, _ = model.predict(obs)
        new_obs, reward, done, terminated, info = env.step(actions)

        if done:
            break

if __name__ == "__main__":
    main()
