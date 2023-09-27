import sys
import argparse
from online_policy_adaptation_using_rollout.ppo_base_policy.training_ppo_agent import PPOBase
from online_policy_adaptation_using_rollout.evaluation.evaluation import EvaluateBasePolicy

if __name__ == '__main__':

    args = sys.argv

    parser = argparse.ArgumentParser(description='Please check the code for options!')
    parser.add_argument("--seed", type=int, default=83)
    parser.add_argument("--target_path_for_models", type=str, default="../trained_models/scenario_3/")
    parser.add_argument("--target_path_to_save_results", type=str, default="../artifacts/scenario_3/")
    parser.add_argument("--num_neurons_per_hidden_layer", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--steps_between_updates", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--num_training_timesteps", type=int, default=1000000)
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--ent_coef", type=float, default=0.05)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--path_to_system_model", type=str, default="../trained_models/scenario_3")
    parser.add_argument("--environment_name", type=str, default="routing-env-v3")
    parser.add_argument("--last_iterarion", type=int, default=4)

    args = parser.parse_args()

    path_to_save_results = args.target_path_to_save_results
    path_to_save_models = args.target_path_for_models
    path_to_system_model = args.path_to_system_model

    seed = args.seed
    num_neurons_per_hidden_layer = args.num_neurons_per_hidden_layer
    num_layers = args.num_layers
    steps_between_updates = args.steps_between_updates
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    device = args.device
    gamma = args.gamma
    num_training_timesteps = args.num_training_timesteps
    verbose = args.verbose
    ent_coef = args.ent_coef
    clip_range = args.clip_range
    environment_name = args.environment_name
    last_iteration = args.last_iteration

    print(
        "You can update the parameters by invoking function with the following options: \n"
        "\t--seed 83\n"
        "\t--target_path_for_models ../../trained_models/scenario_3/\n"
        "\t--target_path_to_save_results ../../artifacts/scenario_3/\n"
        "\t--num_neurons_per_hidden_layer 128\n"
        "\t--num_layers 3\n"
        "\t--steps_between_updates 512\n"
        "\t--learning_rate 0.0005\n"
        "\t--batch_size 64\n"
        "\t--device cpu\n"
        "\t--gamma 0.99\n"
        "\t--num_training_timesteps 1000000"
        "\t--verbose 0\n"
        "\t--ent_coef 0.05\n"
        "\t--clip_range 0.2"
        "\t--path_to_system_model ../../trained_models/scenario_3"
        "\t--environment_name routing-env-v3"
        "--last_iterarion 4"
    )

    ####### Training the base PPO policy
    # training_obj = PPOBase(environment_name, path_to_system_model + "system_model.joblib", num_neurons_per_hidden_layer,
    # num_layers, seed, path_to_save_models,
    # path_to_save_results, verbose, steps_between_updates, batch_size, learning_rate, gamma, ent_coef,
    # clip_range, num_training_timesteps, device)
    #
    # training_obj.train_ppo_base()

    ####### Offline training the base PPO policy
    # training_obj = PPOBase(environment_name, path_to_system_model + "new_system_model.joblib", num_neurons_per_hidden_layer,
    # num_layers, seed, path_to_save_models,
    # path_to_save_results, verbose, steps_between_updates, batch_size, learning_rate, gamma, ent_coef,
    # clip_range, num_training_timesteps, device,policy_name="self_routing_new_")
    #
    # training_obj.train_ppo_base()

    ####### Evaluation of the base policy in the environment before change
    # evaluation_obj = EvaluateBasePolicy(path_to_base_model=path_to_save_models, model_name="self_routing_"+str(seed)
    #                                       + "_" + str(last_iteration) + ".zip",
    #                                     target_path_to_save_evalaution_results=path_to_save_results, episode_lenth=8,
    #                                     env_name=environment_name, path_to_system_model=path_to_system_model + "system_model.joblib")
    #
    # rewards, infos = evaluation_obj.evaluate_with_base_or_offline()
    #
    # print(rewards, infos)

    ####### Evaluation of the base policy in the new environment before change
    # evaluation_obj_new = EvaluateBasePolicy(path_to_base_model=path_to_save_models,
    #                                     model_name="self_routing_" + str(seed) + "_" + str(last_iteration)+".zip",
    #                                     target_path_to_save_evalaution_results=path_to_save_results, episode_lenth=8,
    #                                     env_name=environment_name,
    #                                     path_to_system_model=path_to_system_model + "new_system_model.joblib")
    #
    # rewards_new, infos_new = evaluation_obj_new.evaluate_with_base_or_offline()
    #
    # print(rewards_new, infos_new)

    ###### Evaluation of the base policy in the new environment before change
    evaluation_obj_new = EvaluateBasePolicy(path_to_base_model=path_to_save_models,
                                        model_name="self_routing_" + str(seed) + "_" + str(last_iteration) + ".zip",
                                        target_path_to_save_evalaution_results=path_to_save_results, episode_lenth=8,
                                        env_name=environment_name,
                                        path_to_system_model=path_to_system_model + "new_system_model.joblib")

    rewards_new, infos_new = evaluation_obj_new.evaluate_with_rollout()

    print(rewards_new, infos_new)

    ####### Evaluatiton of the offline retrained in the new environment
    # evaluation_obj_new_offline = EvaluateBasePolicy(path_to_base_model=path_to_save_models,
    #                                         model_name="self_routing_new_" + str(seed)  + "_" + str(last_iteration) + ".zip",
    #                                         target_path_to_save_evalaution_results=path_to_save_results,
    #                                         episode_lenth=8,
    #                                         env_name=environment_name,
    #                                         path_to_system_model=path_to_system_model + "new_system_model.joblib")
    #
    # rewards_new_offline, infos_new_offline = evaluation_obj_new_offline.evaluate_with_base_or_offline()
    #
    # print(rewards_new_offline, infos_new_offline)

    ####### Evaluatiton of the offline retrained in the new environment
    # evaluation_obj_new = EvaluateBasePolicy(path_to_base_model=path_to_save_models,
    #                                         model_name="self_routing_new_" + str(seed)  + "_" + str(last_iteration) + ".zip",
    #                                         target_path_to_save_evalaution_results=path_to_save_results,
    #                                         episode_lenth=8,
    #                                         env_name=environment_name,
    #                                         path_to_system_model=path_to_system_model + "new_system_model.joblib")
    #
    # rewards_new, infos_new = evaluation_obj_new.evaluate_with_base_or_offline()
    #
    # print(rewards_new, infos_new)