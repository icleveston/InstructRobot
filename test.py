from Environment import Environment
from Environment.InstructionSet import InstructionSet
from PIL import Image


def main():

    instruct_cubes_simple = InstructionSet()

    env = Environment('Scenes/Cubes_Simple.ttt', instruction_set=instruct_cubes_simple, random_seed=1)

    action = []

    reward_total = 0
    for i in range(3):
        obs, reward = env.step(action)

        reward_total += reward

    print(f"Active Instruction: {obs[-1][0]} - Reward: {reward_total}")

    visualize_observations(obs)


def visualize_observations(obs: []):
    for o in obs:
        print(o[0])

        im = Image.fromarray(o[2], mode="RGB")
        im.show()


if __name__ == "__main__":
    main()
