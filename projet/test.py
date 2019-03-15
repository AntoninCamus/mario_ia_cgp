import cv2

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

env = gym_super_mario_bros.make('SuperMarioBros-2-1-v3')
env = BinarySpaceToDiscreteSpaceEnv(env, SIMPLE_MOVEMENT)

done = True
map = {}
oldmap = {}
i: int = 0

for step in range(500000):
    if done:
        state = env.reset()
    # 1 px = 18*18
    state, reward, done, info = env.step(env.action_space.sample())

    # for column in state:
    #    for pixel in column:
    #        col = (pixel[0], pixel[1], pixel[2])
    #        if map.get(col) is None:
    #            map[col] = i
    #            i += 1
    # if(oldmap is map):
    #    oldmap = map
    #    print(map)
    # state = cv2.resize(state, (15, 14))
    state = cv2.cvtColor(state, cv2.COLOR_BGR2GRAY)
    cv2.imshow('main', state)
    cv2.waitKey(1)
    print(state)
    # np.reshape(state, (15,14))
    env.render()

env.close()
