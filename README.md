
## Mountain Car
In the mountain car task, the agent is required to move a car uphill to a flag.
Unfortunately, the engine is not powerful enough to make the move directly.
Instead, you need to utilize the U shape hill to accumulate some energy before
you can reach the flag.

Unlike the simple case of Cart Pole, in which we use the reward returned by the
environment to determine the next step, in the Mountain Car case the reward does
not give you any hint where to move. It always returns -1 unless you reach the flag.
If you just use the reward to train an agent, as we did in previous cases, you 
will see the car drives back and forth at the bottom of the valley, but unable to
make any improvement, as no matter where it moves, the reward is always the same.

We turn to think what reward makes the car moving higher. The answer is almost
immediate: energy. With more energy will the car move higher and faster. When
it has enough energy, it will for sure reach the flag. So we need to teach the car
to accumulate more energy. A straightforward way is to code the car to move upward 
when going uphill, and move downward when going downhill to achieve this. 

Instead of taking this approach, here we again use the idea of Q-Learning, and use 
a DQN to let the car learns to increase energy. 
We discard the reward provided by the environment, and compute the reward as following

    def custom_reward(state):
        data = state[0]
        current_pos = data[0]
        current_v = data[1]
        height = abs(current_pos+0.5)
        return 100*current_v*current_v + height

The square of speed computes the kinetic energy, and the height computes the poential
energy, which sums up to the total energy of the car. When the car learns to increase
the total energy, it can easily reach the top of the hill.

Part of the training log is shown below. Just after 10 rounds, the car learns to climb
up the hill to the flag.

    Run: 6, exploration: 0.0027677054964881736, reward: 0.4911656598784185
    Run: 7, exploration: 0.0010785966235237919, reward: 1.0163097117814361
    Run: 8, exploration: 0.001, reward: 1.0865005623620014
    Run: 9, exploration: 0.001, reward: 0.9168002551655375
    Run: 10, exploration: 0.001, reward: 1.2002587158325773
    Run: 11, exploration: 0.001, reward: 1.1239015870528513
