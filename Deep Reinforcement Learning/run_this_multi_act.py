#from maze_env import Maze
from maze_2_multi import Maze
from RL_brain import DeepQNetwork
import numpy as np

def run_maze():
    step = 0
    final_10 = 1

    epnum = 500
    for episode in range(epnum):
        #print(episode)
        #final_50_err += env.step().err
        # initial observation
        observation = env.reset()

        while True:

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, err = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            #if (step > 200) and (step % 5 == 0):
            RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                #print 'done'
                if episode >= epnum-11:
                    final_10 = min(final_10,err)
                if episode == epnum-1:
                    print("Final_10 =", final_10)

                break
            else:
                #print 'no'
                pass
            step += 1

            # end of game
    print('game over')
    return final_10


if __name__ == "__main__":
    # maze game
    dttt=[np.pi/20]
    
    act_num=500
    
    tot_fid=[]
    for dtt in dttt:
        env = Maze(action_space=list(range(act_num)),   #allow two action
                   dt=dtt)               #dt =0.1
    
        error_dt=0
        for ii in range(100):
            if ii==0 and dtt==dttt[0]:
                bl = None
            else:
                bl = True
            
            RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.99,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.001,
                      bol=bl
                      )
            error = run_maze()
            error_dt += error
            print(error)
            
        tot_fid.append(1-error_dt/100)
        print ('fid',dtt , tot_fid)

#step=10
#3                   4                   5                    6                   10                   20                      50                  100                200                 500
#0.9894128527599264  0.9509941953097407  0.9484295297679259   0.9331105057968162  0.9304314656683493   0.816172393324893       0.8249762888957449  0.8640054330536214 0.7750634613207914  0.826130311207746
        
#step=20
#2                  3                   4                   5                    6                   10                   20                      50                  100                200                 500
#0.9991687856344036 0.9994831229496816  0.9767183555664241  0.9774196223588651   0.9879884087544282  0.9015842626808905   0.8597082686679622      0.8497780468676355  0.8568301376600078 0.8296730979004421  0.7620731163369928
        
        
        
        
        
        
