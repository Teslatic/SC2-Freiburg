from assets.memory.ReplayBuffer import ReplayBuffer

# SER needs the skill length as additional information to calculate the TD-target
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'horizon'))

class SERB(ReplayBuffer):
    ''' Skill Experience Replay Buffer Class '''
    def __init__(self, capacity):
    	super(SERB, self).__init__(capacity)
