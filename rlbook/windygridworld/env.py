class Environment:

    def __init__(self):
        self.pos = (0, 0)
        self.wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

    def reset(self):
        self.pos = (3, 0)
        return self.pos

    '''
    action: 0 for up, 1 for right, 2 for down, 3 for right
    '''

    def step(self, action):
        posx = self.pos[0]
        posy = self.pos[1]

        if action == 0:
            posx -= 1
        elif action == 1:
            posy += 1
        elif action == 2:
            posx += 1
        elif action == 3:
            posy -= 1

        if posy < 0:
            posy = 0
        if posy >= 10:
            posy = 9
        posx -= self.wind[posy]

        if posx < 0:
            posx = 0
        if posx >= 7:
            posx = 6

        self.pos = (posx, posy)

        if self.pos[0] == 3 and self.pos[1] == 7:
            return self.pos, 0, True
        return self.pos, -1, False
