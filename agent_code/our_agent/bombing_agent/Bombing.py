import torch
import numpy as np
from collections import deque
import heapq


POSSIBLE_STEPS = np.array([[1,0], [-1,0], [0,1], [0,-1]])

POSSIBLE_DIRECT = {(1,0):0, (-1,0):1, (0,1):2, (0,-1):3}

SEARCH_DIST_LIMIT = 30

MAX_CRATE_POS = 3

MAX_DEAD_ENDS = 2

CRATE_MAX_COUNT = MAX_DEAD_ENDS+MAX_CRATE_POS

def state_to_features(self, game_state: dict) -> torch.tensor:

    maximum_coin_length = 9 
    hero_position = np.array(game_state["self"][3])

    coins = np.array(game_state["coins"])
    coin_collection = coins.tolist()
    coin_total = len(coins)

    field = np.array(game_state["field"])
    explosions = np.array(game_state["explosion_map"])
    bombs = game_state["bombs"]

    if game_state is None:
        return None


    def neighbor_options(pos):
        return [new_pos.tolist() for new_pos in (pos + POSSIBLE_STEPS) if field[new_pos[0], new_pos[1]] == 0]


    def explosion_effect(pos):
        crates_blown = 0

        for direction in POSSIBLE_STEPS:
            for length in range(1, 4):
                x, y = direction * length + pos
                obj = field[x, y]

                if obj == -1:
                    break

                if obj == 1 and future_explosion_map[x, y] == 1:
                    crates_blown += 1

        return crates_blown



    def fill_explosion_map(explosions, bombs, field):
        future_explosion_map = (np.copy(explosions)*-4) + 1 
        for bomb in bombs:
            pos = np.array(bomb[0])
            timer = bomb[1] - 3 
            field[pos[0], pos[1]] = -2 

            for direction in POSSIBLE_STEPS:
                for length in range(0, 4):
                    beam = direction*length + pos
                    obj = field[beam[0], beam[1]]
                    if obj == -1:
                        break
                    if future_explosion_map[beam[0], beam[1]] > timer:
                        future_explosion_map[beam[0], beam[1]] = timer

        return future_explosion_map
    
    def future_explosion_map_new(future_explosion_map, pos):
        new_future_explosion_map = np.copy(future_explosion_map)
        new_future_explosion_map[new_future_explosion_map < 1] -= 1
        new_future_explosion_map[new_future_explosion_map < -3] = 1
        timer = 0

        for direction in POSSIBLE_STEPS:
            for length in range(4):
                x, y = direction * length + pos
                obj = field[x, y]

                if obj == -1:
                    break

                if new_future_explosion_map[x, y] > timer:
                    new_future_explosion_map[x, y] = timer

        return new_future_explosion_map


    def threat(pos, future_explosion_map, turns=0, forbidden_fields = None):
        
        q = deque()

        visited = []

        if forbidden_fields is not None:
            for forbidden_pos in forbidden_fields:
                visited.append(forbidden_pos)

        q.append((pos.tolist(), turns))
        while len(q):
            pos, turns = q.popleft()

            if turns > 4:
                break

            if pos in visited:
                continue


            if turns-1 - future_explosion_map[pos[0], pos[1]] == 3:
                continue
            
            if future_explosion_map[pos[0], pos[1]] == 1:
                return False

            visited.append(pos)
            for neighbor in neighbor_options(pos):
                q.append((neighbor, turns+1))
                
        return True

    crates = np.argwhere(field==1)
    total_crates = len(crates)
    future_explosion_map = fill_explosion_map(explosions, bombs, field)

    if coin_total == 0:
        coins = np.zeros((maximum_coin_length, 2))

    # needed to find the possible moves
    next_position_options = neighbor_options(hero_position)

    # create the result arrays
    inv_coins = [[] for _ in range(4)]
    inv_crate_distances = [[] for _ in range(4)]
    crate_points = [[] for _ in range(4)]

    # create the distance arrays
    coin_distances_after_step = np.empty((4, maximum_coin_length))
    crate_distances_after_step = np.empty((4, CRATE_MAX_COUNT))

    # create the bomb effectiveness array
    expected_destructions_after_step = np.zeros((4, CRATE_MAX_COUNT))

    # Initialize the distance arrays, if no way can be found we consider the distance to be infinite
    coin_distances_after_step.fill(np.inf)
    crate_distances_after_step.fill(np.inf)

    # visited array for bfs
    visited = [hero_position.tolist()]

    # heap queue for bfs
    q = []
    for pos in (hero_position + POSSIBLE_STEPS):
        x, y = pos - hero_position
        heapq.heappush(q, (1, pos.tolist(), POSSIBLE_DIRECT[(x,y)]))

    # Counter for the step arrays
    number_of_found_crate_positions = np.zeros(4)
    number_of_found_dead_ends = np.zeros(4)
    number_of_found_coins = np.zeros(4)

    # condition to quit the search early
    found_one = False
    skipped = [False, False, False, False]

    while len(q) != 0:
        
        # direction = index of the STEP array of the first STEP, first index of our step arrays
        distance, pos, direction = heapq.heappop(q)

        # quit the search early if we found a target and if too much steps are exceeded (relevant if few crates)
        if (distance > SEARCH_DIST_LIMIT) and (found_one==True):
            break
        
        # skip allready visited positions
        if pos in visited:
            continue

        # mark the current node as visited
        visited.append(pos)
        

        # check for other obvious quit early conditions
        if distance == 1:
            # Safely blown up
            if future_explosion_map[pos[0], pos[1]]==-2:
                crate_points[direction] = np.zeros(CRATE_MAX_COUNT)
                placebo1 = np.zeros(CRATE_MAX_COUNT)
                placebo1.fill(-2)
                placebo2 = np.zeros(maximum_coin_length)
                placebo2.fill(-2)
                placebo3 = np.zeros(4)
                placebo3.fill(-2)
                inv_crate_distances[direction] = np.copy(placebo1)
                inv_coins[direction] = np.copy(placebo2)

                skipped[direction] = True
                continue

            if pos not in next_position_options:
                # we are walking against a wall or a crate
                crate_points[direction] = np.zeros(CRATE_MAX_COUNT)
                placebo1 = np.zeros(CRATE_MAX_COUNT)
                placebo1.fill(-1)
                placebo2 = np.zeros(maximum_coin_length)
                placebo2.fill(-1)
                placebo3 = np.zeros(4)
                placebo3.fill(-1)
                inv_crate_distances[direction] = np.copy(placebo1)
                inv_coins[direction] = np.copy(placebo2)

                skipped[direction] = True
                continue


        # coins
        is_coin = pos in coin_collection # check if pos is in coins -> we reached a coin
        if is_coin:
            coin_distances_after_step[direction][int(number_of_found_coins[direction])] = distance
            number_of_found_coins[direction] += 1
        if is_coin and not found_one:
            found_one = True



        neighbors = neighbor_options(pos)

        # visit all neighbors
        ways_out = 0
        for node in neighbors:
            ways_out += 1
            if (distance+1)<=3 and (future_explosion_map[node[0], node[1]] != 1):
                # estimate that we will loose ~ a turn, for each bomb field we cross beacuse of unsafty reasons (e.g. we have to wait)
                heapq.heappush(q, (distance+1, node, direction))
            heapq.heappush(q, (distance+1, node, direction))

        # crates
        if future_explosion_map[pos[0], pos[1]] != 1: # this position is already used -> dont drop a bomb
            continue

        dead_end = False
        if (ways_out == 1) and (number_of_found_dead_ends[direction] < MAX_DEAD_ENDS):
            # we found a unused dead end, this might be a good bomb position
            index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
            crate_distances_after_step[direction][index_crates] = distance
            expected_destructions_after_step[direction][index_crates] = explosion_effect(pos)

            dead_end = True
            number_of_found_dead_ends[direction] += 1
            found_one = True

        # This crates should be closer but are most likely not as good as the dead ends
        if (number_of_found_crate_positions[direction] < MAX_CRATE_POS) and not dead_end:
            for possible_crate in (pos + POSSIBLE_STEPS):
                if field[possible_crate[0], possible_crate[1]] == 1 and (future_explosion_map[possible_crate[0], possible_crate[1]]==1):
                    # one of the neighboring fields is a free crate
                    index_crates = int(number_of_found_crate_positions[direction] + number_of_found_dead_ends[direction])
                    crate_distances_after_step[direction][index_crates] = distance
                    expected_destructions_after_step[direction][index_crates] = explosion_effect(pos)

                    number_of_found_crate_positions[direction] += 1
                    found_one = True
                    break

    for direction in range(4):
        if skipped[direction]:
            continue
        inv_coins[direction] = 1/np.array(coin_distances_after_step[direction])

        inv_crate_distances[direction] = 1/np.array(crate_distances_after_step[direction])

        crate_points[direction] = np.array(expected_destructions_after_step[direction])

    inv_crate_distances = np.array(inv_crate_distances)

    crate_points = np.array(crate_points)

    inv_coins = np.array(inv_coins)


    features = []
    # append the coins feature
    features = np.append(features, np.max(inv_coins, axis=1))

    # append the crates features
    features = np.append(features, np.max(inv_crate_distances * crate_points, axis=1))

    # is it senseful to drop a bomb here?
    neighboring_chest = False
    neighboring_opponent = False
    if future_explosion_map[hero_position[0], hero_position[1]] == 1:
        for pos in hero_position + POSSIBLE_STEPS:
            if (field[pos[0], pos[1]] == 1) and (future_explosion_map[pos[0], pos[1]] == 1): # free crate
                neighboring_chest = True
            

    # Points for opponent and crates
    if neighboring_opponent:
        bomb_here = 5 + explosion_effect(hero_position)

    # Points only for the crates
    elif neighboring_chest:
        bomb_here = explosion_effect(hero_position)

    # We do not get anything if we drop a bomb here
    if not neighboring_chest and not neighboring_opponent:
        bomb_here = -1
    
    # We do not have our bomb
    if not game_state["self"][2]:
        bomb_here = -1

    new_future_explosion_map = future_explosion_map_new(future_explosion_map, hero_position)

    # We would kill ourself if we drop a bomb here
    if threat(hero_position, new_future_explosion_map):
        # print("Sicherer Tod")
        bomb_here = -1
    
    features = np.append(features, bomb_here)

    features = np.append(features,-(future_explosion_map[hero_position[0], hero_position[1]]-1))

    running = np.zeros(4)
    # no bomb at our position
    if future_explosion_map[hero_position[0], hero_position[1]] == 1:
        features = np.append(features, running)
    # Which way can we take without beeing blown up?
    else:
        direction = -1
        for pos in hero_position+POSSIBLE_STEPS:
            direction += 1
            if threat(pos, future_explosion_map, turns=1, forbidden_fields=[hero_position.tolist()]):
                running[direction] = -1
            if field[pos[0], pos[1]] != 0:
                running[direction] = -1

        features = np.append(features, running)

    danger = np.zeros(4)
    if future_explosion_map[hero_position[0], hero_position[1]] == 1: # current position is save
        dim = 0
        for pos in hero_position + POSSIBLE_STEPS:
            if future_explosion_map[pos[0], pos[1]] == -3:
                danger[dim] = -1
            dim += 1
    features = np.append(features, danger)

    features = np.append(features, coin_total + 1/3 * total_crates)
    
    # needed for the rulebased version
    self.features = features

    # needed for the crate features
    self.destroyed_crates = self.bomb_buffer
    self.bomb_buffer = explosion_effect(hero_position)


    # crate a torch tensor that can be returned from the features
    features = torch.from_numpy(features).float()

    return features.unsqueeze(0)