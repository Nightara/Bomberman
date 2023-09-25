# Bomberman
## Repository setup
We will create one separate branch for every agent, no agent runs in the main
branch.
Most agents will also require certain environments to test them in, so maybe we can
create environments in main to share them with each other.


## Usage

Please use the mind_puppets agents(for competition) and the other agent we using q-learning(for experimental purpose/backup) is inside the folder My_code-->MY_CODE_NEW

## How to run

- python main.py play --agent our_agent rule_based_agent rule_based_agent rule_based_agent (our agent against 3 rule based agents)

## General concept
- Main "our_agent" agent supervises multiple sub-agents, each of them specialized
in a single task, and chooses among the suggested moves provided by each sub-agent.
- Rule-based "harness" prevents governor from choosing invalid moves, forces it to
pick a different move if the chosen move would lead to an illegal action (E.g.
trying to walk into a wall or box).
- Every agent takes care of a single task, and contains the same "survival" agent
overriding any agent moves if survival is at risk (E.g. running out of bombs).

## Different Agents
- Governor (See above).
- Survival (See above).
  - Metric: Number of turns survived?
  - Local vision?
  - Neural network?
- Collecting coins.
  - Global vision?
  - Metric: Coins collected per turn?
- Creating coins.
  - Blow up crates.
  - Global vision.
  - Metric: Coins created per turn?
- Find way to opponents
  - Metric: Distance to opponents?
  - Global vision.
- Destroy opponents.
  - Local vision?
  - Metric: Opponents destroyed per turn?
- Block opponents?
  - Prevent oppoents from escaping bombs? Keep opponents in a small part of the
arena?
  - Metric: ???
  - Local vision?

## Task Split (Will be updated as we go)
### Florian
- Survival agent.
- Pathfinding agent.

### Anu
- Coin collecting agent.

### Keerthan Ugrani
- Bombing agent
