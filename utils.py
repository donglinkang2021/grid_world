from typing import Tuple
import matplotlib.patches as patches

# RGB colors
COLOR_FORBID = (0.9290,0.6940,0.125)
COLOR_TARGET = (0.3010,0.7450,0.9330)
COLOR_POLICY = (0.4660,0.6740,0.1880)
COLOR_TRAJECTORY = (0, 1, 0)
COLOR_AGENT = (0,0,1)

def state2rect(state:Tuple[int, int], color:Tuple[float, float, float]):
    return patches.Rectangle(
        xy=(
            state[0]-0.5, 
            state[1]-0.5
        ),
        width=1, height=1, 
        linewidth=1, 
        edgecolor=color, facecolor=color
    )