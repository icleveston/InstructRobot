import random

import numpy as np
from pyrep import PyRep
from pyrep.objects.shape import Shape
from pyrep.const import PrimitiveShape


def main():

    scene_file = 'Scenes/Main.ttt'

    pr = PyRep()
    # Launch the application with a scene file in headless mode
    pr.launch(scene_file)

    red = [1, 0, 0]
    green = [0, 1, 0]
    blue = [0, 0, 1]

    colors = [red, green, blue]

    w = 0.1
    h = 0.1
    d = 0.1

    x = .1
    y = .1
    z = 0.8

    object = Shape.create(type=PrimitiveShape.SPHERE,
                          color=random.choice(colors), size=[w, h, d],
                          position=[x, y, z])



    pr.start()  # Start the simulation
    pr.step()  # Step physics simulation

    done = False

    counter = 0

    while not done:
        pr.step()
        counter += 1

        if counter % 20 == 0:
            object.set_color(random.choice(colors))
            #x += 0.01
            #z += 0.01
            #object.set_position([x, y, z])

            object.add_force(np.array([0, 0, 0]), np.array([1, 0, 0]))

    pr.stop()  # Stop the simulation
    pr.shutdown()  # Close the application



if __name__ == "__main__":
    main()
