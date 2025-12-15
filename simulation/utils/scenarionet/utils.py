# Scenarionet: https://github.com/metadriverse/scenarionet
# Published at August 27th 2023 under Apache 2.0
# All Rights Reserved

import logging
import math
import numpy as np
logger = logging.getLogger(__file__)


def single_worker_preprocess(x, worker_index):
    """
    All scenarios passed to write_to_directory_single_worker will be preprocessed. The input is expected to be a list.
    The output should be a list too. The element in the second list will be processed by convertors. By default, you
    don't need to provide this processor. We override it for waymo convertor to release the memory in time.
    :param x: input
    :param worker_index: worker_index, useful for logging
    :return: input
    """
    return x


def nuplan_to_metadrive_vector(vector, nuplan_center=(0, 0)):
    "All vec in nuplan should be centered in (0,0) to avoid numerical explosion"
    vector = np.array(vector)
    vector -= np.asarray(nuplan_center)
    return vector


def compute_angular_velocity(initial_heading, final_heading, dt):
    """
    Calculate the angular velocity between two headings given in radians.

    Parameters:
    initial_heading (float): The initial heading in radians.
    final_heading (float): The final heading in radians.
    dt (float): The time interval between the two headings in seconds.

    Returns:
    float: The angular velocity in radians per second.
    """

    # Calculate the difference in headings
    delta_heading = final_heading - initial_heading

    # Adjust the delta_heading to be in the range (-π, π]
    delta_heading = (delta_heading + math.pi) % (2 * math.pi) - math.pi

    # Compute the angular velocity
    angular_vel = delta_heading / dt

    return angular_vel


def mph_to_kmh(speed_in_mph: float):
    speed_in_kmh = speed_in_mph * 1.609344
    return speed_in_kmh

