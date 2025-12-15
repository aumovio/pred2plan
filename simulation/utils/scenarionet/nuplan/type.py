# Scenarionet: https://github.com/metadriverse/scenarionet
# Published at August 27th 2023 under Apache 2.0
# All Rights Reserved


import logging

from metadrive.type import MetaDriveType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from nuplan.common.actor_state.tracked_objects_types import TrackedObjectType
    from nuplan.common.maps.maps_datatypes import TrafficLightStatusType

    NuPlanEgoType = TrackedObjectType.EGO
except ImportError as e:
    logger.warning("Can not import nuplan-devkit: {}".format(e))


def get_traffic_obj_type(nuplan_type):
    # we throw out traffic cones and barriers at this point because they are no agents that actually move but rather map features
    if nuplan_type == TrackedObjectType.VEHICLE:
        return MetaDriveType.VEHICLE
    # elif nuplan_type == TrackedObjectType.TRAFFIC_CONE:
    #     return MetaDriveType.TRAFFIC_CONE
    elif nuplan_type == TrackedObjectType.PEDESTRIAN:
        return MetaDriveType.PEDESTRIAN
    elif nuplan_type == TrackedObjectType.BICYCLE:
        return MetaDriveType.CYCLIST
    # elif nuplan_type == TrackedObjectType.BARRIER:
    #     return MetaDriveType.TRAFFIC_BARRIER
    elif nuplan_type == TrackedObjectType.EGO:
        raise ValueError("Ego should not be in detected results")
    else:
        return None


def set_light_status(status):
    if status == TrafficLightStatusType.GREEN:
        return MetaDriveType.LIGHT_GREEN
    elif status == TrafficLightStatusType.RED:
        return MetaDriveType.LIGHT_RED
    elif status == TrafficLightStatusType.YELLOW:
        return MetaDriveType.LIGHT_YELLOW
    elif status == TrafficLightStatusType.UNKNOWN:
        return MetaDriveType.LIGHT_UNKNOWN
