'''
Copyright (C) 2025 co-pace GmbH (a subsidiary of Continental AG).
Licensed under the BSD-3-Clause License.
@author: Yao Yue
'''

OBJECT_TYPES = ['vehicle', #0
                'pedestrian', #1
                'motorcyclist', #2
                'cyclist', #3
                'bus', #4
                'static', #5
                'background', #6
                'construction', #7
                'riderless_bicycle', #8
                'unknown' #9
               ] 

TRACK_CATEGORIES =['TRACK_FRAGMENT', 'UNSCORED_TRACK', 'SCORED_TRACK', 'FOCAL_TRACK']

POLYGON_TYPES = ['VEHICLE', 'BIKE', 'BUS', 'PEDESTRIAN']
POLYGON_IS_INTERSECTIONS = [True, False, None]
POINT_TYPES = ['DASH_SOLID_YELLOW', #0
               'DASH_SOLID_WHITE', #1
               'DASHED_WHITE', #2
               'DASHED_YELLOW', #3
               'DOUBLE_SOLID_YELLOW', #4 
               'DOUBLE_SOLID_WHITE', #5
               'DOUBLE_DASH_YELLOW', #6
               'DOUBLE_DASH_WHITE', #7
               'SOLID_YELLOW', #8
               'SOLID_WHITE', #9
               'SOLID_DASH_WHITE', #10 
               'SOLID_DASH_YELLOW', #11
               'SOLID_BLUE', # 12
               'NONE', #13
               'UNKNOWN', #14
               'CROSSWALK', #15
               'CENTERLINE'] #16