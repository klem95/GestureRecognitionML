class Skeleton:
    # regions
    HEAD_REGION = 0
    UPPER_RIGHT_REGION = 1
    UPPER_LEFT_REGION = 2
    LOWER_RIGHT_REGION = 3
    LOWER_LEFT_REGION = 4

    # joints
    PELVIS = 0
    SPINE_NAVAL = 1
    SPINE_CHEST = 2
    NECK = 3
    CLAVICLE_LEFT = 4
    SHOULDER_LEFT = 5
    ELBOW_LEFT = 6
    WRIST_LEFT = 7
    HAND_LEFT = 8
    HANDTIP_LEFT = 9
    THUMB_LEFT = 10
    CLAVICLE_RIGHT = 11
    SHOULDER_RIGHT = 12
    ELBOW_RIGHT = 13
    WRIST_RIGHT = 14
    HAND_RIGHT = 15
    HANDTIP_RIGHT = 16
    THUMB_RIGHT = 17
    HIP_LEFT = 18
    KNEE_LEFT = 19
    ANKLE_LEFT = 20
    FOOT_LEFT = 21
    HIP_RIGHT = 22
    KNEE_RIGHT = 23
    ANKLE_RIGHT = 24
    FOOT_RIGHT = 25
    HEAD = 26
    NOSE = 27
    EYE_LEFT = 28
    EAR_LEFT = 29
    EYE_RIGHT = 30
    EAR_RIGHT = 31

    region_look_up = {
        HEAD_REGION: [SPINE_CHEST, CLAVICLE_RIGHT, NECK, CLAVICLE_LEFT, HEAD, NOSE, EYE_LEFT, EAR_LEFT, EYE_RIGHT,
                      EAR_RIGHT],
        UPPER_RIGHT_REGION: [HIP_RIGHT, PELVIS, SPINE_NAVAL, SPINE_CHEST, NECK, CLAVICLE_RIGHT, SHOULDER_RIGHT,
                             ELBOW_RIGHT, WRIST_RIGHT, HAND_RIGHT, HANDTIP_RIGHT, THUMB_RIGHT],
        UPPER_LEFT_REGION: [HIP_LEFT, PELVIS, SPINE_NAVAL, SPINE_CHEST, NECK, CLAVICLE_LEFT, SHOULDER_LEFT, ELBOW_LEFT,
                            WRIST_LEFT, HAND_LEFT, HANDTIP_LEFT, THUMB_LEFT],
        LOWER_RIGHT_REGION: [PELVIS, HIP_RIGHT, KNEE_RIGHT, ANKLE_RIGHT, FOOT_RIGHT],
        LOWER_LEFT_REGION: [PELVIS, HIP_LEFT, KNEE_LEFT, ANKLE_LEFT, FOOT_LEFT]
    }
