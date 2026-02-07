import math
from math import acos
import time


def euclidean_distance_general(list1, list2):
    if len(list1) == len(list2):
        return sum((p - q) ** 2 for p, q in zip(list1, list2)) ** 0.5
    else:
        raise ValueError("Lists are not of the same length.")


def calculate_intersection_with_future_check(p1, v1, p2, v2):
    a1, b1 = v1[0], -v2[0]  # x coefficient
    a2, b2 = v1[1], -v2[1]  # y coefficient
    c1 = p2[0] - p1[0]  # x constant
    c2 = p2[1] - p1[1]  # y constant

    determinant = a1 * b2 - a2 * b1
    if determinant == 0:
        return False, None

    # Cramer's Rule
    t = (c1 * b2 - c2 * b1) / determinant
    if t >= 0:
        intersection_point = (p1[0] + v1[0] * t, p1[1] + v1[1] * t)
        return True, intersection_point
    else:
        return False, None


def distance_delta(p1, v1, a1, p2, v2, a2, time=1):
    distance_initial = euclidean_distance_general(p1, p2)

    p1_later = (p1[0] + v1[0] * time + 0.5 * a1[0] * time ** 2, p1[1] + v1[1] * time + 0.5 * a1[1] * time ** 2)
    p2_later = (p2[0] + v2[0] * time + 0.5 * a2[0] * time ** 2, p2[1] + v2[1] * time + 0.5 * a2[1] * time ** 2)

    distance_later = euclidean_distance_general(p1_later, p2_later)

    return distance_initial, (distance_later - distance_initial) / time


def angle_delta(v1, a1, v2, a2, time=1):
    v1_initial = v1
    v2_initial = v2
    norm_v1_initial = (v1_initial[0] ** 2 + v1_initial[1] ** 2) ** 0.5
    norm_v2_initial = (v2_initial[0] ** 2 + v2_initial[1] ** 2) ** 0.5
    if norm_v1_initial * norm_v2_initial == 0:
        cos_theta_initial = 0
    else:
        cos_theta_initial = (v1_initial[0] * v2_initial[0] + v1_initial[1] * v2_initial[1]) / (
                norm_v1_initial * norm_v2_initial)

    cos_theta_initial_clamped = max(min(cos_theta_initial, 1), -1)
    angle_initial = acos(cos_theta_initial_clamped)

    v1_later = (v1[0] + a1[0] * time, v1[1] + a1[1] * time)
    v2_later = (v2[0] + a2[0] * time, v2[1] + a2[1] * time)
    norm_v1_later = (v1_later[0] ** 2 + v1_later[1] ** 2) ** 0.5
    norm_v2_later = (v2_later[0] ** 2 + v2_later[1] ** 2) ** 0.5

    if norm_v1_later * norm_v2_later == 0:
        cos_theta_later = 0
    else:
        cos_theta_later = (v1_later[0] * v2_later[0] + v1_later[1] * v2_later[1]) / (
                norm_v1_later * norm_v2_later)

    cos_theta_later_clamped = max(min(cos_theta_later, 1), -1)
    angle_later = acos(cos_theta_later_clamped)

    return angle_initial, (angle_later - angle_initial) / time


def acceleration_scalar(v, a):
    return v[0] * a[0] + v[1] * a[1]


def calculate_2d_overlap_ratio_different_lengths(p1, p2, length1, length2, width1, width2, extend_multiplier):
    extended_length1 = length1 + length1 * extend_multiplier
    extended_length2 = length2 + length2 * extend_multiplier

    start1_x = p1[0] - length1 / 2
    end1_x = start1_x + extended_length1
    start2_x = p2[0] - length2 / 2
    end2_x = start2_x + extended_length2
    overlap_start_x = max(start1_x, start2_x)
    overlap_end_x = min(end1_x, end2_x)
    overlap_length_x = max(0, overlap_end_x - overlap_start_x)

    start1_y = p1[1] - width1 / 2
    end1_y = p1[1] + width1 / 2
    start2_y = p2[1] - width2 / 2
    end2_y = p2[1] + width2 / 2
    overlap_start_y = max(start1_y, start2_y)
    overlap_end_y = min(end1_y, end2_y)
    overlap_length_y = max(0, overlap_end_y - overlap_start_y)

    overlap_area = overlap_length_x * overlap_length_y

    extended_area1 = extended_length1 * width1
    extended_area2 = extended_length2 * width2
    min_extended_area = min(extended_area1, extended_area2)

    overlap_ratio = overlap_area / min_extended_area if min_extended_area > 0 else 0

    return overlap_ratio


def trend_of_overlap_area(p1, p2, length1, length2, width1, width2, extend_multiplier, v1_vector, v2_vector, a1_vector,
                          a2_vector, delta_t=1):
    """
    Calculates the trend of the overlapping area between two objects in 2D space with given velocity and acceleration vectors.

    :param p1, p2: Initial center positions of the two objects (tuples of (x, y)).
    :param length1, length2: Lengths of the two objects.
    :param width1, width2: Widths of the two objects.
    :param extend_multiplier: Multiplier for length extension.
    :param v1_vector, v2_vector: Initial velocity vectors of the two objects (tuples of (vx, vy)).
    :param a1_vector, a2_vector: Acceleration vectors of the two objects (tuples of (ax, ay)).
    :param delta_t: Time step for comparing the change.
    """

    # Function to update velocity and position based on acceleration vector
    def update_velocity_position(v_vector, a_vector, p, t):
        new_vx = v_vector[0] + a_vector[0] * t
        new_vy = v_vector[1] + a_vector[1] * t
        new_px = p[0] + v_vector[0] * t + 0.5 * a_vector[0] * t ** 2
        new_py = p[1] + v_vector[1] * t + 0.5 * a_vector[1] * t ** 2
        return (new_px, new_py), (new_vx, new_vy)

    # Update velocities and positions
    new_p1, new_v1 = update_velocity_position(v1_vector, a1_vector, p1, delta_t)
    new_p2, new_v2 = update_velocity_position(v2_vector, a2_vector, p2, delta_t)

    # Calculate overlap ratios at t=0 and t=delta_t
    initial_overlap_ratio = calculate_2d_overlap_ratio_different_lengths(p1, p2, length1, length2, width1, width2,
                                                                         extend_multiplier)
    new_overlap_ratio = calculate_2d_overlap_ratio_different_lengths(new_p1, new_p2, length1, length2, width1, width2,
                                                                     extend_multiplier)

    return initial_overlap_ratio, (new_overlap_ratio - initial_overlap_ratio) / delta_t


def vectors_same_direction(vector_a, vector_b, delta):
    dot_product = vector_a[0] * vector_b[0] + vector_a[1] * vector_b[1]

    norm_a = (vector_a[0] ** 2 + vector_a[1] ** 2) ** 0.5
    norm_b = (vector_b[0] ** 2 + vector_b[1] ** 2) ** 0.5
    if norm_a * norm_b == 0:
        return False

    cos_theta = dot_product / (norm_a * norm_b)
    return abs(cos_theta - 1) < delta


def inference_two_nodes(node1, node2, threshold_deltaTTC, threshold_maxTTC, radius, threshold_diffVp, threshold_Vp, threshold_Dp,
                        threshold_follow_v, threshold_dist_parallel, threshold_angle_parallel, threshold_angle_overtake,
                        threshold_Vc, threshold_stop, threshold_betweenV, thresholdEOA, disableVV=True):
    """
    : param node1: dict
    : param node2: dict
    : param threshold_deltaTTC
    : param threshold_maxTTC
    : param radius : first criteria to justify whether connected
    : param threshold_Vp : max speed for a (interactive) pedestrian
    : param threshold_Dp : max threshold for two pedestrians' distance to have interaction
    : param threshold_follow : max threshold for the difference of two objects' speed following each other
    : param threshold_dist_parallel : max threshold for the distance of two objects moving parallel
    : param threshold_angle_parallel : max threshold for the difference of two objects' speed angle moving parallel
    : param threshold_angle_overtake : max threshold for the difference of two (motor)objects' speed angle to overtake
    : param threshold_Vc : max speed for a (interactive) cycle
    : param threshold_stop : max speed for a stopped object
    : return isConnected: bool
    : return possibleInteractions: list
    0: P-P, 1: V-P, 2: P-V, 3: V-V, 4: V-C, 5: C-V, 6: C-P, 7: P-C, 8: C-C
    """
    start_time = time.time()
    isConnected = False
    possibleInteractions = []

    cPointFlag, cPoint = calculate_intersection_with_future_check(node1['pos'], node1['velocity'], node2['pos'],
                                                           node2['velocity'])
    # assert not cPointFlag
    velo1 = euclidean_distance_general(node1['velocity'], [0, 0])
    velo2 = euclidean_distance_general(node2['velocity'], [0, 0])
    if cPointFlag:
        ttc1 = euclidean_distance_general(node1['pos'], cPoint) / velo1
        ttc2 = euclidean_distance_general(node2['pos'], cPoint) / velo2
        deltaTTC = abs(ttc1 - ttc2)
        maxTTC = max(ttc1, ttc2)
        isConnected = deltaTTC <= threshold_deltaTTC and maxTTC <= threshold_maxTTC

    len1 = max(node1['size'][0], node1['size'][1])
    width1 = min(node1['size'][0], node1['size'][1])
    len2 = max(node2['size'][0], node2['size'][1])
    width2 = min(node2['size'][0], node2['size'][1])
    EOA, deltaEOA = trend_of_overlap_area(node1['pos'], node2['pos'], len1, len2, width1, width2, 3, node1['velocity'],
                                          node2['velocity'], node1['acc'], node2['acc'])
    if not isConnected:
        '''
        if euclidean_distance_general(node1['pos'], node2['pos']) > radius:
            return False, None
        else:
            isConnected = True
        '''
        if EOA > 0:
            isConnected = True
        else:
            return False, None, 0
    # prepare metrics
    if cPointFlag:
        ID1, deltaID1 = distance_delta(node1['pos'], node1['velocity'], node1['acc'], cPoint, [0, 0], [0, 0])
        ID2, deltaID2 = distance_delta(node2['pos'], node2['velocity'], node2['acc'], cPoint, [0, 0], [0, 0])

    distance, deltaD = distance_delta(node1['pos'], node1['velocity'], node1['acc'], node2['pos'], node2['velocity'],
                                      node2['acc'])
    angle, deltaAngle = angle_delta(node1['velocity'], node1['acc'], node2['velocity'], node2['acc'])
    acc1 = acceleration_scalar(node1['velocity'], node1['acc'])
    acc2 = acceleration_scalar(node2['velocity'], node2['acc'])

    # For P-P
    if velo1 <= threshold_Vp and velo2 <= threshold_Vp:
        if distance <= threshold_Dp and abs(velo1 - velo2) <= threshold_diffVp:
            possibleInteractions.append(0)
    # For V-P
    if distance >= threshold_betweenV:
        if acc1 <= -threshold_stop or velo1 <= threshold_stop and velo2 <= threshold_Vp:
            if cPointFlag and deltaID1 < 0 and deltaID2 < 0:  # V-P PreYield
                possibleInteractions.append(1)
            elif not cPointFlag and deltaD > 0:  # V-P PostYield
                possibleInteractions.append(1)
        if deltaD > 0 and acc1 >= threshold_stop and velo2 <= threshold_Vp:  # V-P AfterYield
            possibleInteractions.append(1)

    # For P-V
    if distance >= threshold_betweenV:
        if acc1 <= -threshold_stop or velo1 <= threshold_stop:
            if cPointFlag and deltaID1 < 0 and deltaID2 < 0:  # P-V PreYield
                possibleInteractions.append(2)
            elif not cPointFlag and deltaD > 0:  # P-V PostYield
                possibleInteractions.append(2)
        if deltaD > 0 and acc1 >= threshold_stop:  # P-V AfterYield
            possibleInteractions.append(2)

    # For V-V
    if not disableVV and deltaD >= threshold_betweenV * 2:
        if deltaD > 0:
            if (acc1 <= threshold_stop or velo1 <= threshold_stop) or (acc2 <= threshold_stop or velo2 <= threshold_stop):
                if cPointFlag:
                    if deltaID1 < 0 and deltaID2 < 0:  # Yielding
                        possibleInteractions.append(3)
                    elif EOA < thresholdEOA:  # PostYield
                        possibleInteractions.append(3)
        if acc1 >= -threshold_stop or acc2 >= -threshold_stop:  # AfterYield
            possibleInteractions.append(3)
        if abs(velo1 - velo2) < threshold_follow_v and EOA >= thresholdEOA:  # Following
            possibleInteractions.append(3)
        if distance < threshold_dist_parallel and (deltaAngle < threshold_angle_parallel or abs(
                math.pi - deltaAngle) < threshold_angle_parallel):  # Move in parallel
            possibleInteractions.append(3)
        if (acc1 >= -threshold_stop and velo1 > velo2 and deltaEOA < 0 and deltaAngle < threshold_angle_overtake) or (
                acc2 >= -threshold_stop and velo2 > velo1 and deltaEOA < 0 and deltaAngle < threshold_angle_overtake):  # PreOvertake
            possibleInteractions.append(3)
        if (velo1 > velo2 or velo2 > velo1) and deltaEOA > 0 and deltaAngle < threshold_angle_overtake:  # PostOvertake
            possibleInteractions.append(3)

    # For V-C and C-V
    if distance >= threshold_betweenV:
        if acc1 <= -threshold_stop or velo1 <= threshold_stop or acc2 <= -threshold_stop or velo2 <= threshold_stop:
            if cPointFlag and deltaID1 < 0 and deltaID2 < 0:  # Yielding
                possibleInteractions.append(4)
                possibleInteractions.append(5)
            elif not cPointFlag and EOA < thresholdEOA:  # PostYield
                possibleInteractions.append(4)
                possibleInteractions.append(5)
        if deltaD > 0 and acc1 >= threshold_stop or acc2 >= threshold_stop:  # AfterYield
            possibleInteractions.append(4)
            possibleInteractions.append(5)
        if abs(velo1 - velo2) < threshold_follow_v and EOA >= thresholdEOA:  # Following
            possibleInteractions.append(4)
            possibleInteractions.append(5)
        if distance < threshold_dist_parallel and (angle < threshold_angle_parallel or abs(
                math.pi - deltaAngle) < threshold_angle_parallel):  # Move in parallel
            possibleInteractions.append(4)
            possibleInteractions.append(5)
        if (acc1 >= threshold_stop and velo1 > velo2 and deltaEOA < 0 and angle < threshold_angle_overtake) or (
                acc2 >= threshold_stop and velo2 > velo1 and deltaEOA < 0 and angle < threshold_angle_overtake):  # PreOvertake
            possibleInteractions.append(4)
            possibleInteractions.append(5)
        if (velo1 > velo2 or velo2 > velo1) and deltaEOA > 0 and angle < threshold_angle_overtake:  # PostOvertake
            possibleInteractions.append(4)
            possibleInteractions.append(5)

    # For C-P
    if (acc1 <= -threshold_stop or velo1 <= threshold_stop) and velo2 < threshold_Vp:
        if cPointFlag and deltaID1 < 0 and deltaID2 < 0:  # PreYield
            possibleInteractions.append(6)
        elif not cPointFlag and deltaD > 0:  # PostYield
            possibleInteractions.append(6)
    if deltaD > 0 and acc1 >= threshold_stop and velo2 < threshold_Vp:  # AfterYield
        possibleInteractions.append(6)

    # For P-C
    if (acc1 <= -threshold_stop or velo1 <= threshold_stop) and velo2 < threshold_Vc:
        if cPointFlag and deltaID1 < 0 and deltaID2 < 0:  # PreYield
            possibleInteractions.append(7)
        elif not cPointFlag and deltaD > 0:  # PostYield
            possibleInteractions.append(7)
    if deltaD > 0 and acc1 >= threshold_stop and velo2 < threshold_Vc:  # AfterYield
        possibleInteractions.append(7)

    # For C-C
    if velo1 < threshold_Vc and velo2 < threshold_Vc:
        if acc1 >= threshold_stop or acc2 >= threshold_stop and deltaAngle < 0:  # PotentialTurning
            possibleInteractions.append(8)
        if abs(velo1 - velo2) < threshold_follow_v and EOA >= thresholdEOA:  # Following
            possibleInteractions.append(8)
        if distance < threshold_dist_parallel and (angle < threshold_angle_parallel or abs(
                math.pi - angle) < threshold_angle_parallel):  # Move in parallel
            possibleInteractions.append(8)
        if ((acc1 >= threshold_stop or velo1 > velo2) and not vectors_same_direction(node1['velocity'], node1['acc'],
                                                                                  threshold_stop)) or (
                (acc2 >= threshold_stop or velo2 > velo1) and not vectors_same_direction(node2['velocity'], node2['acc'],
                                                                                      threshold_stop)) and deltaEOA < 0 or EOA < threshold_stop:  # PreOvertake
            possibleInteractions.append(8)
        if (velo1 > velo2 or velo2 > velo1) and deltaEOA > 0 and angle < threshold_angle_overtake:  # PostOvertake
            possibleInteractions.append(8)

    possibleInteractions = list(set(possibleInteractions))
    end_time = time.time()
    return isConnected, possibleInteractions, end_time-start_time
