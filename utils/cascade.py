# coding: utf-8

"""Functions for reading and writing CASCADE file format
--------------------------------------------------------

This file contains all functions related to the CASCADE file format reading and writing from the competition.
"""
import os
import re
import xml.etree.ElementTree as ET

import cv2
import numpy as np

from typing import List, Tuple, Dict

def get_mask_slice(dcm_image_dir: str, artery: str, case: str, slice_idx: int, H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """Extract lumen and wall masks from a QVS file for a specific slice of a given artery in a certain case.

    This function extracts lumen and wall masks from a QVS file for a specific slice of a given artery in a certain case.
    The lumen and wall masks are created using the "Lumen" and "Outer Wall" contours, respectively. The wall mask is computed
    by subtracting the lumen mask from a binary mask of the entire image.

    Args:
        dcm_image_dir (str): The directory where the DICOM images are stored.
        artery (str): The name of the artery (e.g., 'LAD', 'RCA', etc.).
        case (str): The case name or ID.
        slice_idx (int): The index of the slice to extract the masks from.
        H (int): The height of the image.
        W (int): The width of the image.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing two 3D numpy arrays representing the lumen and wall masks, respectively.

    Notes:
        The lumen and wall masks are extracted from the QVS file corresponding to the given case and artery.

    See Also:
        get_qvs_fname : A function to generate the QVS file name from the QVJ file name.
        get_contour : A function to extract a contour from a QVS file for a specific slice.
        contour_to_mask : A function to convert a contour to a binary mask.
    """
    qvj_file = dcm_image_dir + '/' + case + artery + ".QVJ"
    qvs_file = os.path.join(dcm_image_dir, get_qvs_fname(qvj_file))
    lumen_cont = get_contour(qvs_file, slice_idx, "Lumen", H, W)
    wall_cont = get_contour(qvs_file, slice_idx, "Outer Wall", H, W)
    lumen_mask = np.expand_dims(contour_to_mask(lumen_cont, H, W), 0)  # 1, H, W
    wall_mask = np.expand_dims(contour_to_mask(wall_cont, H, W), 0) - lumen_mask
    return lumen_mask, wall_mask

def fix_nearest_athero(annotated_slices: List[int], side_status: Dict[int, str]) -> Dict[int, str]:
    """Fix missing athero label by using nearest annotated label.

    This function checks for missing athero labels in the `side_status` dictionary and fixes them
    by using the nearest available annotated label. The nearest label is determined based on the
    Euclidean distance between the missing and available labels.

    Args:
        annotated_slices (List[int]): A list of slice indices that have been annotated.
        side_status (Dict[int, str]): A dictionary containing slice indices as keys and their
                                     corresponding athero labels as values.

    Returns:
        Dict[int, str]: A dictionary with the same keys as `side_status`, but with fixed athero labels
                       for any missing values.
    """
    labeled = list(side_status.keys())
    for anno_id in annotated_slices:
        if not anno_id in side_status.keys():
            closest = min(labeled, key=lambda x: abs(x - anno_id))
            side_status[anno_id] = side_status[closest]
    return side_status


def list_contour_slices(dcm_image_dir, artery, case):
    """
    :param qvs_root: xml root
    :return: slices with annotations
    """
    avail_slices = []
    qvj_file = dcm_image_dir + '/' + case + artery + ".QVJ"
    if os.path.exists(qvj_file):
        qvs_file = os.path.join(dcm_image_dir, get_qvs_fname(qvj_file))
        qvs_root = ET.parse(qvs_file).getroot()
        image_elements = qvs_root.findall("QVAS_Image")
        for slice_id, element in enumerate(image_elements):
            conts = element.findall("QVAS_Contour")
            if len(conts) > 0:
                avail_slices.append(slice_id)
    return avail_slices

### care2 codes
def get_mask_slice_careII(dcm_image_dir, artery, case, slice_idx, H, W):
    qvs_path = dcm_image_dir + "/CASCADE-ICA" + artery + "/E" + case.split("_")[1] + "S101_L.QVS"
    lumen_contour, wall_contour = get_contour_careII(qvs_path, slice_idx)
    lumen_mask = np.expand_dims(contour_to_mask_careII(lumen_contour, (H, W)), 0)  # 1, H, W
    wall_mask = np.expand_dims(contour_to_mask_careII(wall_contour, (H, W)), 0) - lumen_mask
    wall_mask[wall_mask < 0] = 0
    return lumen_mask, wall_mask

def list_available_mask_slices_careII(dcm_image_dir, artery, case):
    """Return available_slices index from qvs files."""
    qvs_path = dcm_image_dir + "/CASCADE-ICA" + artery + "/E" + case.split("_")[1] + "S101_L.QVS"
    available_slices = []
    if os.path.exists(qvs_path):
        qvs_root = ET.parse(qvs_path).getroot()
        qvas_imgs = qvs_root.findall("QVAS_Image")
        for slice_idx, img in enumerate(qvas_imgs):
            contours = img.findall("QVAS_Contour")
            if len(contours) > 0:
                available_slices.append(slice_idx)
    return available_slices


def contour_to_mask_careII(contour, mask_size=(720, 100)):
    """Use opencv to get filled mask from coords."""
    mask = np.zeros(mask_size, dtype=np.uint8)
    contour_resized = []
    for point in contour:
        w = round(point[0] * mask_size[0])
        h = round(point[1] * mask_size[0] - (mask_size[0] - mask_size[1]) / 2)
        contour_resized.append([h, w])
    if contour_resized:
        pts = np.array(contour_resized)
        mask = cv2.fillConvexPoly(mask, pts, color=1).astype(np.float32)
    return mask


def get_contour_careII(qvs_path, slice_idx):
    """Get contours coords from qvs files."""
    qvs_root = ET.parse(qvs_path).getroot()
    qvas_imgs = qvs_root.findall("QVAS_Image")
    if slice_idx > len(qvas_imgs):
        print("there is none", slice_idx)
        return
    assert int(qvas_imgs[slice_idx].get("ImageName").split("I")[-1]) == slice_idx + 1
    contours = qvas_imgs[slice_idx].findall("QVAS_Contour")
    lumen_contour, outerwall_contour = [], []
    for contour in contours:
        if contour.find("ContourType").text == "Lumen":
            lumen_points = contour.find("Contour_Point").findall("Point")
            for point in lumen_points:
                x = float(point.get("x")) / 512
                y = float(point.get("y")) / 512
                lumen_contour.append([x, y])
        if contour.find("ContourType").text == "Outer Wall":
            outer_wall_points = contour.find("Contour_Point").findall("Point")
            for point in outer_wall_points:
                x = float(point.get("x")) / 512
                y = float(point.get("y")) / 512
                outerwall_contour.append([x, y])
    return lumen_contour, outerwall_contour


### Official COSMOS codes

def DSC(label_img, pred_img, smooth=0.1):
    """Dice Coefficiency. ndarray input"""
    A = label_img > 0.5 * np.max(label_img)
    B = pred_img > 0.5 * np.max(pred_img)
    return 2 * (np.sum(A[A == B]) + smooth) / (np.sum(A) + np.sum(B) + smooth)


def get_qvs_fname(qvj_path):
    qvs_element = ET.parse(qvj_path).getroot().find("QVAS_Loaded_Series_List").find("QVASSeriesFileName").text
    return qvs_element


def get_contour(qvs_path, slice_id, cont_type, height, width) -> np.ndarray:
    qvsroot = ET.parse(qvs_path).getroot()
    qvas_img = qvsroot.findall("QVAS_Image")
    conts = qvas_img[slice_id].findall("QVAS_Contour")
    pts = None
    for cont_id, cont in enumerate(conts):
        if cont.find("ContourType").text == cont_type:
            pts = cont.find("Contour_Point").findall("Point")
            break
    if pts is not None:
        contours = []
        for p in pts:
            contx = float(p.get("x")) / 512 * width
            conty = float(p.get("y")) / 512 * height
            # if current pt is different from last pt, add to contours
            if len(contours) == 0 or contours[-1][0] != contx or contours[-1][1] != conty:
                contours.append([contx, conty])
        return np.array(contours)
    return None


def contour_to_mask(contour, height, width) -> np.ndarray:
    """Use opencv to get filled mask from coords. returns `(H, W)`"""
    mask = np.zeros((width, height), dtype=np.uint8)
    contour_resized = []
    for point in contour:
        w = round(point[0])
        h = round(point[1])
        contour_resized.append([h, w])
    pts = np.array(contour_resized)
    mask = cv2.fillConvexPoly(mask, pts, color=1)
    return mask


def mask_to_polygon(mask):
    """2D mask to polygon, for generate prediction qvs files"""
    mask = np.uint8(mask > 0) * 255    
    # Firstly get all contours then return the biggest one
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    area_max = 0
    if len(contours) == 0:
        return []
    cnt_max = contours[0]
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > area_max:
            area_max = area
            cnt_max = cnt
    output = np.squeeze(cnt_max, axis=1)
    return output


def removeAllContours(qvsroot, seriesnum="101"):
    r"""remove contours in QVS files to generate new ones."""
    # find qvas_image
    qvasimgs = qvsroot.findall("QVAS_Image")
    # clear all previous contours
    for imgi in range(len(qvasimgs)):
        cts = qvasimgs[imgi].findall("QVAS_Contour")
        for ctsi in cts:
            qvasimgs[imgi].remove(ctsi)


def write_into_CASCADE(case_dir: str, preds_array: np.ndarray, target_art_dir: str, model_dir=None):
    raw_size = preds_array.shape[1:]
    case_i = os.path.split(case_dir)[-1]
    # bst = joblib.load(model_dir)
    for idx, vessel in enumerate(["L", "R"]):  # TODO: Wrong order?
        qvj_name = case_i + vessel + ".QVJ"
        qvj_file = os.path.join(case_dir, qvj_name)
        if os.path.exists(qvj_file):
            qvs_name = get_qvs_fname(qvj_file)
            tree_qvs = ET.parse(os.path.join(case_dir, qvs_name))
            QVS_root = tree_qvs.getroot()
            removeAllContours(QVS_root)
            tree_qvj = ET.parse(qvj_file)
            for slice_idx in range(raw_size[-1]):
                lumen_slice = preds_array[idx, :, :, slice_idx]
                wall_slice = preds_array[idx + 2, :, :, slice_idx]
                if lumen_slice.sum() > 0.0 and wall_slice.sum() > 0.0:
                    create_contour(QVS_root, lumen_slice, raw_size, "Lumen", slice_idx)
                    create_contour(QVS_root, wall_slice, raw_size, "Outer Wall", slice_idx, athero_status=0)
            os.makedirs(target_art_dir, exist_ok=True)
            tree_qvs.write(os.path.join(target_art_dir, qvs_name))
            tree_qvj.write(os.path.join(target_art_dir, qvj_name))


def create_contour(root, slice, raw_size, cont_type, tslicei, contour_conf=None, athero_status=None):
    qvasimgs = root.findall("QVAS_Image")
    fdqvasimg = -1
    for slicei in range(len(qvasimgs)):
        qvas_idx = int(re.findall("\d+", qvasimgs[slicei].get("ImageName"))[-1])
        if qvas_idx == tslicei:
            fdqvasimg = slicei
            break
    if fdqvasimg == -1:
        print("QVAS_IMAGE not found")
        return
    # clear previous contours if there are
    cts = qvasimgs[tslicei].findall("QVAS_Contour")
    for ctsi in cts:
        ctype = ctsi.findall("ContourType")
        for ctr in ctype:
            if ctr.text == cont_type:
                qvasimgs[fdqvasimg].remove(ctsi)

    if cont_type == "Outer Wall":
        ct = "Outer Wall"
        ctcl = "16776960"
    elif cont_type == "Lumen":
        ct = "Lumen"
        ctcl = "255"

    QVAS_Contour = ET.SubElement(qvasimgs[fdqvasimg], "QVAS_Contour")
    contour = mask_to_polygon(slice.T)
    contour_point = ET.SubElement(QVAS_Contour, "Contour_Point")
    for cord in contour:
        x, y = cord
        Point = ET.SubElement(contour_point, "Point")

        Point.set("x", "%.5f" % (x / raw_size[0] * 512))
        Point.set("y", "%.5f" % (y / raw_size[1] * 512))

    ContourType = ET.SubElement(QVAS_Contour, "ContourType")
    ContourType.text = ct

    ContourColor = ET.SubElement(QVAS_Contour, "ContourColor")
    ContourColor.text = ctcl

    # -------------Ignore below, only for software loading purposes ----------------
    ContourOpenStatus = ET.SubElement(QVAS_Contour, "ContourOpenStatus")
    ContourOpenStatus.text = "1"
    ContourPCConic = ET.SubElement(QVAS_Contour, "ContourPCConic")
    ContourPCConic.text = "0.5"
    ContourSmooth = ET.SubElement(QVAS_Contour, "ContourSmooth")
    ContourSmooth.text = "60"
    Snake_Point = ET.SubElement(QVAS_Contour, "Snake_Point")
    # snake point, fake fill
    for snakei in range(6):
        conti = len(contour) // 6 * snakei
        Point = ET.SubElement(Snake_Point, "Point")
        # if conti == 0:
        #     continue
        Point.set("x", "%.5f" % (contour[conti][0] / raw_size[0] * 512))
        Point.set("y", "%.5f" % (contour[conti][1] / raw_size[0] * 512))

    ContourComments = ET.SubElement(QVAS_Contour, "ContourComments")
    if athero_status is not None:
        ContourComments.text = str(athero_status)
    # -------------Ignore above, only for software loading purposes ----------------

    ContourConf = ET.SubElement(QVAS_Contour, "ContourConf")
    if contour_conf is not None:
        LumenConsistency = ET.SubElement(ContourConf, "LumenConsistency")
        LumenConsistency.text = "%.5f" % contour_conf[0]
        WallConsistency = ET.SubElement(ContourConf, "WallConsistency")
        WallConsistency.text = "%.5f" % contour_conf[1]


def get_bir_slice(qvjroot):
    if qvjroot.find("QVAS_System_Info").find("BifurcationLocation"):
        bif_slice = int(
            qvjroot.find("QVAS_System_Info").find("BifurcationLocation").find("BifurcationImageIndex").get("ImageIndex")
        )
        return bif_slice
    else:
        return -1

def get_loc_prop(qvj_root, bif_slice):
    loc_prop = qvj_root.find('Location_Property')
    results = {}
    for loc in loc_prop.iter('Location'):
        loc_ind = int(loc.get('Index')) + bif_slice
        image_quality = int(loc.find('IQ').text)
        # only slices with Image Quality (IQ) > 1 were labeled 
        # AHAStatus: 1: Normal; > 1 : Atherosclerotic
        AHA_status = float(loc.find('AHAStatus').text)
        if image_quality>1 and AHA_status == 1:
            # print(f"{loc_ind}: Normal")
            results[loc_ind] = 0
        elif image_quality>1 and AHA_status >1:
            results[loc_ind] = 1
            # print(f"{loc_ind}: Atherosclerotic")
    return results

def get_athero_status(input_dir, case_file):
    case_dir = os.path.join(input_dir, case_file)
    results_dict = {}
    for side in ['L', 'R']:
        qvj_file = os.path.join(case_dir, case_file + side + '.QVJ')
        if os.path.exists(qvj_file):
            qvj_root = ET.parse(qvj_file).getroot()
            bif_slice = get_bir_slice(qvj_root)
            side_results = get_loc_prop(qvj_root, bif_slice)
            results_dict[side] = side_results
    return results_dict