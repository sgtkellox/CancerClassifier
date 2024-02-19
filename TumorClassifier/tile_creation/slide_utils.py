def getDiagFromSlide(slide):
    gt_class_name = ""
    if slide.split("-")[0].startswith("A"):
        gt_class_name = "A"
    elif slide.split("-")[0].startswith("G"):
        gt_class_name = "GBM"
    elif slide.split("-")[0].startswith("O"):
        gt_class_name = "O"
    elif slide.split("-")[0].startswith("EPN"):
        gt_class_name = "EPN"
    else:
        gt_class_name = slide.split("-")[0]
    return gt_class_name
    

