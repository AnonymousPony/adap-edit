import lpips


def LPIPS_Similarity(img_0, img_1, name):
    assert name in ("alex", "vgg")  # fmt: skip
    model = lpips.LPIPS(net=name)
    return model(img_0, img_1)