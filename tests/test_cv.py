from meditrack.cv.preprocessing import load_and_preprocess_image

def test_import_preprocess():
    assert callable(load_and_preprocess_image)
