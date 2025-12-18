from meditrack.pipeline.pathway_pipeline import process_single_image

def test_process_callable():
    assert callable(process_single_image)
