import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.services.inference import predict, CLASS_NAMES
import app.services.inference as inference_mod

@pytest.fixture
def dummy_image():
    # uint8 RGB image
    return np.random.randint(0, 255, (448, 448, 3), dtype=np.uint8)

def test_predict_format(dummy_image):
    # Mock the ONNX session
    mock_session = MagicMock()
    
    # Mock inputs/outputs
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    
    # Mock raw logits output (1 batch, 5 classes: Mild, Moderate, No_DR, Proliferate_DR, Severe)
    # Logit highest at index 1 ("Moderate")
    mock_session.run.return_value = [np.array([[1.0, 5.0, 0.5, -1.0, 2.0]], dtype=np.float32)]

    with patch.object(inference_mod, "_session", mock_session):
        result = predict(dummy_image)

    assert "grading" in result
    assert "confidence" in result
    
    # Check grading structure
    assert "level" in result["grading"]
    assert "label" in result["grading"]
    assert "is_referable" in result["grading"]
    
    # With logits [1.0, 5.0, 0.5, -1.0, 2.0], "Moderate" (index 1) should be the highest
    assert result["grading"]["label"] == "Moderate"
    
    # Check confidence structure
    assert "class_probabilities" in result["confidence"]
    assert "referable_probability" in result["confidence"]
    assert "threshold_used" in result["confidence"]
    
    # Ensure all class names are in probabilities
    for name in CLASS_NAMES:
        assert name in result["confidence"]["class_probabilities"]
