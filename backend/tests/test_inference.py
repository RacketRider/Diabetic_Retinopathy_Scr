import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.services.inference import predict, CLASS_NAMES

@pytest.fixture
def dummy_image():
    # 448x448 float32 image matching preprocessing output
    return np.random.rand(448, 448, 3).astype(np.float32)

@patch('app.services.inference.get_session')
def test_predict_format(mock_get_session, dummy_image):
    # Mock the ONNX session
    mock_session = MagicMock()
    
    # Mock inputs/outputs
    mock_input = MagicMock()
    mock_input.name = "input"
    mock_session.get_inputs.return_value = [mock_input]
    
    # Mock raw logits output (1 batch, 5 classes)
    # Logits: [Severe, No_DR, Mild, Proliferate_DR, Moderate]
    mock_session.run.return_value = [np.array([[1.0, 2.0, 0.5, -1.0, 5.0]], dtype=np.float32)]
    mock_get_session.return_value = mock_session

    result = predict(dummy_image, "dummy_model_path.onnx")

    assert "grading" in result
    assert "confidence" in result
    
    # Check grading structure
    assert "level" in result["grading"]
    assert "label" in result["grading"]
    assert "is_referable" in result["grading"]
    
    # With logits [1.0, 2.0, 0.5, -1.0, 5.0], "Moderate" (index 4) should be the highest
    assert result["grading"]["label"] == "Moderate"
    
    # Check confidence structure
    assert "class_probabilities" in result["confidence"]
    assert "referable_probability" in result["confidence"]
    assert "threshold_used" in result["confidence"]
    
    # Ensure all class names are in probabilities
    for name in CLASS_NAMES:
        assert name in result["confidence"]["class_probabilities"]
