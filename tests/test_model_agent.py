"""
Tests for the ModelAgent class.
"""

import pytest
from agents.model_agent import ModelAgent


@pytest.fixture
def agent():
    """Create a ModelAgent instance for testing."""
    return ModelAgent()


class TestModelAgentInit:
    """Tests for ModelAgent initialization."""

    def test_init_without_model_path(self):
        """Test initialization without model path."""
        agent = ModelAgent()
        assert agent.model_path is None
        assert agent.model is None

    def test_init_with_model_path(self):
        """Test initialization with model path."""
        agent = ModelAgent(model_path="/path/to/model.pkl")
        assert agent.model_path == "/path/to/model.pkl"


class TestWaterQualityPrediction:
    """Tests for water quality prediction."""

    def test_predict_with_default_values(self, agent):
        """Test prediction with default values."""
        result = agent.predict_water_quality({})
        assert "quality_score" in result
        assert "quality_label" in result
        assert "parameters" in result
        assert "recommendations" in result

    def test_predict_with_optimal_values(self, agent):
        """Test prediction with optimal water quality values."""
        result = agent.predict_water_quality({
            "ph": 7.5,
            "turbidity": 0.5,
            "temperature": 20.0,
            "dissolved_oxygen": 10.0,
        })
        assert result["quality_score"] >= 80
        assert result["quality_label"] == "Excellent"

    def test_predict_with_poor_values(self, agent):
        """Test prediction with poor water quality values."""
        result = agent.predict_water_quality({
            "ph": 5.0,
            "turbidity": 10.0,
            "temperature": 35.0,
            "dissolved_oxygen": 2.0,
        })
        assert result["quality_score"] < 60
        assert result["quality_label"] in ["Fair", "Poor", "Critical"]

    def test_recommendations_for_low_ph(self, agent):
        """Test that low pH generates appropriate recommendation."""
        result = agent.predict_water_quality({"ph": 5.0})
        assert any("pH is too low" in rec for rec in result["recommendations"])

    def test_recommendations_for_high_ph(self, agent):
        """Test that high pH generates appropriate recommendation."""
        result = agent.predict_water_quality({"ph": 10.0})
        assert any("pH is too high" in rec for rec in result["recommendations"])

    def test_recommendations_for_high_turbidity(self, agent):
        """Test that high turbidity generates filtration recommendation."""
        result = agent.predict_water_quality({"turbidity": 10.0})
        assert any("filtration" in rec.lower() for rec in result["recommendations"])


class TestQualityLabels:
    """Tests for quality label generation."""

    def test_excellent_label(self, agent):
        """Test that scores >= 80 return Excellent."""
        assert agent._get_quality_label(80) == "Excellent"
        assert agent._get_quality_label(100) == "Excellent"

    def test_good_label(self, agent):
        """Test that scores 60-79 return Good."""
        assert agent._get_quality_label(60) == "Good"
        assert agent._get_quality_label(79) == "Good"

    def test_fair_label(self, agent):
        """Test that scores 40-59 return Fair."""
        assert agent._get_quality_label(40) == "Fair"
        assert agent._get_quality_label(59) == "Fair"

    def test_poor_label(self, agent):
        """Test that scores 20-39 return Poor."""
        assert agent._get_quality_label(20) == "Poor"
        assert agent._get_quality_label(39) == "Poor"

    def test_critical_label(self, agent):
        """Test that scores < 20 return Critical."""
        assert agent._get_quality_label(0) == "Critical"
        assert agent._get_quality_label(19) == "Critical"


class TestTrendAnalysis:
    """Tests for trend analysis."""

    def test_analyze_empty_data(self, agent):
        """Test trend analysis with empty data."""
        result = agent.analyze_trends([])
        assert "error" in result

    def test_analyze_single_point(self, agent):
        """Test trend analysis with single data point."""
        result = agent.analyze_trends([{"ph": 7.0}])
        assert "average_score" in result
        assert result["data_points"] == 1

    def test_analyze_improving_trend(self, agent):
        """Test trend analysis with improving data."""
        data = [
            {"ph": 5.0, "turbidity": 10.0},  # Poor quality
            {"ph": 7.0, "turbidity": 2.0},    # Better quality
        ]
        result = agent.analyze_trends(data)
        assert result["trend"] == "improving"

    def test_analyze_multiple_points(self, agent):
        """Test trend analysis with multiple data points."""
        data = [
            {"ph": 7.0},
            {"ph": 7.2},
            {"ph": 7.1},
        ]
        result = agent.analyze_trends(data)
        assert result["data_points"] == 3
        assert "min_score" in result
        assert "max_score" in result
