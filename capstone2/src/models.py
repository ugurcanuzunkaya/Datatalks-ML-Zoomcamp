from pydantic import BaseModel, Field


class StressInput(BaseModel):
    """
    Pydantic model representing the input features for stress prediction.
    Each field corresponds to a specific survey question or physiological measurement.
    """

    anxiety_level: int = Field(..., description="Anxiety Level (0-21)")
    self_esteem: int = Field(..., description="Self Esteem (0-30)")
    mental_health_history: int = Field(
        ..., description="Mental Health History (0: No, 1: Yes)"
    )
    depression: int = Field(..., description="Depression (0-27)")
    headache: int = Field(..., description="Headache frequency (0-5)")
    blood_pressure: int = Field(..., description="Blood Pressure (1-3)")
    sleep_quality: int = Field(..., description="Sleep Quality (0-5)")
    breathing_problem: int = Field(..., description="Breathing Problem (0: No, 1: Yes)")
    noise_level: int = Field(..., description="Noise Level (0-5)")
    living_conditions: int = Field(..., description="Living Conditions (0-5)")
    safety: int = Field(..., description="Safety (0-5)")
    basic_needs: int = Field(..., description="Basic Needs (0-5)")
    academic_performance: int = Field(..., description="Academic Performance (0-5)")
    study_load: int = Field(..., description="Study Load (0-5)")
    teacher_student_relationship: int = Field(
        ..., description="Teacher Student Relationship (0-5)"
    )
    future_career_concerns: int = Field(..., description="Future Career Concerns (0-5)")
    social_support: int = Field(..., description="Social Support (0-5)")
    peer_pressure: int = Field(..., description="Peer Pressure (0-5)")
    extracurricular_activities: int = Field(
        ..., description="Extracurricular Activities (0-5)"
    )
    bullying: int = Field(..., description="Bullying (0-5)")


class PredictionOutput(BaseModel):
    """
    Pydantic model representing the prediction result.
    """

    stress_level_code: int = Field(..., description="Predicted Label (0, 1, 2)")
    stress_level: str = Field(
        ..., description="Human-readable label (Low, Medium, High)"
    )
