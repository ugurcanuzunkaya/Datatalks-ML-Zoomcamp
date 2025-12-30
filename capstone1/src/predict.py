import pickle
import pandas as pd
import numpy as np
from src.models import InsuranceInput, PredictionOutput, SavingsOpportunity


class Predictor:
    def __init__(self, model_path: str):
        with open(model_path, "rb") as f:
            self.model = pickle.load(f)

    def predict_one(self, data: InsuranceInput) -> float:
        df = pd.DataFrame([data.model_dump()])
        prediction = self.model.predict(df)
        return float(prediction[0])

    def get_opportunities(self, data: InsuranceInput, current_cost: float) -> list[SavingsOpportunity]:
        opportunities = []

        # 1. Quit Smoking
        if data.smoker == "yes":
            modified_data = data.model_copy()
            modified_data.smoker = "no"
            new_cost = self.predict_one(modified_data)
            saving = current_cost - new_cost
            if saving > 0:
                opportunities.append(
                    SavingsOpportunity(
                        name="Quit Smoking",
                        description="If you stop smoking, your insurance cost could decrease significantly.",
                        potential_savings=round(saving, 2),
                        new_cost=round(new_cost, 2),
                    )
                )

        # 2. Lower BMI (if overweight/obese)
        if data.bmi > 25:
            target_bmi = 24.9
            modified_data = data.model_copy()
            modified_data.bmi = target_bmi
            new_cost = self.predict_one(modified_data)
            saving = current_cost - new_cost
            if saving > 0:
                opportunities.append(
                    SavingsOpportunity(
                        name="Optimize BMI",
                        description=f"lowering your BMI to {target_bmi} (Healthy weight).",
                        potential_savings=round(saving, 2),
                        new_cost=round(new_cost, 2),
                    )
                )

        # 3. Location Arbitrage (Hypothetical - just for analysis)
        current_region = data.region
        cheapest_region = None
        max_saving = 0

        for region in ["northeast", "northwest", "southeast", "southwest"]:
            if region == current_region:
                continue
            modified_data = data.model_copy()
            modified_data.region = region
            new_cost = self.predict_one(modified_data)
            saving = current_cost - new_cost
            if saving > max_saving:
                max_saving = saving
                cheapest_region = region

        if max_saving > 100:  # Only show if meaningful
            opportunities.append(
                SavingsOpportunity(
                    name="Move Region",
                    description=f"Relocating to {cheapest_region} could save you money.",
                    potential_savings=round(max_saving, 2),
                    new_cost=round(current_cost - max_saving, 2),
                )
            )

        return opportunities

    def make_prediction(self, data: InsuranceInput) -> PredictionOutput:
        current_cost = self.predict_one(data)
        opportunities = self.get_opportunities(data, current_cost)

        return PredictionOutput(current_cost=round(current_cost, 2), opportunities=opportunities)


# Global instance to be initialized on app startup
predictor = None
