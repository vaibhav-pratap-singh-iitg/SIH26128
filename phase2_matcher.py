import json
import os
import numpy as np
import pandas as pd
from phase1_vectorizer import DynamicSafetyNetPipeline


class ClinicalDiagnosisEngine:
    """Computes differential disease probabilities, asymmetric safety-net alert levels,

    and interactive verification prompts for field workers.
    """

    def __init__(
        self,
        pipeline: DynamicSafetyNetPipeline,
        ontology_path: str = "disease_ontology.json",
        safety_net_threshold: float = 0.30,  # Low-barrier trigger to catch early/partial signs
    ):
        self.pipeline = pipeline
        self.safety_net_threshold = safety_net_threshold

        with open(ontology_path, "r", encoding="utf-8") as f:
            self.ontology = json.load(f)

        self.diseases = self.ontology.get("diseases", {})
        self.disease_names = sorted(list(self.diseases.keys()))
        self.num_diseases = len(self.disease_names)

        # Build archetype prototype vectors aligned with Phase 1 vocabulary
        self.prototype_matrix, self.species_mask_matrix = (
            self._build_disease_prototypes()
        )

    def _build_disease_prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        """Constructs weighted clinical prototype vectors and species gating masks

        aligned exactly with the dynamic vocabulary and species indices.
        """
        num_symptoms = len(self.pipeline.vocabulary)
        num_species = len(self.pipeline.species_list)

        prototypes = np.zeros(
            (self.num_diseases, num_symptoms), dtype=np.float32
        )
        species_mask = np.zeros(
            (self.num_diseases, num_species), dtype=np.float32
        )

        for d_idx, d_name in enumerate(self.disease_names):
            profile = self.diseases[d_name]

            # 1. Populate archetype symptom weights
            for sym, weight in profile.get("symptom_weights", {}).items():
                canonical = self.pipeline.normalize_term(sym)
                if canonical in self.pipeline.symptom_to_idx:
                    s_idx = self.pipeline.symptom_to_idx[canonical]
                    prototypes[d_idx, s_idx] = float(weight)

            # 2. Populate species compatibility mask (1.0 = valid host)
            for sp in profile.get("target_species", []):
                sp_clean = sp.strip().lower()
                if sp_clean in self.pipeline.species_to_idx:
                    sp_idx = self.pipeline.species_to_idx[sp_clean]
                    species_mask[d_idx, sp_idx] = 1.0

        return prototypes, species_mask

    def _calculate_cosine_similarity(
        self, patient_symptoms: np.ndarray
    ) -> np.ndarray:
        """Computes weighted cosine similarity between patient symptom vectors

        and all canonical disease prototypes.
        """
        dot_product = np.dot(patient_symptoms, self.prototype_matrix.T)

        patient_norm = np.linalg.norm(patient_symptoms, axis=1, keepdims=True)
        proto_norm = np.linalg.norm(self.prototype_matrix, axis=1, keepdims=True).T

        denom = patient_norm * proto_norm
        denom = np.where(denom == 0.0, 1e-7, denom)  # Avoid division by zero

        similarity = dot_product / denom
        return np.clip(similarity, 0.0, 1.0)

    def diagnose_batch(self, feature_bundle: dict[str, any]) -> pd.DataFrame:
        """Runs differential diagnosis, confidence scoring, safety-net alert routing,

        and gap checklist generation across all input records.
        """
        v_clinical = feature_bundle["v_clinical"]
        df_merged = feature_bundle["enriched_dataframe"].copy()

        num_symptoms = len(self.pipeline.vocabulary)
        num_species = len(self.pipeline.species_list)

        # Slice sub-matrices from Phase 1 V_clinical
        patient_symptoms = v_clinical[:, :num_symptoms]
        patient_species = v_clinical[:, num_symptoms : num_symptoms + num_species]
        patient_severity = v_clinical[:, -2]  # Normalized 0.0 to 1.0
        mortality_flags = v_clinical[:, -1]

        # 1. Raw Symptom Similarity
        raw_sim = self._calculate_cosine_similarity(patient_symptoms)

        # 2. Apply Biological Species Filter (Zero out impossible hosts)
        # Dot product of patient one-hot species with disease target mask
        species_compatibility = np.dot(
            patient_species, self.species_mask_matrix.T
        )
        masked_sim = raw_sim * species_compatibility

        # 3. Softmax Affinity Distribution over compatible diseases
        exp_sim = np.exp(masked_sim * 4.0) * species_compatibility
        sum_exp = np.sum(exp_sim, axis=1, keepdims=True)
        sum_exp = np.where(sum_exp == 0.0, 1e-7, sum_exp)
        affinity_probs = exp_sim / sum_exp

        # 4. Diagnose Each Case
        top_diagnoses = []
        confidence_scores = []
        confidence_margins = []
        differential_records = []
        alert_levels = []
        missing_cardinals = []
        verification_prompts = []

        for i in range(len(df_merged)):
            row_probs = affinity_probs[i]
            row_sims = masked_sim[i]
            species_name = df_merged["species"].iloc[i]
            reported_symptoms = [
                self.pipeline.normalize_term(s)
                for s in str(df_merged["symptoms"].iloc[i]).split(",")
            ]

            # Sorted matches
            ranked_indices = np.argsort(row_sims)[::-1]
            top_idx = ranked_indices[0]
            runner_up_idx = ranked_indices[1] if len(ranked_indices) > 1 else top_idx

            top_disease = self.disease_names[top_idx]
            top_similarity = float(row_sims[top_idx])
            top_affinity = float(row_probs[top_idx])
            runner_up_similarity = float(row_sims[runner_up_idx])

            margin = top_similarity - runner_up_similarity

            # Check if animal showed no match or species has zero allowed diseases
            if species_compatibility[i, top_idx] == 0.0 or top_similarity < 0.05:
                top_disease = "Inconclusive / Routine Noise"
                top_affinity = 0.0
                confidence = 0.0
                margin = 0.0
            else:
                confidence = round(top_affinity * 100.0, 1)

            # Build Top-3 Differential Summary String
            diff_list = []
            for rank in range(min(3, self.num_diseases)):
                idx = ranked_indices[rank]
                if species_compatibility[i, idx] > 0 and row_sims[idx] > 0.05:
                    diff_list.append(
                        f"{self.disease_names[idx]} ({round(row_sims[idx] * 100, 1)}%)"
                    )
            diff_str = " | ".join(diff_list) if diff_list else "Inconclusive"

            # 5. Asymmetric Safety-Net Decision Logic
            # Trades false alarms for low false negatives on high-threat diseases
            sev = patient_severity[i] * 10.0
            is_mort = mortality_flags[i] == 1.0

            if is_mort and sev >= 7.0:
                alert = "EMERGENCY_RED"
            elif top_similarity >= self.safety_net_threshold and sev >= 5.0:
                alert = "ALERT_RED"
            elif top_similarity >= 0.15 or sev >= 4.0:
                alert = "WATCH_AMBER"
            else:
                alert = "ROUTINE_GREEN"

            # 6. Interactive Symptom Gap Checker
            missing_cardinal_list = []
            prompt = "No follow-up required."

            if top_disease in self.diseases:
                target_profile = self.diseases[top_disease]["symptom_weights"]
                cardinal_targets = [
                    sym for sym, w in target_profile.items() if w >= 2.5
                ]
                missing_cardinal_list = [
                    sym
                    for sym in cardinal_targets
                    if sym not in reported_symptoms
                ]

                if missing_cardinal_list:
                    readable_missing = ", ".join(
                        [s.replace("_", " ") for s in missing_cardinal_list]
                    )
                    prompt = (
                        f"Targeting {top_disease.replace('_', ' ')}: Check if animal exhibits: "
                        f"[{readable_missing}]."
                    )

            top_diagnoses.append(top_disease)
            confidence_scores.append(confidence)
            confidence_margins.append(round(margin * 100.0, 1))
            differential_records.append(diff_str)
            alert_levels.append(alert)
            missing_cardinals.append(",".join(missing_cardinal_list))
            verification_prompts.append(prompt)

        # 7. Append Diagnostic Columns
        df_merged["predicted_disease"] = top_diagnoses
        df_merged["diagnostic_confidence_pct"] = confidence_scores
        df_merged["confidence_margin_pct"] = confidence_margins
        df_merged["differential_diagnosis_top3"] = differential_records
        df_merged["safety_net_alert_level"] = alert_levels
        df_merged["missing_cardinal_symptoms"] = missing_cardinals
        df_merged["followup_verification_prompt"] = verification_prompts

        return df_merged


# =====================================================================
# VERIFICATION & EXECUTION TEST
# =====================================================================
if __name__ == "__main__":
    # Ensure inputs from previous modules exist
    try:
        df_reports = pd.read_csv("symptom_mortality_reports.csv")
        df_geo = pd.read_csv("geo_hierarchy.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Required data files not found. Please run 'generate_and_plot.py' first."
        )

    # 1. Execute Feature Pipeline (Phase 1)
    pipeline = DynamicSafetyNetPipeline(ontology_path="disease_ontology.json")
    pipeline.fit_vocabulary(
        df_reports["symptoms"], raw_species_series=df_reports["species"]
    )
    feature_bundle = pipeline.transform(df_reports, df_geo)

    # 2. Execute Diagnosis Engine (Phase 2)
    engine = ClinicalDiagnosisEngine(
        pipeline=pipeline,
        ontology_path="disease_ontology.json",
        safety_net_threshold=0.30,  # Catch even 30% matching profiles to prevent missed outbreaks
    )
    df_diagnosed = engine.diagnose_batch(feature_bundle)

    # 3. Save Enriched Diagnostic Feed
    df_diagnosed.to_csv("diagnosed_clinical_feed.csv", index=False)

    print("Phase 2 Clinical Diagnosis Complete:")
    print(f" - Diagnosed Cases:          {len(df_diagnosed)}")
    print(f" - Safety-Net Triage Breakdown:\n{df_diagnosed['safety_net_alert_level'].value_counts()}")

    # 4. Inspect Sample Outbreak Alert
    red_alerts = df_diagnosed[
        df_diagnosed["safety_net_alert_level"].isin(["ALERT_RED", "EMERGENCY_RED"])
    ]
    if not red_alerts.empty:
        sample = red_alerts.iloc[0]
        print("\nSample High-Priority Alert Inspection:")
        print(f" - Report ID:           {sample['report_id']}")
        print(f" - Species / Village:   {sample['species']} | {sample['village_id']}")
        print(f" - Reported Symptoms:   {sample['symptoms']}")
        print(f" - Primary Diagnosis:   {sample['predicted_disease']} ({sample['diagnostic_confidence_pct']}%)")
        print(f" - Differential Top-3:  {sample['differential_diagnosis_top3']}")
        print(f" - Safety Alert Status: {sample['safety_net_alert_level']}")
        print(f" - Dynamic Gap Prompt:  {sample['followup_verification_prompt']}")