import json
import os
from datetime import datetime
import numpy as np
import pandas as pd


class DynamicSafetyNetPipeline:
    """Dynamically parses and vectorizes livestock surveillance records using

    external medical ontologies, alias normalization, and planar spatial projections.
    """

    def __init__(
        self,
        ontology_path: str = "disease_ontology.json",
        anchor_lat: float = 18.5204,
        anchor_lon: float = 73.8567,
        default_novel_weight: float = 1.0,
    ):
        self.ontology_path = ontology_path
        self.anchor_lat = anchor_lat
        self.anchor_lon = anchor_lon
        self.default_novel_weight = default_novel_weight

        self.ontology = self._load_ontology()
        self.known_weights = self._extract_known_weights()
        self.alias_map = self.ontology.get("alias_map", {})

        # Vocabulary containers (fit dynamically)
        self.vocabulary: list[str] = []
        self.symptom_to_idx: dict[str, int] = {}
        self.species_list: list[str] = []
        self.species_to_idx: dict[str, int] = {}

    def _load_ontology(self) -> dict:
        if not os.path.exists(self.ontology_path):
            raise FileNotFoundError(
                f"Ontology configuration not found at '{self.ontology_path}'. "
                f"Please ensure 'disease_ontology.json' is present in the workspace."
            )
        with open(self.ontology_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _extract_known_weights(self) -> dict[str, float]:
        weights = {}
        for disease, profile in self.ontology.get("diseases", {}).items():
            for symptom, weight in profile.get("symptom_weights", {}).items():
                # If a symptom appears in multiple diseases, preserve the maximum safety weight
                weights[symptom] = max(weights.get(symptom, 0.0), float(weight))
        for noise_sym, weight in self.ontology.get("noise_symptoms", {}).items():
            weights[noise_sym] = weights.get(noise_sym, float(weight))
        return weights

    def normalize_term(self, term: str) -> str:
        """Cleans terms and maps regional or colloquial vernacular to canonical medical tokens."""
        cleaned = term.strip().lower()
        return self.alias_map.get(cleaned, cleaned)

    def fit_vocabulary(
        self,
        raw_symptom_series: pd.Series,
        raw_species_series: pd.Series = None,
    ):
        """Discovers all active symptoms and species across the dataset and combines them

        with the pre-configured clinical ontology terms.
        """
        discovered_symptoms = set()
        for entry in raw_symptom_series.dropna():
            for item in str(entry).split(","):
                canonical = self.normalize_term(item)
                if canonical:
                    discovered_symptoms.add(canonical)

        # Union discovered symptoms with curated ontology terms
        all_symptoms = sorted(
            list(discovered_symptoms.union(self.known_weights.keys()))
        )
        self.vocabulary = all_symptoms
        self.symptom_to_idx = {sym: idx for idx, sym in enumerate(all_symptoms)}

        # Fit species dynamically
        if raw_species_series is not None:
            discovered_species = sorted(
                list(raw_species_series.dropna().astype(str).unique())
            )
        else:
            discovered_species = ["cattle", "buffalo", "goat", "sheep", "poultry"]

        self.species_list = discovered_species
        self.species_to_idx = {
            sp: idx for idx, sp in enumerate(discovered_species)
        }

    def get_symptom_weight(self, symptom: str) -> float:
        """Returns the clinical weight; defaults unlisted novel symptoms to a safe fallback weight."""
        return self.known_weights.get(symptom, self.default_novel_weight)

    def _project_coordinates(
        self, lats: np.ndarray, lons: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Projects latitude/longitude onto a local planar Cartesian grid (km)

        centered at the anchor coordinates via equirectangular projection.
        """
        lat_rad = np.radians(self.anchor_lat)
        y_km = (lats - self.anchor_lat) * 110.574
        x_km = (lons - self.anchor_lon) * (111.320 * np.cos(lat_rad))
        return np.round(x_km, 4), np.round(y_km, 4)

    def transform(
        self, df_reports: pd.DataFrame, df_geo: pd.DataFrame
    ) -> dict[str, any]:
        """Executes full multi-hot weighting, demographic joining, and matrix transformations."""
        if not self.vocabulary:
            raise RuntimeError(
                "Pipeline vocabulary has not been fitted. Call 'fit_vocabulary' prior to 'transform'."
            )

        # 1. Join geographic and livestock population tables
        merged_df = df_reports.merge(
            df_geo[
                [
                    "village_id",
                    "latitude",
                    "longitude",
                    "cattle_pop",
                    "buffalo_pop",
                    "goat_pop",
                    "sheep_pop",
                    "poultry_pop",
                ]
            ],
            on="village_id",
            how="left",
        )

        # 2. Planar Projection & Continuous Time Calculation
        x_km, y_km = self._project_coordinates(
            merged_df["latitude"].values, merged_df["longitude"].values
        )
        merged_df["x_km"] = x_km
        merged_df["y_km"] = y_km

        t0 = pd.to_datetime(merged_df["timestamp"]).min()
        delta_t_days = (
            pd.to_datetime(merged_df["timestamp"]) - t0
        ).dt.total_seconds() / 86400.0
        merged_df["delta_t_days"] = np.round(delta_t_days, 4)

        # 3. Denominator Assignment: Species Base Population
        def get_denominator(row):
            sp = str(row.get("species", "")).lower()
            col_name = f"{sp}_pop"
            return row.get(col_name, 1000)

        merged_df["species_census"] = merged_df.apply(get_denominator, axis=1)

        # 4. Asymmetric Clinical Multi-Hot Vectorization
        num_reports = len(merged_df)
        num_symptoms = len(self.vocabulary)
        clinical_symptom_matrix = np.zeros(
            (num_reports, num_symptoms), dtype=np.float32
        )

        for i, symptom_str in enumerate(merged_df["symptoms"]):
            if pd.isna(symptom_str) or not symptom_str:
                continue
            active_symptoms = [
                self.normalize_term(s) for s in str(symptom_str).split(",")
            ]
            for sym in active_symptoms:
                if sym in self.symptom_to_idx:
                    idx = self.symptom_to_idx[sym]
                    clinical_symptom_matrix[i, idx] = self.get_symptom_weight(
                        sym
                    )

        # 5. Dynamic Species One-Hot Encoding
        num_species = len(self.species_list)
        species_matrix = np.zeros((num_reports, num_species), dtype=np.float32)
        for i, sp in enumerate(merged_df["species"]):
            if sp in self.species_to_idx:
                species_matrix[i, self.species_to_idx[sp]] = 1.0

        # 6. Severity & Mortality Rescaling
        norm_severity = (
            merged_df["severity_score"].values.reshape(-1, 1).astype(np.float32)
            / 10.0
        )
        mortality_flag = (
            (merged_df["report_type"] == "mortality_alert")
            .astype(np.float32)
            .values.reshape(-1, 1)
        )

        # Concatenate into full Clinical Feature Vector V_clinical
        v_clinical = np.hstack(
            [
                clinical_symptom_matrix,
                species_matrix,
                norm_severity,
                mortality_flag,
            ]
        )

        # 7. Construct Spatiotemporal Vector V_st: [x_km, y_km, delta_t, species_census]
        v_st = np.column_stack(
            [
                merged_df["x_km"].values,
                merged_df["y_km"].values,
                merged_df["delta_t_days"].values,
                merged_df["species_census"].values,
            ]
        ).astype(np.float32)

        return {
            "enriched_dataframe": merged_df,
            "v_clinical": v_clinical,
            "v_st": v_st,
            "vocabulary": self.vocabulary,
            "species_labels": self.species_list,
            "feature_dim_clinical": v_clinical.shape[1],
            "feature_dim_st": v_st.shape[1],
            "known_weights": self.known_weights,
        }


# =====================================================================
# VERIFICATION & EXECUTION TEST
# =====================================================================
if __name__ == "__main__":
    # Ensure inputs are present
    try:
        df_reports = pd.read_csv("symptom_mortality_reports.csv")
        df_geo = pd.read_csv("geo_hierarchy.csv")
    except FileNotFoundError:
        raise FileNotFoundError(
            "Required datasets not found. Run 'generate_and_plot.py' first to generate CSVs."
        )

    # 1. Initialize pipeline with external ontology
    pipeline = DynamicSafetyNetPipeline(ontology_path="disease_ontology.json")

    # 2. Fit dynamically across incoming symptom streams and species
    pipeline.fit_vocabulary(
        df_reports["symptoms"], raw_species_series=df_reports["species"]
    )

    # 3. Transform reports into safety-net matrices
    feature_bundle = pipeline.transform(df_reports, df_geo)

    print("Phase 1 Dynamic Feature Extraction Complete:")
    print(f" - Reports Processed:        {len(feature_bundle['enriched_dataframe'])}")
    print(
        f" - V_clinical Shape:         {feature_bundle['v_clinical'].shape} "
        f"({len(feature_bundle['vocabulary'])} symptoms + {len(feature_bundle['species_labels'])} species + 2 status flags)"
    )
    print(
        f" - V_st Shape:               {feature_bundle['v_st'].shape} (X_km, Y_km, Delta_t_days, Species_Census)"
    )
    print(
        f" - Dynamic Vocabulary Size:  {len(feature_bundle['vocabulary'])} clinical tokens"
    )

    # 4. Demonstrate vernacular alias resolution & novel symptom fallback
    print("\nTesting Alias Resolution & Novel Fallbacks:")
    test_vernacular = ["skin bumps", "गाठी", "corneal_opacity_novel"]
    for sym in test_vernacular:
        canonical = pipeline.normalize_term(sym)
        weight = pipeline.get_symptom_weight(canonical)
        print(f" -> Input: '{sym}' => Normalized: '{canonical}' => Assigned Weight: {weight}")