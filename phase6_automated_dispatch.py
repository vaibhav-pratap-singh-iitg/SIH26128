import json
import os
from datetime import datetime, timedelta
import pandas as pd

# =====================================================================
# MODIFIABLE DISPATCH HYPERPARAMETERS & ROUTING CONFIG
# =====================================================================
DISPATCH_CONFIG = {
    # Outbreak threshold to trigger automated lab escalation & broadcasts
    "trigger_cluster_status": ["ACTIVE_OUTBREAK_SURGE"],
    "min_cluster_confidence": 75.0,  # Minimum confidence % to trigger emergency response
    "max_samples_per_village": 3,  # Max sentinel biological samples to draw per village
    # Diagnostic Facility Routing Table by District
    "diagnostic_lab_routing": {
        "Pune": "District Veterinary Polyclinic & Diagnostic Lab, Pune",
        "Ahmednagar": "District Animal Disease Investigation Lab, Ahmednagar",
        "Kolhapur": "Regional Veterinary Disease Diagnostic Laboratory, Kolhapur",
        "DEFAULT": "Central Disease Investigation Laboratory (CDIL), Pune",
    },
    # Biological Sample Protocols by Pathogen
    "sample_protocols": {
        "Lumpy_Skin_Disease": {
            "specimen": "Skin Scraping / Scab & EDTA Blood",
            "transport_medium": "Viral Transport Medium (VTM) at 4°C",
            "test_type": "Real-Time PCR (Capripoxvirus-specific)",
        },
        "Foot_and_Mouth_Disease": {
            "specimen": "Vesicular Fluid & Tongue Epithelium",
            "transport_medium": "Phosphate Buffered Glycerol (pH 7.2–7.6)",
            "test_type": "Antigen Detection ELISA / RT-PCR",
        },
        "PPR_Goat_Plague": {
            "specimen": "Nasal / Ocular Swab & Whole Blood",
            "transport_medium": "Sterile Saline on Ice",
            "test_type": "Immunocapture ELISA",
        },
        "Anthrax": {
            "specimen": "Peripheral Blood Smear (Ear Vein)",
            "transport_medium": "Air-dried sterile slide container (DO NOT OPEN CARCASS)",
            "test_type": "Polychrome Methylene Blue (McFadyean Reaction)",
        },
        "DEFAULT": {
            "specimen": "Whole Blood & Serum",
            "transport_medium": "Ice pack cold chain (4°C)",
            "test_type": "Differential Multiplex Panel",
        },
    },
}


# =====================================================================
# AUTOMATED INCIDENT DISPATCH ENGINE
# =====================================================================
class SurveillanceDispatchEngine:

    def __init__(self, config: dict):
        self.config = config

    def _get_lab_destination(self, district: str) -> str:
        return self.config["diagnostic_lab_routing"].get(
            district, self.config["diagnostic_lab_routing"]["DEFAULT"]
        )

    def _get_sampling_protocol(self, disease: str) -> dict:
        return self.config["sample_protocols"].get(
            disease, self.config["sample_protocols"]["DEFAULT"]
        )

    def generate_containment_actions(
        self,
        df_clustered: pd.DataFrame,
        df_users: pd.DataFrame,
        df_geo: pd.DataFrame,
        advisories: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
        """Scans radar output, generates lab manifests, dispatches multilingual

        SMS advisories, and compiles incident command reports.
        """
        # Merge administrative district names into the clustered reports
        df_radar = df_clustered.merge(
            df_geo[["village_id", "block_name", "district_name"]],
            on="village_id",
            how="left",
            suffixes=("", "_geo"),
        )
        if "district_name_geo" in df_radar.columns:
            df_radar["district_name"] = df_radar["district_name_geo"].combine_first(
                df_radar.get("district_name")
            )

        # 1. Filter Active Threat Clusters
        threat_mask = (
            df_radar["cluster_status"].isin(
                self.config["trigger_cluster_status"]
            )
        ) & (
            df_radar["outbreak_confidence_index"]
            >= self.config["min_cluster_confidence"]
        )

        threat_cases = df_radar[threat_mask]
        active_cids = sorted(threat_cases["cluster_id"].unique())

        lab_orders = []
        sms_broadcasts = []
        command_briefings = []

        sample_counter = 80001
        order_counter = 5001

        for cid in active_cids:
            c_data = threat_cases[threat_cases["cluster_id"] == cid]
            top_pathogen = c_data["predicted_disease"].mode()[0]
            primary_district = c_data["district_name"].mode()[0]
            affected_villages = c_data["village_id"].unique().tolist()
            conf_idx = c_data["outbreak_confidence_index"].iloc[0]
            velocity = c_data["transmission_velocity"].iloc[0]
            attack_density = c_data["attack_density_per_1k"].iloc[0]

            protocol = self._get_sampling_protocol(top_pathogen)
            assigned_lab = self._get_lab_destination(primary_district)

            # -------------------------------------------------------------
            # A. AUTOMATED LAB SAMPLING MANIFEST (Sentinel Selection)
            # -------------------------------------------------------------
            for vil in affected_villages:
                vil_cases = c_data[c_data["village_id"] == vil]
                # Prioritize severe cases and mortalities for lab verification
                sentinel_cases = vil_cases.sort_values(
                    by=["mortality_count", "severity_score"], ascending=False
                ).head(self.config["max_samples_per_village"])

                for _, case in sentinel_cases.iterrows():
                    lab_orders.append(
                        {
                            "order_id": f"ORD_{order_counter}",
                            "sample_id": f"SMP_{sample_counter}",
                            "linked_report_id": case["report_id"],
                            "cluster_id": cid,
                            "village_id": vil,
                            "district": primary_district,
                            "species": case["species"],
                            "suspected_pathogen": top_pathogen,
                            "specimen_required": protocol["specimen"],
                            "transport_protocol": protocol["transport_medium"],
                            "recommended_test": protocol["test_type"],
                            "destination_facility": assigned_lab,
                            "dispatch_priority": (
                                "EMERGENCY_STAT"
                                if case["mortality_count"] > 0
                                else "HIGH"
                            ),
                            "order_timestamp": datetime.now().strftime(
                                "%Y-%m-%d %H:%M:%S"
                            ),
                        }
                    )
                    sample_counter += 1
                    order_counter += 1

            # -------------------------------------------------------------
            # B. GEO-FENCED MULTILINGUAL ADVISORY BROADCAST
            # -------------------------------------------------------------
            # Pull advisory text templates (fallback to General Mortality Spike if missing)
            adv_payload = advisories.get(
                top_pathogen, advisories.get("General_Mortality_Spike", {})
            )
            text_en = adv_payload.get(
                "advisory_en",
                f"Health Alert: Suspected {top_pathogen} outbreak. Isolate symptomatic animals immediately.",
            )
            text_mr = adv_payload.get(
                "advisory_mr",
                f"आरोग्य चेतावणी: आपल्या भागात {top_pathogen} चा संशयित प्रादुर्भाव. आजारी जनावरांना त्वरित वेगळे करा.",
            )

            # Target all registered livestock holders & workers in affected villages
            target_recipients = df_users[
                df_users["village_id"].isin(affected_villages)
            ]

            for _, user in target_recipients.iterrows():
                lang = user.get("preferred_language", "Marathi")
                # Route appropriate language payload
                if lang == "Marathi":
                    body = f"[पशुसंवर्धन विभाग चेतावणी - {vil}] {text_mr}"
                else:
                    body = f"[Animal Health Warning - {vil}] {text_en}"

                sms_broadcasts.append(
                    {
                        "broadcast_id": f"SMS_{len(sms_broadcasts) + 1:05d}",
                        "recipient_user_id": user["user_id"],
                        "recipient_role": user["role"],
                        "village_id": user["village_id"],
                        "target_cluster_id": cid,
                        "language": lang,
                        "message_body": body,
                        "delivery_channel": (
                            "SMS_AND_IVR"
                            if user["role"] == "farmer"
                            else "OFFICER_PORTAL_ALERT"
                        ),
                        "status": "QUEUED_FOR_DISPATCH",
                    }
                )

            # -------------------------------------------------------------
            # C. DISTRICT INCIDENT COMMAND BRIEFING
            # -------------------------------------------------------------
            command_briefings.append(
                {
                    "incident_id": f"INC_DVO_{primary_district.upper()}_{cid:03d}",
                    "cluster_id": int(cid),
                    "primary_district": primary_district,
                    "dominant_pathogen": top_pathogen,
                    "confidence_score": f"{conf_idx}%",
                    "transmission_velocity": f"{velocity} new cases/day",
                    "attack_density": f"{attack_density} per 1,000 head",
                    "affected_villages": affected_villages,
                    "total_active_cases": len(c_data),
                    "sentinel_samples_ordered": len(
                        [o for o in lab_orders if o["cluster_id"] == cid]
                    ),
                    "advisories_broadcasted": len(
                        [
                            b
                            for b in sms_broadcasts
                            if b["target_cluster_id"] == cid
                        ]
                    ),
                    "recommended_quarantine_radius_km": 10.0,
                    "action_required": (
                        "Immediate movement restriction on cloven-hoofed animals; "
                        "Establish ring vaccination boundary within 5 km."
                    ),
                }
            )

        df_lab_orders = pd.DataFrame(lab_orders)
        df_sms = pd.DataFrame(sms_broadcasts)

        return df_lab_orders, df_sms, command_briefings


# =====================================================================
# EXECUTION SCRIPT
# =====================================================================
if __name__ == "__main__":
    try:
        df_clustered = pd.read_csv("surveillance_radar_output.csv")
        df_users = pd.read_csv("users.csv")
        df_geo = pd.read_csv("geo_hierarchy.csv")
        with open("multilingual_advisories.json", "r", encoding="utf-8") as f:
            advisories = json.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Missing required dataset: {e}. Ensure Phase 1–4 scripts have run."
        )

    # 1. Initialize & Run Dispatch Engine
    engine = SurveillanceDispatchEngine(DISPATCH_CONFIG)
    df_lab_orders, df_sms, briefings = engine.generate_containment_actions(
        df_clustered, df_users, df_geo, advisories
    )

    # 2. Export Operational Manifests
    df_lab_orders.to_csv("dispatched_lab_orders.csv", index=False)
    df_sms.to_csv("dispatched_sms_broadcasts.csv", index=False)
    with open(
        "incident_command_briefings.json", "w", encoding="utf-8"
    ) as f:
        json.dump(briefings, f, ensure_ascii=False, indent=2)

    # 3. Print Operational Summary
    print("Phase 6 Automated Dispatch Complete:")
    print("=" * 60)
    print(f" - Active Threat Clusters Evaluated : {len(briefings)}")
    print(f" - Diagnostic Lab Orders Generated  : {len(df_lab_orders)}")
    print(f" - Multilingual Alerts Queued       : {len(df_sms)}")
    print("=" * 60)

    # 4. Display Incident Command Briefing
    if briefings:
        top_incident = briefings[0]
        print("\n[!] Incident Command Briefing #1:")
        for k, v in top_incident.items():
            print(f"    {k:<35}: {v}")

    # 5. Display Sample Dispatched SMS (Marathi vs. English)
    if not df_sms.empty:
        print("\nSample Field Broadcast Payloads:")
        marathi_sample = df_sms[df_sms["language"] == "Marathi"].iloc[0]
        english_sample = df_sms[df_sms["language"] == "English"].iloc[0]

        print(
            f" - [Marathi / Recipient {marathi_sample['recipient_user_id']}]:\n   {marathi_sample['message_body']}\n"
        )
        print(
            f" - [English / Recipient {english_sample['recipient_user_id']}]:\n   {english_sample['message_body']}\n"
        )