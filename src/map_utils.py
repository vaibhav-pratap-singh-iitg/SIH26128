"""
Interactive hotspot map built with Folium.

Takes a village-level risk table (from risk_scoring.calculate_village_risk,
merged with regions.csv for coordinates) and renders a color-coded map with
a popup per village and a legend.
"""

from __future__ import annotations

import folium
import pandas as pd

CATEGORY_COLORS = {
    "LOW": "#2e7d32",
    "MEDIUM": "#f9a825",
    "HIGH": "#ef6c00",
    "CRITICAL": "#c62828",
}

MAHARASHTRA_CENTER = (19.6, 76.0)


def build_hotspot_map(village_risk_df: pd.DataFrame) -> folium.Map:
    """
    `village_risk_df` must have: village, district, block, latitude,
    longitude, risk_score, risk_category, report_count_7d, affected_animals,
    deaths, mortality_rate, high_concern_count, report_growth_rate.
    """
    m = folium.Map(location=MAHARASHTRA_CENTER, zoom_start=7, tiles="OpenStreetMap")

    # Drop rows with missing coordinates to avoid map crashes
    plot_df = village_risk_df.dropna(subset=["latitude", "longitude"])

    for _, row in plot_df.iterrows():
        category = row.get("risk_category", "LOW")
        color = CATEGORY_COLORS.get(category, "#2e7d32")
        radius = 6 + min(14, row.get("risk_score", 0) / 6)

        popup_html = f"""
        <b>{row['village']}</b><br>
        {row['block']}, {row['district']}<br>
        <hr style="margin:4px 0;">
        Risk score: <b>{row['risk_score']:.0f}/100</b> ({category})<br>
        Reports (7d): {int(row.get('report_count_7d', 0))}<br>
        Affected animals: {int(row.get('affected_animals', 0))}<br>
        Deaths: {int(row.get('deaths', 0))}<br>
        Mortality rate: {row.get('mortality_rate', 0):.1%}<br>
        HIGH-concern reports: {int(row.get('high_concern_count', 0))}<br>
        7-day growth ratio: {row.get('report_growth_rate', 0):.2f}x
        """
        folium.CircleMarker(
            location=(row["latitude"], row["longitude"]),
            radius=radius,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            weight=1,
            popup=folium.Popup(popup_html, max_width=260),
            tooltip=f"{row['village']} - {category}",
        ).add_to(m)

    _add_legend(m)
    return m


def _add_legend(m: folium.Map) -> None:
    legend_html = """
    <div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999;
                background: white; padding: 10px 14px; border-radius: 6px;
                border: 1px solid #ccc; font-size: 13px; box-shadow: 0 1px 4px rgba(0,0,0,0.2);">
        <b>Risk category</b><br>
    """
    for label, color in CATEGORY_COLORS.items():
        legend_html += (
            f'<span style="display:inline-block;width:10px;height:10px;'
            f'background:{color};border-radius:50%;margin-right:6px;"></span>{label}<br>'
        )
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))
