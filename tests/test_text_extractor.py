"""
Tests for modules/text_extractor.py

All tests operate on synthetic text strings — no PDF files or Ollama required.
"""

import unittest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.text_extractor import (
    extract_general_info,
    extract_chainage,
    extract_structures,
    extract_road_geometry,
    extract_utilities,
    extract_annotations,
)


class TestExtractGeneralInfo(unittest.TestCase):

    def test_nhai_authority(self):
        text = "NATIONAL HIGHWAYS AUTHORITY OF INDIA\nSome project"
        result = extract_general_info(text)
        self.assertTrue(any("Authority" in i for i in result))

    def test_drawing_number(self):
        text = "DRAWING NUMBER: NHAI/GL/2022/001"
        result = extract_general_info(text)
        self.assertTrue(any("Drawing Number" in i for i in result))

    def test_scale(self):
        text = "Scale: 1:500"
        result = extract_general_info(text)
        self.assertTrue(any("Scale" in i and "1:500" in i for i in result))

    def test_revision(self):
        text = "Rev. A"
        result = extract_general_info(text)
        self.assertTrue(any("Revision" in i for i in result))


class TestExtractChainage(unittest.TestCase):

    def test_start_end_extracted(self):
        text = "CH: 0+000 to some point CH: 3+000"
        result = extract_chainage(text)
        self.assertTrue(any("Start Chainage" in i for i in result))
        self.assertTrue(any("End Chainage" in i for i in result))

    def test_no_all_chainages_found_string(self):
        """Regression: 'All Chainages Found' mega-string should never appear."""
        text = "CH: 0+000  CH: 0+500  CH: 1+000  CH: 1+500  CH: 2+000"
        result = extract_chainage(text)
        self.assertFalse(any("All Chainages Found" in i for i in result))

    def test_single_chainage(self):
        text = "CH: 1+500"
        result = extract_chainage(text)
        self.assertTrue(any("1+500" in i for i in result))

    def test_coverage_range(self):
        text = "CH: 0+000 To CH: 3+000"
        result = extract_chainage(text)
        self.assertTrue(any("Sheet Coverage" in i for i in result))


class TestExtractStructures(unittest.TestCase):

    def test_culvert_with_chainage(self):
        text = "BOX CULVERT 2x1m at CH: 1+200 LHS"
        result = extract_structures(text)
        self.assertTrue(any("Culvert" in i for i in result))

    def test_legend_noise_filtered(self):
        """Short legend entries like 'Bridge' alone should be excluded."""
        text = "Bridge\nBox Culvert"  # legend items without chainage
        result = extract_structures(text)
        # Should not include bare 'Bridge' or 'Box Culvert' (< 20 chars, no CH)
        for item in result:
            self.assertTrue(len(item) > 20 or "CH" in item.upper(), f"Noise item leaked: {item}")

    def test_bridge_with_ch_included(self):
        text = "MINOR BRIDGE at CH: 2+500, span 5m"
        result = extract_structures(text)
        self.assertTrue(any("Bridge" in i or "Structure" in i for i in result))


class TestExtractRoadGeometry(unittest.TestCase):

    def test_design_speed_with_unit(self):
        text = "Design Speed: 100 km/h"
        result = extract_road_geometry(text)
        self.assertTrue(any("Design Speed" in i and "100" in i for i in result))

    def test_v_equals_with_unit(self):
        text = "V = 80 km/h"
        result = extract_road_geometry(text)
        self.assertTrue(any("80" in i for i in result))

    def test_easting_coord_not_captured(self):
        """Regression: easting coords like 'N = 2555316' must not be captured."""
        text = "N = 2555316 E = 456789"
        result = extract_road_geometry(text)
        speed_items = [i for i in result if "Design Speed" in i]
        self.assertEqual(len(speed_items), 0)

    def test_radius(self):
        text = "R = 600 m"
        result = extract_road_geometry(text)
        self.assertTrue(any("Radius" in i and "600" in i for i in result))

    def test_superelevation(self):
        text = "Superelevation: 4.5%"
        result = extract_road_geometry(text)
        self.assertTrue(any("Superelevation" in i for i in result))


class TestExtractUtilities(unittest.TestCase):

    def test_gas_pipeline(self):
        text = "GAS PIPE LINE crosses at CH: 1+400"
        result = extract_utilities(text)
        self.assertTrue(any("Gas Pipeline" in i for i in result))

    def test_bus_bay(self):
        text = "BUS BAY at CH: 2+600 RHS"
        result = extract_utilities(text)
        self.assertTrue(any("Bus Bay" in i for i in result))

    def test_railway(self):
        text = "RAILWAY LEVEL CROSSING at CH: 0+850"
        result = extract_utilities(text)
        self.assertTrue(any("Railway" in i for i in result))

    def test_ofc(self):
        text = "OFC cable alongside road"
        result = extract_utilities(text)
        self.assertTrue(any("OFC" in i for i in result))


class TestExtractAnnotations(unittest.TestCase):

    def test_benchmark(self):
        text = "B.M. No. 5 RL = 123.456"
        result = extract_annotations(text)
        self.assertTrue(any("Benchmark" in i for i in result))

    def test_gps_point(self):
        text = "GPS Point at CH: 1+000"
        result = extract_annotations(text)
        self.assertTrue(any("GPS" in i for i in result))

    def test_north_direction(self):
        text = "NORTH symbol shown on drawing"
        result = extract_annotations(text)
        self.assertTrue(any("North Direction" in i for i in result))

    def test_km_stone(self):
        text = "KM STONE 12"
        result = extract_annotations(text)
        self.assertTrue(any("KM Stone" in i for i in result))


if __name__ == "__main__":
    unittest.main()
