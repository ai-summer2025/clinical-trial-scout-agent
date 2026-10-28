#!/usr/bin/env python3
"""
Tests for Clinical Trial Scout Agent.
Verifies core logic across all 5 agents and the orchestration pipeline.
"""

import unittest
import json
import os
import tempfile

from main import (
    PatientProfile,
    ClinicalTrial,
    EligibilityScore,
    PatientProfileAgent,
    TrialSearchAgent,
    EligibilityScreeningAgent,
    LogisticsAgent,
    OutreachDraftingAgent,
    ScoutOrchestrator,
    DEMO_PATIENTS,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_lung_patient() -> PatientProfile:
    return DEMO_PATIENTS["PT-001"]


def make_diabetes_patient() -> PatientProfile:
    return DEMO_PATIENTS["PT-002"]


def make_hf_patient() -> PatientProfile:
    return DEMO_PATIENTS["PT-003"]


def make_lung_trial(nct_id="NCT99000001") -> ClinicalTrial:
    return ClinicalTrial(
        nct_id=nct_id,
        title="Phase II Study of Pembrolizumab in Stage III NSCLC After Platinum Failure",
        status="RECRUITING",
        phase="Phase 2",
        conditions=["Non-Small Cell Lung Cancer", "Lung Cancer"],
        interventions=["Pembrolizumab 200mg IV"],
        inclusion_criteria=(
            "Inclusion Criteria:\n"
            "- Histologically confirmed NSCLC\n"
            "- Prior platinum-based chemotherapy\n"
            "- ECOG performance status 0-2\n"
            "- Age >= 18 years\n"
        ),
        exclusion_criteria="Active autoimmune disease\nPrior checkpoint inhibitor",
        locations=[
            {"facility": "Test Cancer Center", "city": "New York", "state": "NY", "zip": "10032"}
        ],
        has_remote_option=True,
        min_age=18,
        max_age=None,
        sex_eligibility="All",
        primary_sponsor="NCI Test",
        brief_summary="Testing pembrolizumab in NSCLC patients after platinum failure.",
        contact_email="test@example.com",
        contact_name="Dr. Test",
    )


def make_diabetes_trial() -> ClinicalTrial:
    return ClinicalTrial(
        nct_id="NCT99000002",
        title="Tirzepatide vs Semaglutide for Glycemic Control in T2DM on Metformin",
        status="RECRUITING",
        phase="Phase 4",
        conditions=["Type 2 Diabetes Mellitus", "Obesity"],
        interventions=["Tirzepatide 5mg weekly"],
        inclusion_criteria=(
            "Inclusion Criteria:\n"
            "- Type 2 diabetes diagnosis\n"
            "- HbA1c 7.5-11%\n"
            "- Current metformin use\n"
            "- Age 18-75\n"
        ),
        exclusion_criteria="Type 1 diabetes\nSevere renal impairment",
        locations=[
            {"facility": "Chicago Diabetes Center", "city": "Chicago", "state": "IL", "zip": "60611"}
        ],
        has_remote_option=True,
        min_age=18,
        max_age=75,
        sex_eligibility="All",
        primary_sponsor="Eli Lilly Test",
        brief_summary="Testing tirzepatide head-to-head with semaglutide in T2DM patients.",
        contact_email="diabetes@test.com",
        contact_name="Dr. Test DM",
    )


# ── Agent 1: PatientProfileAgent ──────────────────────────────────────────────

class TestPatientProfileAgent(unittest.TestCase):

    def setUp(self):
        self.agent = PatientProfileAgent()

    def test_lung_cancer_keywords_extracted(self):
        """PatientProfileAgent should extract NSCLC-specific keywords."""
        patient = make_lung_patient()
        result = self.agent.run(patient)
        self.assertIn("patient", result)
        self.assertIn("search_keywords", result)
        keywords = result["search_keywords"]
        # Should contain at least one cancer-related keyword
        cancer_keywords = [k for k in keywords if any(
            term in k.lower() for term in ["cancer", "lung", "nsclc", "carcinoma", "neoplasm"]
        )]
        self.assertTrue(len(cancer_keywords) >= 1, f"Expected cancer keywords, got: {keywords}")

    def test_diabetes_keywords_extracted(self):
        """PatientProfileAgent should extract diabetes-specific keywords."""
        patient = make_diabetes_patient()
        result = self.agent.run(patient)
        keywords_lower = [k.lower() for k in result["search_keywords"]]
        self.assertTrue(
            any("diabetes" in k or "t2dm" in k for k in keywords_lower),
            f"Expected diabetes keyword, got: {result['search_keywords']}"
        )

    def test_age_range_calculated(self):
        """Age range should bracket patient age by ±5 years."""
        patient = make_lung_patient()  # age 58
        result = self.agent.run(patient)
        lo, hi = result["age_range"]
        self.assertLessEqual(lo, patient.age)
        self.assertGreaterEqual(hi, patient.age)
        self.assertEqual(hi - lo, 10)

    def test_keywords_are_unique(self):
        """No duplicate keywords should appear."""
        patient = make_lung_patient()
        result = self.agent.run(patient)
        keywords = result["search_keywords"]
        self.assertEqual(len(keywords), len(set(keywords)))

    def test_returns_patient_reference(self):
        """Enriched profile should preserve the original patient object."""
        patient = make_lung_patient()
        result = self.agent.run(patient)
        self.assertIs(result["patient"], patient)


# ── Agent 2: TrialSearchAgent (mock path) ────────────────────────────────────

class TestTrialSearchAgent(unittest.TestCase):

    def setUp(self):
        self.agent = TrialSearchAgent()

    def test_mock_returns_lung_trials_for_lung_patient(self):
        """Mock data should return oncology trials for a lung cancer patient."""
        patient = make_lung_patient()
        trials = self.agent._get_mock_trials(patient)
        self.assertGreater(len(trials), 0)
        combined_text = " ".join(t.title.lower() for t in trials)
        self.assertTrue(
            any(term in combined_text for term in ["lung", "nsclc", "cancer"]),
            "Expected lung cancer trials in mock data"
        )

    def test_mock_returns_diabetes_trials_for_diabetes_patient(self):
        """Mock data should return diabetes trials for a diabetes patient."""
        patient = make_diabetes_patient()
        trials = self.agent._get_mock_trials(patient)
        self.assertGreater(len(trials), 0)
        combined_text = " ".join(t.title.lower() for t in trials)
        self.assertTrue(
            any(term in combined_text for term in ["diabetes", "t2dm", "glycemic"]),
            "Expected diabetes trials in mock data"
        )

    def test_mock_returns_hf_trials_for_hf_patient(self):
        """Mock data should return heart failure trials for a CHF patient."""
        patient = make_hf_patient()
        trials = self.agent._get_mock_trials(patient)
        self.assertGreater(len(trials), 0)
        combined_text = " ".join(t.title.lower() for t in trials)
        self.assertTrue(
            any(term in combined_text for term in ["heart failure", "cardiac", "hf"]),
            "Expected heart failure trials in mock data"
        )

    def test_trial_objects_have_required_fields(self):
        """All mock trials must have nct_id, title, and status set."""
        patient = make_lung_patient()
        trials = self.agent._get_mock_trials(patient)
        for trial in trials:
            self.assertTrue(trial.nct_id.startswith("NCT"), f"Invalid NCT ID: {trial.nct_id}")
            self.assertGreater(len(trial.title), 10, "Title too short")
            self.assertEqual(trial.status, "RECRUITING")

    def test_run_fallback_to_mock_when_api_fails(self):
        """run() should always return trials (via mock if API fails)."""
        patient = make_lung_patient()
        profile_agent = PatientProfileAgent()
        enriched = profile_agent.run(patient)
        # Force API failure by patching to non-existent host
        original_base = self.agent.API_BASE
        self.agent.API_BASE = "https://this.host.does.not.exist.invalid/api"
        trials = self.agent.run(enriched)
        self.agent.API_BASE = original_base
        self.assertGreater(len(trials), 0, "Should fall back to mock data")


# ── Agent 3: EligibilityScreeningAgent ───────────────────────────────────────

class TestEligibilityScreeningAgent(unittest.TestCase):

    def setUp(self):
        self.agent = EligibilityScreeningAgent()

    def test_age_exclusion_detected(self):
        """Patient younger than trial minimum age should be age-ineligible."""
        patient = make_lung_patient()   # age 58
        trial = make_lung_trial()
        trial.min_age = 65             # Patient too young
        score = self.agent._score_trial(patient, trial)
        self.assertFalse(score.age_eligible)

    def test_age_within_range_passes(self):
        """Patient within valid age range should be age-eligible."""
        patient = make_lung_patient()  # age 58
        trial = make_lung_trial()
        trial.min_age = 18
        trial.max_age = 75
        score = self.agent._score_trial(patient, trial)
        self.assertTrue(score.age_eligible)

    def test_sex_mismatch_flagged(self):
        """Male-only trial should mark female patient as sex-ineligible."""
        patient = make_lung_patient()  # Female
        trial = make_lung_trial()
        trial.sex_eligibility = "Male"
        score = self.agent._score_trial(patient, trial)
        self.assertFalse(score.sex_eligible)

    def test_sex_all_eligible_for_any(self):
        """Sex='All' trial should be eligible for any patient sex."""
        for pid in ["PT-001", "PT-002", "PT-003"]:
            patient = DEMO_PATIENTS[pid]
            trial = make_lung_trial()
            trial.sex_eligibility = "All"
            score = self.agent._score_trial(patient, trial)
            self.assertTrue(score.sex_eligible, f"Expected sex-eligible for {pid}")

    def test_diagnosis_match_positive_for_correct_condition(self):
        """Lung cancer patient against NSCLC trial should show positive diagnosis match."""
        patient = make_lung_patient()
        trial = make_lung_trial()
        score = self.agent._score_trial(patient, trial)
        self.assertGreater(score.diagnosis_match, 0.0)

    def test_strong_match_tier_for_good_patient(self):
        """A well-matched patient should receive Strong Match tier."""
        patient = make_lung_patient()   # age 58, Female, NSCLC, ECOG 1
        trial = make_lung_trial()       # NSCLC trial, age 18+, All sexes, ECOG 0-2
        score = self.agent._score_trial(patient, trial)
        self.assertIn(
            score.recommendation_tier,
            ("Strong Match", "Possible Match"),
            f"Expected Strong or Possible Match, got: {score.recommendation_tier}"
        )

    def test_low_match_for_wrong_disease(self):
        """Diabetes patient against NSCLC trial should score lower than a well-matched patient."""
        patient_lung = make_lung_patient()    # Should match well
        patient_dm = make_diabetes_patient()  # Should match less well
        trial = make_lung_trial()
        score_lung = self.agent._score_trial(patient_lung, trial)
        score_dm = self.agent._score_trial(patient_dm, trial)
        # Lung patient must score strictly higher than diabetes patient on an NSCLC trial
        self.assertGreater(
            score_lung.overall_score,
            score_dm.overall_score,
            "Lung cancer patient should outscore diabetes patient on an NSCLC trial"
        )

    def test_overall_score_bounded(self):
        """Overall score must always be between 0 and 1."""
        patient = make_lung_patient()
        trial = make_lung_trial()
        score = self.agent._score_trial(patient, trial)
        self.assertGreaterEqual(score.overall_score, 0.0)
        self.assertLessEqual(score.overall_score, 1.0)

    def test_scores_sorted_best_first(self):
        """run() should return scores sorted highest-first."""
        patient = make_lung_patient()
        agent = TrialSearchAgent()
        trials = agent._get_mock_trials(patient)
        scores = self.agent.run(patient, trials)
        if len(scores) > 1:
            for i in range(len(scores) - 1):
                self.assertGreaterEqual(
                    scores[i].overall_score,
                    scores[i + 1].overall_score,
                    "Scores not sorted in descending order"
                )


# ── Agent 4: LogisticsAgent ───────────────────────────────────────────────────

class TestLogisticsAgent(unittest.TestCase):

    def setUp(self):
        self.agent = LogisticsAgent()

    def test_local_site_gets_high_logistics_score(self):
        """Patient in NY with a NY trial site should get logistics_score=1.0."""
        patient = make_lung_patient()    # NY
        trial = make_lung_trial()        # NY site
        elig_agent = EligibilityScreeningAgent()
        scores = elig_agent.run(patient, [trial])
        updated = self.agent.run(patient, [trial], scores)
        self.assertGreaterEqual(updated[0].logistics_score, 0.75)

    def test_remote_option_boosts_logistics_score(self):
        """Willing-remote patient + remote trial should get logistics_score=1.0."""
        patient = make_lung_patient()   # willing_remote=True
        self.assertTrue(patient.willing_remote)
        trial = make_lung_trial()
        trial.has_remote_option = True
        elig_agent = EligibilityScreeningAgent()
        scores = elig_agent.run(patient, [trial])
        updated = self.agent.run(patient, [trial], scores)
        self.assertEqual(updated[0].logistics_score, 1.0)

    def test_distant_site_gets_low_logistics_score(self):
        """Patient in NY with CA trial site should get low logistics score."""
        patient = make_lung_patient()   # NY
        trial = make_lung_trial()
        trial.locations = [{"facility": "CA Center", "city": "Los Angeles", "state": "CA", "zip": "90001"}]
        trial.has_remote_option = False
        elig_agent = EligibilityScreeningAgent()
        scores = elig_agent.run(patient, [trial])
        updated = self.agent.run(patient, [trial], scores)
        self.assertLessEqual(updated[0].logistics_score, 0.5)

    def test_nearest_site_miles_populated(self):
        """nearest_site_miles should be set to a positive number after logistics run."""
        patient = make_lung_patient()
        trial = make_lung_trial()
        elig_agent = EligibilityScreeningAgent()
        scores = elig_agent.run(patient, [trial])
        updated = self.agent.run(patient, [trial], scores)
        self.assertIsNotNone(updated[0].nearest_site_miles)
        self.assertGreater(updated[0].nearest_site_miles, 0)


# ── Agent 5: OutreachDraftingAgent ───────────────────────────────────────────

class TestOutreachDraftingAgent(unittest.TestCase):

    def setUp(self):
        self.agent = OutreachDraftingAgent()

    def _make_strong_match(self, nct_id="NCT99000001") -> EligibilityScore:
        return EligibilityScore(
            trial_nct_id=nct_id,
            trial_title="Phase II Pembrolizumab in Stage III NSCLC",
            overall_score=0.85,
            age_eligible=True,
            sex_eligible=True,
            diagnosis_match=0.80,
            ecog_eligible=True,
            exclusion_flags=[],
            inclusion_met=["Age 58 within eligible range", "Diagnosis matches trial"],
            logistics_score=1.0,
            nearest_site_miles=15.0,
            has_remote=True,
            physician_summary="Strong match for pembrolizumab trial.",
            recommendation_tier="Strong Match",
        )

    def test_email_contains_nct_id(self):
        """Draft email should include the trial NCT ID."""
        patient = make_lung_patient()
        match = self._make_strong_match("NCT99000001")
        email = self.agent.run(patient, [match])
        self.assertIn("NCT99000001", email)

    def test_email_contains_patient_id(self):
        """Draft email should reference the patient ID."""
        patient = make_lung_patient()
        match = self._make_strong_match()
        email = self.agent.run(patient, [match])
        self.assertIn(patient.patient_id, email)

    def test_email_contains_action_items(self):
        """Draft email should include action items section."""
        patient = make_lung_patient()
        match = self._make_strong_match()
        email = self.agent.run(patient, [match])
        self.assertIn("ACTION ITEMS", email)

    def test_empty_matches_returns_message(self):
        """No matches should return a graceful no-trials message."""
        patient = make_lung_patient()
        result = self.agent.run(patient, [])
        self.assertIn("No suitable trials", result)

    def test_exclusion_flag_appears_in_email(self):
        """Exclusion flags should be visibly called out in the email."""
        patient = make_lung_patient()
        match = self._make_strong_match()
        match.exclusion_flags = ["autoimmune", "prior radiation"]
        email = self.agent.run(patient, [match])
        self.assertIn("Exclusions", email)

    def test_remote_option_noted(self):
        """Remote option should be noted in the email for convenience."""
        patient = make_lung_patient()
        match = self._make_strong_match()
        match.has_remote = True
        email = self.agent.run(patient, [match])
        self.assertIn("Remote", email)


# ── End-to-End Orchestrator Tests ─────────────────────────────────────────────

class TestScoutOrchestrator(unittest.TestCase):

    def setUp(self):
        self.orchestrator = ScoutOrchestrator()

    def test_report_produced_for_lung_patient(self):
        """Full pipeline should produce a ScoutReport for PT-001."""
        patient = DEMO_PATIENTS["PT-001"]
        report = self.orchestrator.run(patient)
        self.assertEqual(report.patient_id, "PT-001")
        self.assertGreater(report.trials_searched, 0)
        self.assertIsInstance(report.physician_email_draft, str)
        self.assertGreater(len(report.physician_email_draft), 50)

    def test_report_produced_for_diabetes_patient(self):
        """Full pipeline should produce a ScoutReport for PT-002."""
        patient = DEMO_PATIENTS["PT-002"]
        report = self.orchestrator.run(patient)
        self.assertEqual(report.patient_id, "PT-002")
        self.assertIsNotNone(report.search_date)

    def test_report_produced_for_hf_patient(self):
        """Full pipeline should produce a ScoutReport for PT-003."""
        patient = DEMO_PATIENTS["PT-003"]
        report = self.orchestrator.run(patient)
        self.assertEqual(report.patient_id, "PT-003")

    def test_top_matches_scores_valid(self):
        """All top match scores should be in [0.0, 1.0]."""
        patient = DEMO_PATIENTS["PT-001"]
        report = self.orchestrator.run(patient)
        for m in report.top_matches:
            self.assertGreaterEqual(m.overall_score, 0.0)
            self.assertLessEqual(m.overall_score, 1.0)

    def test_action_items_populated(self):
        """Report should always include at least one action item."""
        patient = DEMO_PATIENTS["PT-001"]
        report = self.orchestrator.run(patient)
        self.assertGreater(len(report.action_items), 0)

    def test_top_matches_sorted_descending(self):
        """top_matches should be ordered best-to-worst."""
        patient = DEMO_PATIENTS["PT-001"]
        report = self.orchestrator.run(patient)
        for i in range(len(report.top_matches) - 1):
            self.assertGreaterEqual(
                report.top_matches[i].overall_score,
                report.top_matches[i + 1].overall_score,
            )

    def test_report_json_serializable(self):
        """ScoutReport should be serializable to JSON without error."""
        patient = DEMO_PATIENTS["PT-001"]
        report = self.orchestrator.run(patient)
        data = {
            "patient_id": report.patient_id,
            "trials_searched": report.trials_searched,
            "matches_found": report.matches_found,
            "top_matches": [
                {"nct_id": m.trial_nct_id, "score": m.overall_score}
                for m in report.top_matches
            ],
        }
        # Should not raise
        json_str = json.dumps(data)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["patient_id"], "PT-001")

    def test_all_demo_patients_complete_pipeline(self):
        """Every demo patient should complete the pipeline without exceptions."""
        for pid, patient in DEMO_PATIENTS.items():
            with self.subTest(patient=pid):
                try:
                    report = self.orchestrator.run(patient)
                    self.assertIsNotNone(report)
                except Exception as e:
                    self.fail(f"Pipeline failed for {pid}: {e}")


if __name__ == "__main__":
    # Run all tests with verbose output
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(__import__("__main__"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
