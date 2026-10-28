#!/usr/bin/env python3
"""
Clinical Trial Scout Agent
==========================
An agentic system that autonomously matches patients to open clinical trials
using the ClinicalTrials.gov v2 API with multi-step eligibility reasoning.

Each "agent" in the pipeline is a focused reasoning unit that transforms
patient data and trial data through a specific analytical lens before
passing results downstream.

Pipeline:
  PatientProfileAgent → TrialSearchAgent → EligibilityScreeningAgent
  → LogisticsAgent → OutreachDraftingAgent → ReportAgent
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict, field
from typing import Optional
import urllib.request
import urllib.parse
import urllib.error


# ─────────────────────────────────────────────
# DATA MODELS
# ─────────────────────────────────────────────

@dataclass
class PatientProfile:
    """Realistic de-identified patient profile used for trial matching."""
    patient_id: str
    age: int
    sex: str                         # "Male" | "Female"
    diagnoses: list[str]             # ICD-10 descriptions
    medications: list[str]           # Current medications
    ecog_status: int                 # 0=fully active, 1-2=restricted, 3-4=limited
    zip_code: str
    city: str
    state: str
    prior_treatments: list[str]
    comorbidities: list[str]
    willing_remote: bool
    notes: str = ""


@dataclass
class ClinicalTrial:
    """Parsed trial record from ClinicalTrials.gov."""
    nct_id: str
    title: str
    status: str
    phase: str
    conditions: list[str]
    interventions: list[str]
    inclusion_criteria: str
    exclusion_criteria: str
    locations: list[dict]
    has_remote_option: bool
    min_age: Optional[int]
    max_age: Optional[int]
    sex_eligibility: str             # "All" | "Male" | "Female"
    primary_sponsor: str
    brief_summary: str
    contact_email: str
    contact_name: str


@dataclass
class EligibilityScore:
    """Structured eligibility assessment for one patient ↔ trial pair."""
    trial_nct_id: str
    trial_title: str
    overall_score: float             # 0.0–1.0 composite
    age_eligible: bool
    sex_eligible: bool
    diagnosis_match: float           # 0.0–1.0 keyword overlap
    ecog_eligible: bool
    exclusion_flags: list[str]       # Specific exclusion concerns
    inclusion_met: list[str]         # Inclusion criteria met
    logistics_score: float           # 0.0–1.0 (lower burden = higher)
    nearest_site_miles: Optional[float]
    has_remote: bool
    physician_summary: str
    recommendation_tier: str         # "Strong Match" | "Possible Match" | "Low Match"


@dataclass
class ScoutReport:
    """Final report produced by the ReportAgent."""
    patient_id: str
    search_date: str
    trials_searched: int
    matches_found: int
    top_matches: list[EligibilityScore]
    physician_email_draft: str
    action_items: list[str]


# ─────────────────────────────────────────────
# MOCK PATIENT PROFILES  (realistic demo data)
# ─────────────────────────────────────────────

DEMO_PATIENTS = {
    "PT-001": PatientProfile(
        patient_id="PT-001",
        age=58,
        sex="Female",
        diagnoses=["Non-small cell lung cancer, stage IIIA", "Hypertension"],
        medications=["Carboplatin", "Paclitaxel", "Lisinopril"],
        ecog_status=1,
        zip_code="10032",
        city="New York",
        state="NY",
        prior_treatments=["Chemotherapy (first-line carboplatin/paclitaxel)"],
        comorbidities=["Hypertension"],
        willing_remote=True,
        notes="Patient completed first-line chemo. Oncologist exploring second-line options."
    ),
    "PT-002": PatientProfile(
        patient_id="PT-002",
        age=45,
        sex="Male",
        diagnoses=["Type 2 Diabetes Mellitus", "Obesity (BMI 34)"],
        medications=["Metformin", "Atorvastatin"],
        ecog_status=0,
        zip_code="60601",
        city="Chicago",
        state="IL",
        prior_treatments=["Diet modification", "Metformin monotherapy for 3 years"],
        comorbidities=["Hyperlipidemia"],
        willing_remote=True,
        notes="HbA1c 8.2% — suboptimal glycemic control. Interested in novel therapies."
    ),
    "PT-003": PatientProfile(
        patient_id="PT-003",
        age=72,
        sex="Male",
        diagnoses=["Chronic Heart Failure (EF 35%)", "Atrial Fibrillation"],
        medications=["Carvedilol", "Entresto", "Apixaban", "Furosemide"],
        ecog_status=2,
        zip_code="77001",
        city="Houston",
        state="TX",
        prior_treatments=["Guideline-directed medical therapy", "ICD implantation"],
        comorbidities=["Chronic Kidney Disease Stage 3", "Atrial Fibrillation"],
        willing_remote=False,
        notes="Stable HF but declining functional capacity over 6 months."
    ),
}


# ─────────────────────────────────────────────
# AGENT 1 — PatientProfileAgent
# ─────────────────────────────────────────────

class PatientProfileAgent:
    """
    Normalises and enriches the patient profile.
    Converts diagnoses to search keywords used by the TrialSearchAgent.
    """

    DIAGNOSIS_KEYWORD_MAP = {
        "non-small cell lung cancer": ["lung cancer", "NSCLC", "lung neoplasm"],
        "type 2 diabetes mellitus":   ["diabetes", "type 2 diabetes", "T2DM"],
        "chronic heart failure":      ["heart failure", "cardiac failure", "HF reduced EF"],
        "obesity":                    ["obesity", "overweight"],
        "hypertension":               ["hypertension", "high blood pressure"],
        "atrial fibrillation":        ["atrial fibrillation", "AFib"],
        "chronic kidney disease":     ["chronic kidney disease", "CKD", "renal impairment"],
    }

    def run(self, patient: PatientProfile) -> dict:
        print(f"\n[PatientProfileAgent] Processing profile for {patient.patient_id}...")
        keywords = []
        for dx in patient.diagnoses:
            dx_lower = dx.lower()
            for key, synonyms in self.DIAGNOSIS_KEYWORD_MAP.items():
                if key in dx_lower:
                    keywords.extend(synonyms)
                    break
            else:
                # Fallback: use first 3 significant words
                words = [w for w in dx.split() if len(w) > 3][:3]
                keywords.extend(words)

        enriched = {
            "patient": patient,
            "search_keywords": list(set(keywords)),
            "age_range": (max(0, patient.age - 5), patient.age + 5),
        }
        print(f"  ✓ Search keywords derived: {enriched['search_keywords'][:5]}")
        return enriched


# ─────────────────────────────────────────────
# AGENT 2 — TrialSearchAgent
# ─────────────────────────────────────────────

class TrialSearchAgent:
    """
    Queries ClinicalTrials.gov v2 REST API for open interventional trials.
    Falls back to realistic mock data if the API is unreachable.
    """

    API_BASE = "https://clinicaltrials.gov/api/v2/studies"
    MAX_TRIALS = 15

    def run(self, enriched_profile: dict) -> list[ClinicalTrial]:
        keywords = enriched_profile["search_keywords"]
        patient = enriched_profile["patient"]
        query_term = keywords[0] if keywords else enriched_profile["patient"].diagnoses[0]

        print(f"\n[TrialSearchAgent] Searching ClinicalTrials.gov for: '{query_term}'...")
        trials = self._fetch_from_api(query_term, patient.sex)

        if not trials:
            print("  ⚠  API unavailable — loading curated mock trial data")
            trials = self._get_mock_trials(patient)

        print(f"  ✓ Found {len(trials)} candidate trial(s)")
        return trials

    def _fetch_from_api(self, query_term: str, patient_sex: str) -> list[ClinicalTrial]:
        params = {
            "query.cond": query_term,
            "filter.overallStatus": "RECRUITING",
            "filter.studyType": "INTERVENTIONAL",
            "pageSize": self.MAX_TRIALS,
            "format": "json",
        }
        url = f"{self.API_BASE}?{urllib.parse.urlencode(params)}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "ClinicalTrialScoutAgent/1.0"})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read())
            return [self._parse_trial(s) for s in data.get("studies", [])]
        except (urllib.error.URLError, Exception):
            return []

    def _parse_trial(self, study: dict) -> ClinicalTrial:
        proto = study.get("protocolSection", {})
        id_mod = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        desc_mod = proto.get("descriptionModule", {})
        design_mod = proto.get("designModule", {})
        elig_mod = proto.get("eligibilityModule", {})
        contacts_mod = proto.get("contactsLocationsModule", {})
        cond_mod = proto.get("conditionsModule", {})
        intervention_mod = proto.get("armsInterventionsModule", {})
        sponsor_mod = proto.get("sponsorCollaboratorsModule", {})

        # Parse locations
        locations = []
        for loc in contacts_mod.get("locations", [])[:5]:
            locations.append({
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "zip": loc.get("zip", ""),
            })

        # Check for remote/virtual option
        criteria_text = elig_mod.get("eligibilityCriteria", "").lower()
        has_remote = any(word in criteria_text for word in ["remote", "virtual", "telehealth", "home-based"])

        # Parse age
        def parse_age(s):
            if not s:
                return None
            try:
                return int(''.join(filter(str.isdigit, s.split()[0])))
            except (ValueError, IndexError):
                return None

        # Contact info
        central_contacts = contacts_mod.get("centralContacts", [{}])
        contact = central_contacts[0] if central_contacts else {}

        # Interventions
        interventions = [
            i.get("name", "") for i in intervention_mod.get("interventions", [])[:3]
        ]

        return ClinicalTrial(
            nct_id=id_mod.get("nctId", ""),
            title=id_mod.get("officialTitle", id_mod.get("briefTitle", ""))[:120],
            status=status_mod.get("overallStatus", ""),
            phase=", ".join(design_mod.get("phases", ["N/A"])),
            conditions=cond_mod.get("conditions", [])[:5],
            interventions=interventions,
            inclusion_criteria=elig_mod.get("eligibilityCriteria", "")[:1000],
            exclusion_criteria="",  # Embedded in eligibilityCriteria
            locations=locations,
            has_remote_option=has_remote,
            min_age=parse_age(elig_mod.get("minimumAge")),
            max_age=parse_age(elig_mod.get("maximumAge")),
            sex_eligibility=elig_mod.get("sex", "All"),
            primary_sponsor=sponsor_mod.get("leadSponsor", {}).get("name", "")[:80],
            brief_summary=desc_mod.get("briefSummary", "")[:500],
            contact_email=contact.get("email", ""),
            contact_name=contact.get("name", ""),
        )

    def _get_mock_trials(self, patient: PatientProfile) -> list[ClinicalTrial]:
        """Realistic mock trials keyed to each demo patient diagnosis."""
        dx_lower = " ".join(patient.diagnoses).lower()

        if "lung" in dx_lower or "nsclc" in dx_lower:
            return [
                ClinicalTrial(
                    nct_id="NCT05100001",
                    title="Phase II Study of Nivolumab Plus Chemotherapy in Stage III NSCLC After First-Line Platinum Failure",
                    status="RECRUITING",
                    phase="Phase 2",
                    conditions=["Non-Small Cell Lung Carcinoma", "Stage III Lung Cancer"],
                    interventions=["Nivolumab 360mg IV", "Carboplatin AUC5", "Pemetrexed 500mg/m2"],
                    inclusion_criteria="Inclusion Criteria:\n- Histologically confirmed NSCLC\n- Stage IIIA or IIIB\n- Prior platinum-based chemotherapy\n- ECOG performance status 0-2\n- Age >= 18 years\n- Adequate organ function",
                    exclusion_criteria="Exclusion Criteria:\n- Active autoimmune disease\n- Prior checkpoint inhibitor therapy\n- Symptomatic brain metastases",
                    locations=[
                        {"facility": "Memorial Cancer Center", "city": "New York", "state": "NY", "zip": "10065"},
                        {"facility": "Columbia University Medical Center", "city": "New York", "state": "NY", "zip": "10032"},
                    ],
                    has_remote_option=True,
                    min_age=18,
                    max_age=None,
                    sex_eligibility="All",
                    primary_sponsor="National Cancer Institute",
                    brief_summary="This trial evaluates the safety and efficacy of nivolumab combined with chemotherapy in patients with stage III NSCLC who have progressed after first-line platinum-based chemotherapy.",
                    contact_email="trials@memorialcancer.org",
                    contact_name="Dr. Sarah Chen",
                ),
                ClinicalTrial(
                    nct_id="NCT05100002",
                    title="KRAS G12C Inhibitor AMG-510 (Sotorasib) in Metastatic NSCLC: Expanded Access Protocol",
                    status="RECRUITING",
                    phase="Phase 3",
                    conditions=["Non-Small Cell Lung Cancer", "KRAS G12C Mutation"],
                    interventions=["Sotorasib 960mg oral daily"],
                    inclusion_criteria="Inclusion Criteria:\n- Advanced or metastatic NSCLC\n- Documented KRAS G12C mutation\n- Prior chemotherapy and/or immunotherapy\n- ECOG 0-2",
                    exclusion_criteria="Active untreated CNS metastases",
                    locations=[
                        {"facility": "NYU Perlmutter Cancer Center", "city": "New York", "state": "NY", "zip": "10016"},
                    ],
                    has_remote_option=True,
                    min_age=18,
                    max_age=None,
                    sex_eligibility="All",
                    primary_sponsor="Amgen Inc.",
                    brief_summary="Expanded access trial for sotorasib (AMG-510) targeting the KRAS G12C mutation in advanced NSCLC patients who have exhausted standard options.",
                    contact_email="amgen.trials@nyu.edu",
                    contact_name="Dr. Michael Torres",
                ),
                ClinicalTrial(
                    nct_id="NCT05100003",
                    title="Stereotactic Ablative Radiotherapy (SABR) vs Chemotherapy in Stage III NSCLC (SAFRON III)",
                    status="RECRUITING",
                    phase="Phase 3",
                    conditions=["Non-Small Cell Lung Cancer"],
                    interventions=["SABR 54Gy/3 fractions", "Standard chemotherapy"],
                    inclusion_criteria="Inclusion Criteria:\n- Stage IIIA NSCLC\n- Age 18+\n- ECOG 0-1\n- Tumor <= 5cm\n- No prior radiation",
                    exclusion_criteria="Prior thoracic radiation\nActive second malignancy",
                    locations=[
                        {"facility": "Boston Medical Center", "city": "Boston", "state": "MA", "zip": "02118"},
                    ],
                    has_remote_option=False,
                    min_age=18,
                    max_age=80,
                    sex_eligibility="All",
                    primary_sponsor="Radiation Therapy Oncology Group",
                    brief_summary="Randomized comparison of SABR versus standard chemotherapy as consolidation treatment following chemoradiotherapy in stage III NSCLC.",
                    contact_email="safron3@rtog.org",
                    contact_name="Dr. Linda Wu",
                ),
            ]
        elif "diabetes" in dx_lower:
            return [
                ClinicalTrial(
                    nct_id="NCT05200001",
                    title="GLP-1 Agonist Tirzepatide vs Semaglutide for Glycemic Control in T2DM Inadequately Controlled on Metformin",
                    status="RECRUITING",
                    phase="Phase 4",
                    conditions=["Type 2 Diabetes Mellitus", "Obesity"],
                    interventions=["Tirzepatide 5mg/10mg/15mg weekly", "Semaglutide 1mg weekly"],
                    inclusion_criteria="Inclusion Criteria:\n- Type 2 diabetes diagnosis >= 1 year\n- HbA1c 7.5-11%\n- Current metformin use >= 3 months\n- Age 18-75\n- BMI 27-45 kg/m2",
                    exclusion_criteria="Type 1 diabetes\nPrior GLP-1 agonist use\nSevere renal impairment (eGFR < 30)",
                    locations=[
                        {"facility": "Northwestern Medicine Diabetes Center", "city": "Chicago", "state": "IL", "zip": "60611"},
                        {"facility": "University of Illinois Hospital", "city": "Chicago", "state": "IL", "zip": "60612"},
                    ],
                    has_remote_option=True,
                    min_age=18,
                    max_age=75,
                    sex_eligibility="All",
                    primary_sponsor="Eli Lilly and Company",
                    brief_summary="Head-to-head comparison of tirzepatide versus semaglutide in patients with T2DM suboptimally controlled on metformin monotherapy, measuring HbA1c reduction and weight loss at 52 weeks.",
                    contact_email="diabetes.trials@northwestern.edu",
                    contact_name="Dr. Aisha Patel",
                ),
                ClinicalTrial(
                    nct_id="NCT05200002",
                    title="Digital Therapeutics Plus CGM for Lifestyle Intervention in T2DM: The ENGAGE-DM Trial",
                    status="RECRUITING",
                    phase="Phase 3",
                    conditions=["Type 2 Diabetes", "Obesity"],
                    interventions=["Continuous Glucose Monitor", "Digital Coaching App", "Metformin continuation"],
                    inclusion_criteria="Inclusion Criteria:\n- Diagnosed T2DM\n- HbA1c 7.0-10.0%\n- Smartphone ownership\n- Age 21-70",
                    exclusion_criteria="Insulin therapy\nNon-English speaking",
                    locations=[
                        {"facility": "Rush University Medical Center", "city": "Chicago", "state": "IL", "zip": "60612"},
                    ],
                    has_remote_option=True,
                    min_age=21,
                    max_age=70,
                    sex_eligibility="All",
                    primary_sponsor="NIDDK",
                    brief_summary="Randomized trial testing whether continuous glucose monitoring combined with an AI-driven digital therapeutic app improves glycemic control compared to usual care in adults with T2DM.",
                    contact_email="engage.dm@rush.edu",
                    contact_name="Dr. James Park",
                ),
            ]
        else:  # Heart failure default
            return [
                ClinicalTrial(
                    nct_id="NCT05300001",
                    title="Sacubitril/Valsartan vs Standard Care in HFrEF Patients Hospitalized for Acute Decompensation",
                    status="RECRUITING",
                    phase="Phase 4",
                    conditions=["Heart Failure with Reduced Ejection Fraction", "Cardiac Decompensation"],
                    interventions=["Sacubitril/Valsartan 97/103mg twice daily", "Standard ACE inhibitor therapy"],
                    inclusion_criteria="Inclusion Criteria:\n- Chronic HF with EF <= 40%\n- Age >= 18\n- Prior ACE inhibitor or ARB tolerance\n- BNP > 150 pg/mL or NT-proBNP > 600 pg/mL",
                    exclusion_criteria="eGFR < 30\nPotassium > 5.4 mEq/L\nSymptomatic hypotension (SBP < 100)",
                    locations=[
                        {"facility": "Houston Methodist Hospital", "city": "Houston", "state": "TX", "zip": "77030"},
                        {"facility": "Texas Heart Institute", "city": "Houston", "state": "TX", "zip": "77030"},
                    ],
                    has_remote_option=False,
                    min_age=18,
                    max_age=None,
                    sex_eligibility="All",
                    primary_sponsor="Novartis Pharmaceuticals",
                    brief_summary="Prospective randomized trial evaluating in-hospital initiation of sacubitril/valsartan versus continuation of standard ACE inhibitor therapy in patients with HFrEF hospitalized for acute decompensated heart failure.",
                    contact_email="hf.trials@houstonmethodist.org",
                    contact_name="Dr. Carlos Rivera",
                ),
                ClinicalTrial(
                    nct_id="NCT05300002",
                    title="Remote Monitoring and Telehealth Management for Chronic Heart Failure: REMOTE-HF Trial",
                    status="RECRUITING",
                    phase="Phase 3",
                    conditions=["Chronic Heart Failure", "Heart Failure Reduced Ejection Fraction"],
                    interventions=["CardioMEMS implantable PA pressure sensor", "Telehealth nurse case management"],
                    inclusion_criteria="Inclusion Criteria:\n- NYHA Class III HF\n- EF <= 45%\n- Prior HF hospitalization in 12 months\n- Age >= 40",
                    exclusion_criteria="Life expectancy < 6 months\nActive infection\nSevere mitral stenosis",
                    locations=[
                        {"facility": "Baylor St. Luke's Medical Center", "city": "Houston", "state": "TX", "zip": "77030"},
                    ],
                    has_remote_option=True,
                    min_age=40,
                    max_age=None,
                    sex_eligibility="All",
                    primary_sponsor="Abbott Laboratories",
                    brief_summary="Randomized trial of remote hemodynamic monitoring plus structured telehealth management versus usual care in patients with chronic heart failure to reduce 6-month hospitalization rates.",
                    contact_email="remote.hf@bcm.edu",
                    contact_name="Dr. Priya Gupta",
                ),
            ]


# ─────────────────────────────────────────────
# AGENT 3 — EligibilityScreeningAgent
# ─────────────────────────────────────────────

class EligibilityScreeningAgent:
    """
    Scores each trial against the patient profile using structured criteria matching.
    Identifies exclusion flags and generates a physician-readable eligibility summary.
    """

    EXCLUSION_TRIGGERS = {
        "autoimmune":        ["autoimmune", "lupus", "rheumatoid", "crohn"],
        "active infection":  ["active infection", "sepsis", "hiv", "hepatitis"],
        "prior radiation":   ["prior radiation", "prior thoracic radiation"],
        "renal impairment":  ["renal impairment", "egfr < 30", "creatinine"],
        "liver disease":     ["hepatic impairment", "cirrhosis", "liver disease"],
        "pregnancy":         ["pregnant", "nursing", "breastfeeding"],
        "brain metastases":  ["brain metastases", "cns metastases", "brain mets"],
    }

    def run(self, patient: PatientProfile, trials: list[ClinicalTrial]) -> list[EligibilityScore]:
        print(f"\n[EligibilityScreeningAgent] Screening {len(trials)} trial(s) for {patient.patient_id}...")
        scores = []
        for trial in trials:
            score = self._score_trial(patient, trial)
            scores.append(score)
            print(f"  → {trial.nct_id}: {score.overall_score:.0%} match — {score.recommendation_tier}")

        # Sort best matches first
        scores.sort(key=lambda s: s.overall_score, reverse=True)
        return scores

    def _score_trial(self, patient: PatientProfile, trial: ClinicalTrial) -> EligibilityScore:
        # 1. Age eligibility
        age_eligible = True
        if trial.min_age is not None and patient.age < trial.min_age:
            age_eligible = False
        if trial.max_age is not None and patient.age > trial.max_age:
            age_eligible = False

        # 2. Sex eligibility
        sex_eligible = trial.sex_eligibility in ("All", patient.sex)

        # 3. Diagnosis keyword match (0.0–1.0)
        all_text = (
            trial.title + " " + trial.brief_summary + " " +
            " ".join(trial.conditions)
        ).lower()
        patient_dx_words = set()
        for dx in patient.diagnoses:
            patient_dx_words.update(dx.lower().split())
        dx_matches = sum(1 for w in patient_dx_words if len(w) > 4 and w in all_text)
        dx_score = min(1.0, dx_matches / max(1, len(patient_dx_words) * 0.3))

        # 4. ECOG eligibility (look for ECOG mention in inclusion criteria)
        ecog_eligible = True
        inc_text = trial.inclusion_criteria.lower()
        for allowed_ecog in range(3):
            if f"ecog {allowed_ecog}" in inc_text or f"ecog performance status {allowed_ecog}" in inc_text:
                if patient.ecog_status > allowed_ecog + 1:
                    ecog_eligible = False
                break

        # 5. Exclusion flags
        exclusion_flags = []
        patient_text = " ".join(
            patient.diagnoses + patient.comorbidities + patient.medications + patient.prior_treatments
        ).lower()
        exc_text = (trial.inclusion_criteria + " " + trial.exclusion_criteria).lower()
        for flag_name, keywords in self.EXCLUSION_TRIGGERS.items():
            if any(kw in patient_text for kw in keywords) and any(kw in exc_text for kw in keywords):
                exclusion_flags.append(flag_name)

        # 6. Inclusion criteria met (simple heuristic)
        inclusion_met = []
        if age_eligible:
            inclusion_met.append(f"Age {patient.age} within eligible range")
        if sex_eligible:
            inclusion_met.append(f"Sex eligibility ({patient.sex}) satisfied")
        if ecog_eligible:
            inclusion_met.append(f"ECOG status {patient.ecog_status} within range")
        if dx_score > 0.3:
            inclusion_met.append("Diagnosis matches trial condition focus")

        # 7. Logistics score (set by LogisticsAgent later; default here)
        logistics_score = 0.5

        # 8. Composite score
        weights = {"age": 0.20, "sex": 0.10, "dx": 0.35, "ecog": 0.20, "excl": 0.15}
        excl_penalty = min(1.0, len(exclusion_flags) * 0.4)
        raw = (
            weights["age"] * (1 if age_eligible else 0) +
            weights["sex"] * (1 if sex_eligible else 0) +
            weights["dx"] * dx_score +
            weights["ecog"] * (1 if ecog_eligible else 0) +
            weights["excl"] * (1 - excl_penalty)
        )
        overall = max(0.0, min(1.0, raw))

        # 9. Tier assignment
        if overall >= 0.70 and not exclusion_flags:
            tier = "Strong Match"
        elif overall >= 0.45:
            tier = "Possible Match"
        else:
            tier = "Low Match"

        # 10. Physician summary
        summary_parts = [
            f"Trial {trial.nct_id} ({trial.phase}) — {tier}.",
            f"Eligibility: Age {'✓' if age_eligible else '✗'}, Sex {'✓' if sex_eligible else '✗'}, ECOG {'✓' if ecog_eligible else '✗'}.",
        ]
        if exclusion_flags:
            summary_parts.append(f"⚠ Exclusion concerns: {', '.join(exclusion_flags)}.")
        if inclusion_met:
            summary_parts.append(f"Criteria met: {'; '.join(inclusion_met[:3])}.")

        return EligibilityScore(
            trial_nct_id=trial.nct_id,
            trial_title=trial.title,
            overall_score=overall,
            age_eligible=age_eligible,
            sex_eligible=sex_eligible,
            diagnosis_match=dx_score,
            ecog_eligible=ecog_eligible,
            exclusion_flags=exclusion_flags,
            inclusion_met=inclusion_met,
            logistics_score=logistics_score,
            nearest_site_miles=None,
            has_remote=trial.has_remote_option,
            physician_summary=" ".join(summary_parts),
            recommendation_tier=tier,
        )


# ─────────────────────────────────────────────
# AGENT 4 — LogisticsAgent
# ─────────────────────────────────────────────

class LogisticsAgent:
    """
    Assesses travel burden by estimating distance from patient ZIP to trial sites.
    Uses a state-based distance heuristic (no paid API needed).
    Updates logistics_score and nearest_site_miles on each EligibilityScore.
    """

    # Approximate inter-state distances (miles) from key origin states
    STATE_DISTANCE_TABLE = {
        ("NY", "NY"): 15,   ("NY", "NJ"): 40,   ("NY", "CT"): 80,
        ("NY", "MA"): 220,  ("NY", "PA"): 100,  ("NY", "IL"): 800,
        ("NY", "TX"): 1600, ("NY", "CA"): 2800,
        ("IL", "IL"): 15,   ("IL", "WI"): 100,  ("IL", "IN"): 160,
        ("IL", "OH"): 350,  ("IL", "NY"): 800,  ("IL", "TX"): 1000,
        ("IL", "CA"): 2000,
        ("TX", "TX"): 20,   ("TX", "OK"): 200,  ("TX", "LA"): 340,
        ("TX", "NY"): 1600, ("TX", "IL"): 1000, ("TX", "CA"): 1400,
        ("TX", "MA"): 1800,
    }
    DEFAULT_DISTANCE = 500  # miles, for unlisted combinations

    def run(
        self,
        patient: PatientProfile,
        trials: list[ClinicalTrial],
        scores: list[EligibilityScore],
    ) -> list[EligibilityScore]:
        print(f"\n[LogisticsAgent] Calculating travel burden for {patient.patient_id}...")
        score_map = {s.trial_nct_id: s for s in scores}
        trial_map = {t.nct_id: t for t in trials}

        for nct_id, score in score_map.items():
            trial = trial_map.get(nct_id)
            if not trial:
                continue

            # Find nearest site
            min_dist = None
            for loc in trial.locations:
                site_state = loc.get("state", "").upper()
                dist = self.STATE_DISTANCE_TABLE.get(
                    (patient.state.upper(), site_state),
                    self.DEFAULT_DISTANCE,
                )
                if min_dist is None or dist < min_dist:
                    min_dist = dist

            score.nearest_site_miles = min_dist or self.DEFAULT_DISTANCE

            # Logistics score: <50mi=1.0, <200mi=0.75, <500mi=0.5, <1000mi=0.25, >=1000=0.1
            if score.has_remote and patient.willing_remote:
                score.logistics_score = 1.0
                logistics_note = "Remote participation available ✓"
            elif min_dist is not None:
                if min_dist < 50:
                    score.logistics_score = 1.0
                    logistics_note = f"Site {min_dist:.0f} miles away (local) ✓"
                elif min_dist < 200:
                    score.logistics_score = 0.75
                    logistics_note = f"Site {min_dist:.0f} miles away (manageable)"
                elif min_dist < 500:
                    score.logistics_score = 0.50
                    logistics_note = f"Site {min_dist:.0f} miles away (requires travel)"
                elif min_dist < 1000:
                    score.logistics_score = 0.25
                    logistics_note = f"Site {min_dist:.0f} miles away (significant travel)"
                else:
                    score.logistics_score = 0.10
                    logistics_note = f"Site {min_dist:.0f} miles away (major travel burden)"
            else:
                score.logistics_score = 0.3
                logistics_note = "Site location unknown"

            # Incorporate logistics into overall score (10% weight adjustment)
            score.overall_score = round(
                min(1.0, score.overall_score * 0.9 + score.logistics_score * 0.1), 3
            )
            score.physician_summary += f" Logistics: {logistics_note}."
            print(f"  → {nct_id}: nearest site {score.nearest_site_miles:.0f} mi | logistics score {score.logistics_score:.2f}")

        return scores


# ─────────────────────────────────────────────
# AGENT 5 — OutreachDraftingAgent
# ─────────────────────────────────────────────

class OutreachDraftingAgent:
    """
    Composes a physician-ready email summarizing the top trial matches.
    The email is structured for immediate action — physicians can forward
    it to their patient coordinator with one click.
    """

    def run(self, patient: PatientProfile, top_matches: list[EligibilityScore]) -> str:
        if not top_matches:
            return "[OutreachDraftingAgent] No suitable trials found — no email drafted."

        print(f"\n[OutreachDraftingAgent] Drafting physician email for {patient.patient_id}...")
        strong = [m for m in top_matches if m.recommendation_tier == "Strong Match"]
        possible = [m for m in top_matches if m.recommendation_tier == "Possible Match"]

        lines = [
            "─" * 72,
            "DRAFT EMAIL — Clinical Trial Opportunities for Your Patient",
            "─" * 72,
            "",
            f"To: Treating Physician",
            f"From: Clinical Trial Scout Agent",
            f"Re: Patient {patient.patient_id} — Clinical Trial Matches",
            f"Date: {time.strftime('%B %d, %Y')}",
            "",
            f"Dear Physician,",
            "",
            f"I have identified {len(top_matches)} potential clinical trial(s) for your patient "
            f"(ID: {patient.patient_id}, Age {patient.age}, {patient.sex}) diagnosed with "
            f"{patient.diagnoses[0]}. Below is a ranked summary for your review.",
            "",
        ]

        if strong:
            lines.append("━━ STRONG MATCHES ━━")
            lines.append("")
            for i, m in enumerate(strong, 1):
                lines += self._format_match_block(i, m)

        if possible:
            lines.append("━━ POSSIBLE MATCHES (require chart review) ━━")
            lines.append("")
            for i, m in enumerate(possible, 1):
                lines += self._format_match_block(i, m)

        lines += [
            "",
            "─" * 72,
            "ACTION ITEMS:",
            "  1. Review eligibility criteria at clinicaltrials.gov using NCT IDs above",
            "  2. Discuss top matches with patient at next visit",
            "  3. Contact trial coordinator for pre-screening call",
            "  4. Request full eligibility checklist from study coordinator",
            "",
            "This summary was generated by Clinical Trial Scout Agent.",
            "Always verify eligibility with the trial's official documentation.",
            "─" * 72,
        ]

        return "\n".join(lines)

    def _format_match_block(self, rank: int, m: EligibilityScore) -> list[str]:
        excl_note = f" ⚠ Exclusions: {', '.join(m.exclusion_flags)}" if m.exclusion_flags else ""
        remote_note = " 📡 Remote option available" if m.has_remote else ""
        return [
            f"  [{rank}] {m.trial_nct_id} — Match Score: {m.overall_score:.0%}",
            f"      Title: {m.trial_title[:80]}...",
            f"      {m.recommendation_tier}{excl_note}{remote_note}",
            f"      Nearest site: {m.nearest_site_miles:.0f} miles" if m.nearest_site_miles else "",
            f"      Criteria met: {', '.join(m.inclusion_met[:2]) if m.inclusion_met else 'Review required'}",
            "",
        ]


# ─────────────────────────────────────────────
# ORCHESTRATOR — ScoutOrchestrator
# ─────────────────────────────────────────────

class ScoutOrchestrator:
    """
    Coordinates the full 5-agent pipeline for clinical trial matching.
    Produces a structured ScoutReport with physician-ready outputs.
    """

    def __init__(self):
        self.profile_agent     = PatientProfileAgent()
        self.search_agent      = TrialSearchAgent()
        self.eligibility_agent = EligibilityScreeningAgent()
        self.logistics_agent   = LogisticsAgent()
        self.outreach_agent    = OutreachDraftingAgent()

    def run(self, patient: PatientProfile) -> ScoutReport:
        print("=" * 72)
        print(f"  CLINICAL TRIAL SCOUT AGENT — Patient: {patient.patient_id}")
        print("=" * 72)

        # Step 1: Profile enrichment
        enriched = self.profile_agent.run(patient)

        # Step 2: Trial search
        trials = self.search_agent.run(enriched)

        if not trials:
            print("\n  ⚠  No trials found. Consider broadening search criteria.")
            return ScoutReport(
                patient_id=patient.patient_id,
                search_date=time.strftime("%Y-%m-%d"),
                trials_searched=0,
                matches_found=0,
                top_matches=[],
                physician_email_draft="No trials found.",
                action_items=["Expand search criteria", "Search disease-specific registries"],
            )

        # Step 3: Eligibility scoring
        scores = self.eligibility_agent.run(patient, trials)

        # Step 4: Logistics assessment
        scores = self.logistics_agent.run(patient, trials, scores)

        # Step 5: Filter meaningful matches
        top_matches = [s for s in scores if s.overall_score >= 0.40][:5]

        # Step 6: Draft outreach
        email_draft = self.outreach_agent.run(patient, top_matches)

        # Build action items
        action_items = []
        if top_matches:
            action_items.append(f"Review {len(top_matches)} top trial match(es) with patient at next visit")
        strong = [m for m in top_matches if m.recommendation_tier == "Strong Match"]
        if strong:
            action_items.append(f"Contact coordinator for {strong[0].trial_nct_id} — highest priority match")
        if any(m.exclusion_flags for m in top_matches):
            action_items.append("Clarify exclusion criteria with study coordinators before pre-screening")
        action_items.append("Access full criteria at clinicaltrials.gov")

        return ScoutReport(
            patient_id=patient.patient_id,
            search_date=time.strftime("%Y-%m-%d"),
            trials_searched=len(trials),
            matches_found=len(top_matches),
            top_matches=top_matches,
            physician_email_draft=email_draft,
            action_items=action_items,
        )


# ─────────────────────────────────────────────
# CLI OUTPUT
# ─────────────────────────────────────────────

def print_report(report: ScoutReport) -> None:
    print("\n" + "=" * 72)
    print("  SCOUT REPORT SUMMARY")
    print("=" * 72)
    print(f"  Patient:         {report.patient_id}")
    print(f"  Search date:     {report.search_date}")
    print(f"  Trials searched: {report.trials_searched}")
    print(f"  Matches found:   {report.matches_found}")
    print()

    if report.top_matches:
        print("  TOP TRIAL MATCHES:")
        print("  " + "─" * 68)
        for m in report.top_matches:
            print(f"  {'✅' if m.recommendation_tier == 'Strong Match' else '🔵'} "
                  f"{m.trial_nct_id} | {m.overall_score:.0%} | {m.recommendation_tier}")
            print(f"     {m.trial_title[:65]}...")
            if m.exclusion_flags:
                print(f"     ⚠ Exclusion flags: {', '.join(m.exclusion_flags)}")
            print()

    print("  ACTION ITEMS:")
    for i, item in enumerate(report.action_items, 1):
        print(f"  {i}. {item}")
    print()
    print(report.physician_email_draft)


def save_report_json(report: ScoutReport, output_path: str) -> None:
    """Save report as JSON for downstream systems."""
    data = {
        "patient_id": report.patient_id,
        "search_date": report.search_date,
        "trials_searched": report.trials_searched,
        "matches_found": report.matches_found,
        "top_matches": [
            {
                "nct_id": m.trial_nct_id,
                "title": m.trial_title,
                "score": m.overall_score,
                "tier": m.recommendation_tier,
                "exclusion_flags": m.exclusion_flags,
                "inclusion_met": m.inclusion_met,
                "nearest_site_miles": m.nearest_site_miles,
                "has_remote": m.has_remote,
            }
            for m in report.top_matches
        ],
        "action_items": report.action_items,
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\n  💾 Report saved to {output_path}")


# ─────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Clinical Trial Scout Agent — AI-powered patient-to-trial matching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --patient PT-001          # Lung cancer patient
  python main.py --patient PT-002          # Type 2 diabetes patient
  python main.py --patient PT-003          # Heart failure patient
  python main.py --patient PT-001 --json report.json
  python main.py --list-patients
        """,
    )
    parser.add_argument(
        "--patient",
        choices=list(DEMO_PATIENTS.keys()),
        default="PT-001",
        help="Demo patient ID to match (default: PT-001)",
    )
    parser.add_argument(
        "--json",
        metavar="FILE",
        help="Save structured JSON report to FILE",
    )
    parser.add_argument(
        "--list-patients",
        action="store_true",
        help="Show available demo patients",
    )
    args = parser.parse_args()

    if args.list_patients:
        print("\nAvailable demo patients:\n")
        for pid, p in DEMO_PATIENTS.items():
            print(f"  {pid}: Age {p.age} {p.sex} | {p.diagnoses[0]}")
            print(f"       ECOG {p.ecog_status} | {p.city}, {p.state}")
        print()
        sys.exit(0)

    patient = DEMO_PATIENTS[args.patient]
    orchestrator = ScoutOrchestrator()
    report = orchestrator.run(patient)
    print_report(report)

    if args.json:
        save_report_json(report, args.json)


if __name__ == "__main__":
    main()
