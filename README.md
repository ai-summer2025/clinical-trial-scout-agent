# 🔬 Clinical Trial Scout Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![ClinicalTrials.gov](https://img.shields.io/badge/data-ClinicalTrials.gov-orange.svg)](https://clinicaltrials.gov)
[![Agentic AI](https://img.shields.io/badge/AI-Agentic%20Pipeline-purple.svg)](#architecture)

> **An agentic AI system that autonomously matches patients to open clinical trials — turning a 4-hour manual process into a 10-second automated report.**

---

## The Problem

**85% of clinical trials fail to recruit enough patients.** This costs an estimated **$1.8 billion per delayed drug approval** (Tufts Center for the Study of Drug Development, 2022). Meanwhile, thousands of eligible patients who might benefit from cutting-edge treatments never hear about those trials — because manual matching is complex, time-consuming, and requires specialist expertise most care teams don't have.

A single clinical trial coordinator may spend **3–5 hours per patient** reviewing eligibility criteria, calling trial sites, and drafting physician recommendations. Most give up after checking 2–3 trials. The result: mismatched supply and demand at massive scale, costing patients lives and pharma companies billions.

Current solutions are either expensive enterprise software (Tempus, Flatiron) locked behind six-figure contracts, or basic keyword search tools that return dozens of irrelevant results requiring manual review.

---

## How It Works — Agentic Architecture

Clinical Trial Scout Agent runs a **5-agent sequential reasoning pipeline**, where each agent transforms the data before handing it to the next:

```
Patient Profile
      │
      ▼
┌─────────────────────┐
│ PatientProfileAgent │  ← Normalises diagnoses → search keywords
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ TrialSearchAgent    │  ← Queries ClinicalTrials.gov v2 REST API
└─────────────────────┘       (auto-falls back to curated mock data)
      │
      ▼
┌──────────────────────────┐
│ EligibilityScreeningAgent│  ← Scores age, sex, ECOG, diagnosis match,
└──────────────────────────┘    exclusion flags → composite 0–100% score
      │
      ▼
┌─────────────────────┐
│ LogisticsAgent      │  ← Calculates travel burden by state,
└─────────────────────┘    flags remote-eligible trials
      │
      ▼
┌──────────────────────────┐
│ OutreachDraftingAgent    │  ← Writes physician-ready email with
└──────────────────────────┘    ranked matches + action items
      │
      ▼
  ScoutReport (JSON + CLI)
```

Each agent has a single responsibility, making the system easy to extend, debug, or swap individual components.

---

## Installation

```bash
git clone https://github.com/ai-summer2025/clinical-trial-scout-agent.git
cd clinical-trial-scout-agent

# No external API keys required — uses ClinicalTrials.gov (free, public)
pip install -r requirements.txt
```

---

## Usage

### Run a demo patient match

```bash
# Lung cancer patient (PT-001) — default
python main.py --patient PT-001

# Type 2 diabetes patient (PT-002)
python main.py --patient PT-002

# Chronic heart failure patient (PT-003)
python main.py --patient PT-003
```

### List available demo patients

```bash
python main.py --list-patients
```

Output:
```
Available demo patients:

  PT-001: Age 58 Female | Non-small cell lung cancer, stage IIIA
       ECOG 1 | New York, NY
  PT-002: Age 45 Male | Type 2 Diabetes Mellitus
       ECOG 0 | Chicago, IL
  PT-003: Age 72 Male | Chronic Heart Failure (EF 35%)
       ECOG 2 | Houston, TX
```

### Save structured JSON report

```bash
python main.py --patient PT-001 --json output/report_PT001.json
```

### Example Output (abbreviated)

```
========================================================================
  CLINICAL TRIAL SCOUT AGENT — Patient: PT-001
========================================================================

[PatientProfileAgent] Processing profile for PT-001...
  ✓ Search keywords derived: ['NSCLC', 'lung cancer', 'lung neoplasm']

[TrialSearchAgent] Searching ClinicalTrials.gov for: 'NSCLC'...
  ✓ Found 3 candidate trial(s)

[EligibilityScreeningAgent] Screening 3 trial(s) for PT-001...
  → NCT05100001: 88% match — Strong Match
  → NCT05100002: 75% match — Strong Match
  → NCT05100003: 52% match — Possible Match

[LogisticsAgent] Calculating travel burden for PT-001...
  → NCT05100001: nearest site 15 mi | logistics score 1.00
  → NCT05100002: nearest site 15 mi | logistics score 1.00
  → NCT05100003: nearest site 220 mi | logistics score 0.75

[OutreachDraftingAgent] Drafting physician email for PT-001...

========================================================================
  SCOUT REPORT SUMMARY
========================================================================
  Patient:         PT-001
  Search date:     2026-10-28
  Trials searched: 3
  Matches found:   3

  TOP TRIAL MATCHES:
  ────────────────────────────────────────────────────────────────────
  ✅ NCT05100001 | 88% | Strong Match
     Phase II Study of Nivolumab Plus Chemotherapy in Stage III NSCLC...

  ✅ NCT05100002 | 75% | Strong Match
     KRAS G12C Inhibitor AMG-510 (Sotorasib) in Metastatic NSCLC...

  🔵 NCT05100003 | 52% | Possible Match
     Stereotactic Ablative Radiotherapy (SABR) vs Chemotherapy...
```

---

## Why This Matters

| Metric | Current State | With Scout Agent |
|---|---|---|
| Time to identify top 3 trials | 3–5 hours/patient | < 10 seconds |
| Trials reviewed per coordinator | 2–3 per session | All recruiting trials |
| Eligibility scoring consistency | Varies by coordinator | Reproducible, weighted |
| Physician outreach | Ad-hoc, manual | Auto-drafted, structured |
| Annual trial recruitment gap | $1.8B cost to pharma | Addressable at scale |

Clinical trial participation rates in the US are below **5% of eligible cancer patients**. Systems like this, deployed at scale, could change that — and accelerate drug approval timelines that affect millions of patients globally.

---

## Extending the System

The agent pipeline is designed for extension:

```python
# Add a new agent by following the pattern:
class InsuranceCoverageAgent:
    """Checks whether the patient's insurance covers trial participation costs."""
    def run(self, patient: PatientProfile, scores: list[EligibilityScore]) -> list[EligibilityScore]:
        ...
```

Other extensions to try:
- **PubMedSearchAgent**: Pull recent publications to explain trial scientific rationale
- **EHR Integration**: Replace mock patients with FHIR-formatted patient data
- **Slack/Email Notifier**: Auto-send reports to care team channels
- **Feedback Loop**: Track which matches led to enrollment and tune weights

---

## Running Tests

```bash
python -m pytest test_main.py -v
```

All 28 tests should pass in under 5 seconds, with no external API calls required.

---

## Project Structure

```
clinical-trial-scout-agent/
├── main.py            # All agents + orchestrator + CLI
├── test_main.py       # 28 unit tests covering all agents + E2E
├── requirements.txt   # Minimal dependencies
├── .gitignore
└── README.md
```

---

## Data Sources

- **ClinicalTrials.gov v2 API** — Free, public, maintained by the U.S. National Library of Medicine. No API key required. Rate limits apply (respectful use only).
- Demo patient data is **entirely synthetic** — no real patient information is used.

---

## Contributing

Contributions welcome! Some ideas:

1. **Add FHIR patient input** — parse real HL7 FHIR bundles instead of mock objects
2. **Distance calculation** — replace state-level heuristics with actual ZIP-code geodistance
3. **Expand eligibility parsing** — use NLP to extract structured criteria from free-text
4. **Add more agent types** — PubMed citation agent, insurance coverage agent
5. **Build a web UI** — wrap the pipeline in a FastAPI endpoint

Please open an issue first to discuss major changes.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Research Sources

- Tufts Center for the Study of Drug Development: *Impact Report on Clinical Trial Productivity* (2022)
- ASCO: *Barriers to Clinical Trial Enrollment* survey (2024)
- NIH ClinicalTrials.gov v2 API documentation: https://clinicaltrials.gov/data-api/api
- AHRQ: *Patient-Centered Outcomes Research* (2023)

---

*Built with Python | No paid APIs | Works offline via realistic mock data*
