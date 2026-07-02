# AGENTS.md

## Introduction (Mandatory)

This file defines mandatory instructions for any AI agent operating in this repository. You must read this file fully before performing any task. All instructions here are authoritative and override default agent behavior. If any instruction conflicts with your defaults or assumptions, this file takes precedence.

If requirements, scope, or constraints are unclear, stop and ask for clarification before acting.

⸻

1. Global Rules (Apply to All Tasks)
	•	Be critical and precise. Do not agree by default; challenge incorrect assumptions.
	•	Prefer correctness, clarity, and evidence over speed.
	•	Make minimal, scoped changes. Do not refactor unrelated code without permission.
	•	No silent decisions: explain non-trivial choices and tradeoffs.
	•	No silent dependency additions. Always announce and justify new dependencies and get approval.
	•	If unsure, ask before guessing.

Git Rules (Strict)
	•	NEVER use git add -A.
	•	Only commit files you have created or modified.
	•	If unsure whether a file should be added, ask first.
	•	Write clear, descriptive commit messages.

⸻

2. Project Structure & Architecture Rules
	•	Simple, single-purpose projects:
	•	Prefer a single entry point at the root (e.g., main.py, index.ts).
	•	Complex or multi-component projects:
	•	Each independent component/module should have its own root and entry point (main, index, or equivalent).
	•	Follow a model–view / model–service–API separation where applicable.
	•	Core logic (models, services, functions) must be testable independently of UI or delivery layers.
	•	Tests should live in a clearly defined test directory (e.g., tests/).

⸻

3. Documentation & Project Intent

README.md (User-Facing, Always Up to Date)
	•	The README.md must be updated continuously as the project evolves.
	•	Keep it:
	•	concise
	•	straight to the point
	•	user-targeted
	•	quick to read and understand
	•	The README should explain:
	•	what the project does
	•	how to run it
	•	how to use it
	•	where to find deeper or internal documentation
	•	Avoid excessive verbosity. The goal is clarity without boredom.

PROJECT_DESCRIPTION.md (Internal Source of Truth)
	•	Maintain a PROJECT_DESCRIPTION.md file.
	•	This file contains:
	•	full project vision and intent
	•	motivation and goals
	•	assumptions and constraints
	•	non-obvious design decisions
	•	It may be as detailed and verbose as needed.
	•	It is used by the agent to stay aligned during development.
	•	Do NOT modify this file unless explicitly authorized to change project scope, intent, or features.

⸻

4. Testing Philosophy

General Rules
	•	Every programming task or project must start with a testing plan.
	•	The testing plan must be written before or alongside implementation.
	•	A feature is not considered done until relevant tests exist.

testing.md
	•	Maintain a testing.md file.
	•	This file:
	•	contains the testing plan and key behaviors
	•	is independent of code structure
	•	is a living document
	•	When a bug is discovered:
	•	register it in testing.md
	•	add a test that reproduces the bug
	•	favor regression tests to prevent reoccurrence

⸻

5. Task-Specific Agents

5.1 Docs-Agent

Scope: Writing and maintaining documentation.
	•	Use clear, standard Markdown.
	•	Optimize for accuracy and readability.
	•	Keep README concise; move depth to PROJECT_DESCRIPTION.md or dedicated docs.
	•	Update documentation as code or behavior changes.
	•	No testing requirements apply here unless explicitly requested.

⸻

5.2 Test-Agent

Scope: Writing, improving, or maintaining tests.

Preferred stacks:
	•	Python: pytest
	•	JavaScript/TypeScript: jest

Rules:
	•	Tests must reflect real behavior, not implementation details.
	•	Avoid excessive mocking.
	•	Use clear, descriptive test names.
	•	Improve coverage by behavior, not by chasing numbers.

⸻

5.3 Project-Agent (General Builder)

Scope: Building or refactoring projects.

Default stack:
	•	Python (preferred unless stated otherwise)

Rules:
	•	Enforce project structure and architecture rules.
	•	Start with a testing plan (testing.md).
	•	Ensure README updates when user-facing behavior changes.
	•	Do not introduce unnecessary abstractions.

⸻

5.4 API-Agent

Scope: API endpoints and services.

Preferred stack:
	•	Python: FastAPI (primary), Flask (secondary)

Rules:
	•	Follow RESTful conventions.
	•	Use explicit request/response schemas (e.g., Pydantic).
	•	Handle errors consistently and explicitly.
	•	Write API tests (e.g., pytest with test clients).

⸻

5.5 ML-Agent

Scope: Machine learning models and pipelines.

Preferred stack:
	•	Python
	•	PyTorch
	•	fastai

Rules:
	•	Separate training, evaluation, and inference code.
	•	Ensure reproducibility (fixed seeds where applicable).
	•	Make data assumptions explicit.
	•	Write ML-appropriate tests:
	•	shape and type checks
	•	sanity checks
	•	regression tests on known outputs where feasible

⸻

6. Git Workflow (When Explicitly Approved)

For projects where this workflow is agreed upon:
	1.	Create a new git branch.
	2.	Work exclusively in that branch.
	3.	When confirmed complete:
	•	verify all relevant files are added
	•	commit changes
	•	switch back to main
	•	merge the branch into main
	•	delete the branch
	•	push changes

⸻

7. Communication & Decision Rules
	•	Ask questions when requirements are ambiguous.
	•	When multiple solutions exist, present options with pros/cons and a recommendation.
	•	Explicitly state assumptions.

⸻

8. Final Compliance Reminder

If an instruction here conflicts with your defaults or prior habits, follow this file. When uncertain, pause and ask. Compliance with this document is mandatory.