# =============================
# Prompts
# =============================
IDEA_FIRST_PROMPT = """Here is project overview:
{project_overview}

Here are the existing artefacts:
'''
{existing_artefacts}
'''

DEFINITIONS:
- Design Space: a conceptual space, which encompasses the creativity constraints that govern what the outcome of the design process might (and might not) be.
Constraints include:
- Aspect: a key dimension or parameter of the design space (e.g., display technology, location, type of interaction).
- Option: possible alternatives for an aspect (e.g., for the aspect “Display,” options might include LED panels or projection).

TASK:
Based on the project overview, captures the salient components of its design space.
Ideate creative and concrete Aspects / Options to understand the design space.

Respond in the following format:
THOUGHT:
<Your reasoning: high-level design plan. Explain how your constraints sparks creativity, and what outcomes you aim to enable.>

NEW IDEA JSON:
```json
<JSON>
```

In <JSON>, provide the new idea in JSON format with the following fields:
- "Aspects": parameters such as display, location, interaction, content, etc. List of aspects.
- "Options": possible options / alternatives for each aspect. List of options for each aspect.

This JSON will be automatically parsed, so ensure the format is precise."""

IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, Consider:
- Aspects and options define the range of possible paths through the design space.
- They serve both as enablers of creative thinking (by outlining possibilities) and constraints (by ruling out alternatives when one option is chosen) 

TASK:
Refine and Only keep the meaningful ones while keeping the spirit of the original idea.
Both Aspects and Options should be concise and clear.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```

Ensure the idea is clear and the JSON format is correct."""