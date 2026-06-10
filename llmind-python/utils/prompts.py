SYSTEM_PROMPT = {
  "project": "The project aims to create a large-scale interactive media installation that engages the public and contributes to the cultural program of Aarhus 2017. The installation will activate the city's greenhouse dome, transforming it into a living canvas of light that symbolizes growth, cycles, and participation. At its core, the project seeks to enable citizens to interact with the greenhouse facade in ways that foster a sense of participation and ownership, while also exploring the intersection of interaction design, light art, and architecture through iterative prototyping and experimentation. In doing so, it reflects themes of sustainability, growth, and community in direct alignment with the ambitions of the European Capital of Culture program.",
  "system": "You are a creative professional designer and analytical thinker. You use concise expression with maximum information density that are highly based on facts and data."
}

IDEA_FIRST_PROMPT = """Here are the existing artefacts:
'''
{existing_artefacts}
'''
TASK:
Based on the existing artefacts, captures the salient components of its design space.
Ideate creative and concrete Aspects / Options to understand the design space.

DEFINITIONS:
- Design Space: a conceptual space, which encompasses the creativity constraints that govern what the outcome of the design process might (and might not) be.

Respond in the following format:
THOUGHT:
<Your reasoning: high-level design plan. Explain how your constraints sparks creativity, and what outcomes you aim to enable.>

NEW IDEA JSON:
```json
<JSON>
```

In <JSON>, provide the new ideas in JSON format with the following fields under the root "Taxonomy":
- Aspect: a key dimension or parameter of the design space (e.g., display technology, location, type of interaction).
- Option: possible alternatives for an aspect (e.g., for the aspect “Display,” options might include LED panels or projection).
For each Aspect, provide comprehensive Options. 
Nor futher nesting is allowed.

This JSON will be automatically parsed, so ensure the format is precise."""

IDEA_REFLECTION_PROMPT = """Round {current_round}/{num_reflections}.
In your thoughts, Consider:
- Aspects and options define the range of possible paths through the design space.
- They serve both as enablers of creative thinking (by outlining possibilities) and constraints (by ruling out alternatives when one option is chosen) 

TASK:
Refine and consolidate them into more meaningful ones while keeping the spirit of the original idea.
Both Aspects and Options should be concise and clear.

Respond in the same format as before:
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
In <JSON>, provide the refined ideas in JSON format that strictly follows the same structure as before.:

Ensure the idea is clear and the JSON format is correct."""


USER_PROMPT_TEMPLATE = """TASK:
Explore into the given Aspect in the current Taxonomy, suggest new simple Options that:
- define the range of possible paths through the design space.
- serve both as enablers of creative thinking (by outlining possibilities) and constraints (by ruling out alternatives when one option is chosen)

ASPECT TO EXPLORE:
ID: {{SELECTED_NODE_ID}}
Topic: {{SELECTED_NODE_TOPIC}}

RELATED PROJECTS:
{{RELATED_PROJECTS}}

TAXONOMY:
{{TAXONOMY}}

Respond in the following format
THOUGHT:
<THOUGHT>

NEW IDEA JSON:
```json
<JSON>
```
In <JSON> provide ID of the node that you explored and create new Node ids accordingly, and Options.
Suppose explored node is "Aspect", each of new Nodes is "Option":
- Aspect: a key dimension or parameter of the design space (e.g., display technology, location, type of interaction).
- Option: possible alternatives for an aspect (e.g., for the aspect "Display," options might include LED panels or projection).
the options should be concise and simple; each "desc" is ONE dense, specific sentence describing the option (it is embedded for retrieval — be concrete, not generic).

Use the following JSON structure exactly:
{
  "parent_id": "<SELECTED_NODE_ID>",
  "options": [
    { "id": "<new_node_id>", "topic": "<option label>", "desc": "<one-sentence description>" },
    { "id": "<new_node_id>", "topic": "<option label>", "desc": "<one-sentence description>" }
  ]
}"""


# Version tag logged with every generate-at call so prompt/seeding variants can
# be compared in data/projection/generate_log.jsonl. Bump when the prompt or
# the seeding strategy changes behaviour.
GENERATE_AT_PROMPT_VERSION = 2

GENERATE_AT_PROMPT = """TASK:
The designer clicked an UNOCCUPIED location on a 2D map of the design space — a
gap where no real project and no existing idea sits. Propose new design Options
that conceptually belong AT that location.

SURROUNDING REAL PROJECTS (the gap's neighbourhood — these bracket the clicked
location from different sides):
{{RELATED_PROJECTS}}

NEARBY EXISTING IDEAS (already on the map near the click — do NOT duplicate or
trivially rephrase any of these):
{{NEARBY_OPTIONS}}

PARENT ASPECT (the new Options will be filed under this dimension):
ID: {{SELECTED_NODE_ID}}
Topic: {{SELECTED_NODE_TOPIC}}

TAXONOMY (context only):
{{TAXONOMY}}

INSTRUCTIONS:
- Propose Options that sit conceptually BETWEEN the surrounding projects yet are
  distinct from every one of them: fill the gap, do not imitate the nearest
  project and do not average them into something generic.
- Each Option must be a plausible alternative within the parent Aspect.
- Each "desc" is ONE dense, specific sentence describing the option. It is
  embedded to position the idea on the map, so make it concrete: name the
  mechanism, material, interaction, or context that makes this option itself.
- 2 to 4 Options; concise topics (2-5 words).

Respond in the following format
THOUGHT:
<Why these options fill the gap between the surrounding projects.>

NEW IDEA JSON:
```json
<JSON>
```
Use the following JSON structure exactly:
{
  "parent_id": "<SELECTED_NODE_ID>",
  "options": [
    { "id": "<new_node_id>", "topic": "<option label>", "desc": "<one-sentence description>" }
  ]
}"""